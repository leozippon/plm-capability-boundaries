"""The manuscript package carries every module its entry points import.

A packaged entry point that imports a module the package does not hold fails on
import and reproduces nothing, which is how two ledger source mappings went
missing: ``build_derived_numbers.build`` imports its emitters inside the
function, so nobody maintaining the package by hand saw them appear.

The second group of tests fixes where staging must fail and where it must not.
A source that should be in the repository and is not is a fault; a harness
scratch source that has been cleaned is not, because the packaged copy holds
those bytes and its own digest stands for them. What stays a fault there is a
scratch source still present and no longer the bytes the package was staged
from, and a packaged copy that is gone with only such a source behind it.
"""
from pathlib import Path
import hashlib
import json
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


def test_manifest_schema_accepts_the_written_id_and_the_one_it_replaced():
    assert packager.SCHEMA == 'compact-manuscript-handoff'
    assert packager.accepted_schema('compact-manuscript-handoff')
    assert packager.accepted_schema('compact-manuscript-handoff-v1')
    for unknown in (None, '', 'compact-manuscript-handoff-v2',
                    'compact_manuscript_handoff', ' compact-manuscript-handoff'):
        assert not packager.accepted_schema(unknown)


def test_the_staged_manifest_is_still_read_by_this_packager():
    package = staged_package()
    previous = json.loads((package / packager.MANIFEST_NAME).read_text())
    assert packager.accepted_schema(previous['schema'])


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


def test_a_copy_of_the_same_size_is_still_compared_by_its_bytes(tmp_path):
    """Staging compares what the two files carry, not how large they are."""
    derived, _ = packager.derived_code()
    for path, source in derived.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((packager.REPO_ROOT / source).read_bytes())
    path = f'{ANALYSIS}/reporting/palette.py'
    source = derived[path]
    original = (packager.REPO_ROOT / source).read_bytes()
    changed = bytes([original[0] ^ 0x01]) + original[1:]
    assert len(changed) == len(original)
    (tmp_path / path).write_bytes(changed)

    assert packager.missing_modules(derived, tmp_path) == [f'{path}: differs from {source}']


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


def test_submitted_figure_count_comes_from_the_document_including_its_sections(tmp_path):
    # The compiled PDFs are checked against the figures the manuscript includes,
    # which only the assembled sources say. A count taken from a code constant
    # described whichever figures some renderer happened to name.
    submission = tmp_path / 'submission'
    (submission / 'sections').mkdir(parents=True)
    (submission / 'main.tex').write_text('\\input{sections/results}\n')
    (submission / 'sections/results.tex').write_text(
        r'\includegraphics[width=\linewidth]{figures/one.pdf}' '\n'
        r'% \includegraphics{figures/commented-out.pdf}' '\n'
        r'\includegraphics{figures/two.pdf}' '\n')
    (submission / 'supplementary-information.tex').write_text(
        r'\includegraphics{figures/three.pdf}')
    assert packager.included_figures(tmp_path) == {'main': 2, 'supplementary-information': 1}

    (submission / 'main.tex').write_text('\\section{Results}\n')
    with pytest.raises(SystemExit, match='no included figure found'):
        packager.included_figures(tmp_path)


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


RETAINED = b'attempt,outcome\n1,admitted\n'
SUPPLEMENT = 'data/generation-details/attempts.csv.gz'
SCRATCH = '.pi/tasks/detail-data-20261005/generation-details/attempts.csv.gz'


@pytest.fixture
def harness_staged(tmp_path, monkeypatch):
    """A packaged copy whose only recorded source is a harness scratch path."""
    scratch = tmp_path / SCRATCH
    scratch.parent.mkdir(parents=True)
    scratch.write_bytes(RETAINED)
    packaged = tmp_path / 'manuscript' / SUPPLEMENT
    packaged.parent.mkdir(parents=True)
    packaged.write_bytes(RETAINED)
    monkeypatch.setattr(packager, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(packager, 'PACKAGE', tmp_path / 'manuscript')
    return {'scratch': scratch, 'packaged': packaged,
            'sources': {SUPPLEMENT: (SCRATCH, None)},
            'recorded': {SUPPLEMENT: hashlib.sha256(RETAINED).hexdigest()}}


def test_an_ephemeral_source_that_is_gone_leaves_the_entry_exactly_as_staged(harness_staged):
    harness_staged['scratch'].unlink()
    problems: list[str] = []
    staged, digests = packager.stage(harness_staged['sources'], harness_staged['recorded'],
                                     True, problems)

    assert problems == [] and staged == []
    entry, = packager.manifest_entries([SUPPLEMENT], harness_staged['sources'], digests)
    assert entry == {'path': SUPPLEMENT, 'bytes': len(RETAINED),
                     'sha256': hashlib.sha256(RETAINED).hexdigest(),
                     'source': SCRATCH, 'source_state': 'ephemeral',
                     'source_sha256': harness_staged['recorded'][SUPPLEMENT]}


def test_an_ephemeral_source_that_changed_is_a_fault_and_is_never_restaged_from(harness_staged):
    harness_staged['scratch'].write_bytes(b'attempt,outcome\n1,refused\n')
    problems: list[str] = []
    staged, _ = packager.stage(harness_staged['sources'], harness_staged['recorded'],
                               True, problems)

    assert staged == [] and harness_staged['packaged'].read_bytes() == RETAINED
    assert len(problems) == 1
    assert problems[0].startswith(SUPPLEMENT) and 'no longer the bytes' in problems[0]


def test_a_copy_that_rotted_under_a_cleaned_source_is_caught_by_its_own_digest(harness_staged):
    harness_staged['scratch'].unlink()
    harness_staged['packaged'].write_bytes(b'attempt,outcome\n')
    problems: list[str] = []
    _, digests = packager.stage(harness_staged['sources'], harness_staged['recorded'],
                                True, problems)

    # A source that is gone can say nothing, so the recomputed entry is what
    # catches it: its sha256 no longer matches the one the manifest records,
    # which is what --check compares and fails on.
    assert problems == []
    entry, = packager.manifest_entries([SUPPLEMENT], harness_staged['sources'], digests)
    assert entry['sha256'] != hashlib.sha256(RETAINED).hexdigest()
    assert entry['source_sha256'] == harness_staged['recorded'][SUPPLEMENT]


def test_a_packaged_copy_lost_under_an_ephemeral_source_cannot_be_recovered(harness_staged):
    harness_staged['packaged'].unlink()
    problems: list[str] = []
    packager.stage(harness_staged['sources'], harness_staged['recorded'], True, problems)

    assert len(problems) == 1
    assert problems[0].startswith(SUPPLEMENT) and 'nothing can restage it' in problems[0]


def test_a_repository_source_that_is_gone_is_still_a_fault(harness_staged):
    sources = {'data/figure-validation.json': ('scripts/capability/reporting/gone.json', None)}
    problems: list[str] = []
    _, digests = packager.stage(sources, {}, True, problems)

    assert digests == {}
    assert problems == ['data/figure-validation.json: recorded source is gone: '
                        'scripts/capability/reporting/gone.json']


def test_adapter_stages_only_modules_the_package_carries():
    package = staged_package()
    derived, _ = packager.derived_code()
    for name in packager.adapter_modules(package):
        flat = f'code/{name}'
        assert flat in derived or flat in packager.PACKAGE_ONLY_CODE
        assert (package / flat).is_file()
