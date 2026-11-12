"""What a deletion-hold check has to do to be worth running before a sweep.

Two provenance losses came from a pattern-scoped sweep that could not see an
individually enumerated hold, so the conditions pinned here are the ones that
make the check load-bearing: a path named as held is refused by name, a sweep
that names nothing held passes, and a check that cannot read its hold documents
refuses instead of reporting a clean set. The last matters most -- a guard that
degrades to "no holds found" when a document moves is worse than no guard,
because the sweep then reports that it consulted the holds.
"""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ops import check_deletion_holds as holds

ALLOWLIST = """# Repository deletion allowlist

## 3. Allowlist

| `scripts/capability/reporting/__pycache__/palette.cpython-311.pyc` | 2,332 | regenerates |

## 4. Held list

### 4.6 No confirmed producer regenerates them

| `scripts/capability/reporting/__pycache__/statistical_ladder_quantities.cpython-311.pyc` | 9,757 | orphan bytecode |

The directory `results/extensions/statistical_ladder_20261106/potts-v2/work` is held whole.
"""

HANDOFF = """# Hand-off

## Pending work with clocks on it

- A piece of orphaned bytecode is being held, not removed: `logs/operations/orphan-note.txt`.
"""


def documents(root: Path, allowlist: str = ALLOWLIST, handoff: str = HANDOFF) -> Path:
    (root / 'docs').mkdir(parents=True, exist_ok=True)
    (root / holds.HOLD_DOCUMENTS[0]).write_text(allowlist)
    (root / holds.HOLD_DOCUMENTS[1]).write_text(handoff)
    return root


def test_a_named_hold_stops_the_sweep_that_names_nothing_else_held(tmp_path):
    root = documents(tmp_path)
    sweep = ['scripts/capability/reporting/__pycache__/palette.cpython-311.pyc',
             'tests/reporting/__pycache__/test_palette.cpython-311-pytest-9.0.2.pyc']
    assert holds.refuse_held_paths(sweep, root) == 2
    assert holds.refuse_held_paths([root / sweep[0]], root) == 1
    orphan = 'scripts/capability/reporting/__pycache__/statistical_ladder_quantities.cpython-311.pyc'
    with pytest.raises(SystemExit, match=f'REFUSED: 1 of 3 .*\n  {orphan} -- {holds.HOLD_DOCUMENTS[0]}:11'):
        holds.refuse_held_paths([*sweep, './' + orphan], root)
    with pytest.raises(SystemExit, match=f'logs/operations/orphan-note.txt -- {holds.HOLD_DOCUMENTS[1]}:5'):
        holds.refuse_held_paths(['logs/operations/orphan-note.txt'], root)
    # A hold on a directory is not a hold on what the sweep finds inside it: the
    # documents name directories in passing, so containment would read those as
    # holds over whole trees. The tree's own path is still refused.
    inside = 'results/extensions/statistical_ladder_20261106/potts-v2/work/fit.json'
    assert holds.refuse_held_paths([inside], root) == 1
    with pytest.raises(SystemExit, match='potts-v2/work --'):
        holds.refuse_held_paths([str(Path(inside).parent)], root)


def test_the_check_refuses_rather_than_passing_when_it_cannot_do_its_job(tmp_path):
    root = documents(tmp_path)
    with pytest.raises(SystemExit, match='no candidate paths were given'):
        holds.refuse_held_paths([], root)
    with pytest.raises(SystemExit, match='lies outside'):
        holds.refuse_held_paths(['/etc/hosts'], root)
    with pytest.raises(SystemExit, match='is not a path inside'):
        holds.refuse_held_paths(['../outside/cache.pyc'], root)
    (root / holds.HOLD_DOCUMENTS[1]).unlink()
    with pytest.raises(SystemExit, match=f'cannot read the hold document {holds.HOLD_DOCUMENTS[1]}'):
        holds.refuse_held_paths(['scripts/x/__pycache__/y.cpython-311.pyc'], root)
    documents(root, handoff='# Hand-off\n\n## Items\n\nNothing is held here, and no path is named.\n')
    with pytest.raises(SystemExit, match='no held path could be parsed'):
        holds.refuse_held_paths(['scripts/x/__pycache__/y.cpython-311.pyc'], root)


def test_the_live_hold_documents_still_refuse_the_path_that_was_destroyed():
    if not all((ROOT / name).is_file() for name in holds.HOLD_DOCUMENTS):
        pytest.skip("the hold documents live under the gitignored docs/ and are not distributed")
    orphan = 'scripts/capability/reporting/__pycache__/statistical_ladder_quantities.cpython-311.pyc'
    parsed = holds.read_holds(ROOT)
    assert orphan in parsed and parsed[orphan].document == holds.HOLD_DOCUMENTS[0]
    command = [sys.executable, str(ROOT / 'scripts/ops/check_deletion_holds.py'), '-']
    refused = subprocess.run(command, input=f'{orphan}\n', text=True, capture_output=True)
    assert refused.returncode != 0 and orphan in refused.stderr
    cleared = subprocess.run(command, input='scripts/capability/reporting/__pycache__/palette.cpython-311.pyc\n',
        text=True, capture_output=True)
    assert cleared.returncode == 0 and 'none is held' in cleared.stdout
