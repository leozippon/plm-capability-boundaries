"""What "the Results keep their numbers in order" has to mean to be checkable.

The claim the check exists for is a statement about two revisions of the source,
so the two things that can quietly make it vacuous are both pinned here: a
revision whose Results are not found must refuse rather than contribute an empty
sequence that compares equal to anything, and a number must be a quantity rather
than a digit inside a checkpoint name or a LaTeX length.
"""
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capability.reporting import check_source_integrity as integrity

MONOLITHIC = (r'\documentclass[11pt]{sn-jnl}'
              '\n' r'\includegraphics[width=0.8\textwidth]{figures/figure-1-framework.pdf}'
              '\n' r'\begin{document}'
              '\n' 'The abstract quotes 999 before the Results begin.'
              '\n' r'\section{Results}'
              '\n' r'\subsection{First}'
              '\n' 'ProGen3-3B gains 0.0142 TM across 8 domains.'
              '\n' r'\section{Discussion}'
              '\n' 'The discussion quotes 777 after the Results end.')


def submission(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_a_token_is_a_quantity_not_a_checkpoint_digit_or_a_latex_length():
    tokens = [token for token, _ in integrity.numeric_tokens(
        r'ProGen3-3B and Qwen2.5-0.5b reach 0.0142 TM on 1,024 of 25,728 variants'
        r' in a 0.8\textwidth panel\label{fig:x}\cite{ref2024}.')]
    assert tokens == ['0.0142', '1,024', '25,728']
    # The context travels with the token, because an edit list that names a
    # number without its sentence cannot be acted on.
    _, context = integrity.numeric_tokens('ProtGPT2 reaches 0.0142 TM.')[0]
    assert '0.0142' in context


def test_modular_results_are_read_in_the_order_main_inputs_them(tmp_path):
    root = submission(tmp_path, {
        'main.tex': (r'\begin{document}\input{sections/frontmatter}'
                     '\n' r'\input{sections/results-second}'
                     '\n' r'\input{sections/results-first}'
                     '\n' r'% \input{sections/results-dropped}'
                     '\n' r'\input{sections/methods}'),
        'sections/frontmatter.tex': 'Front matter quotes 111.',
        'sections/results-second.tex': 'Second says 22.',
        'sections/results-first.tex': 'First says 33.',
        'sections/results-dropped.tex': 'Dropped says 44.',
        'sections/methods.tex': 'Methods says 555.'})
    layout, sources = integrity.results_sources(root)
    assert layout == 'modular'
    assert [name for name, _ in sources] == ['sections/results-second.tex',
                                             'sections/results-first.tex']
    record = integrity.revision(root)
    assert [entry['sequence'] for entry in record['files']] == [['22'], ['33']]
    assert record['tokens'] == 2 and record['results_files'] == 2
    # Only the Results contribute: front matter, Methods and a commented-out
    # input are not Results prose.
    assert [token for token, _, _ in record['_sequence']] == ['22', '33']


def test_a_single_file_revision_is_read_from_its_results_region(tmp_path):
    root = submission(tmp_path, {'main.tex': MONOLITHIC})
    layout, sources = integrity.results_sources(root)
    assert layout == 'monolithic' and [name for name, _ in sources] == ['main.tex']
    record = integrity.revision(root)
    # 999 precedes the heading, 777 follows the next section, the included
    # figure's 0.8 is geometry and 3B is part of a checkpoint name.
    assert [token for token, _, _ in record['_sequence']] == ['0.0142', '8']
    assert record['files'][0]['sha256'] == integrity.digest(root / 'main.tex')


def test_a_revision_whose_results_cannot_be_read_refuses(tmp_path):
    nothing = submission(tmp_path / 'nothing', {'main.tex': r'\begin{document}\section{Methods}'})
    with pytest.raises(ValueError, match=r'\\section\{Results\}'):
        integrity.results_sources(nothing)
    empty = submission(tmp_path / 'empty', {
        'main.tex': r'\begin{document}\input{sections/results-one}',
        'sections/results-one.tex': 'Prose with no quantity at all.'})
    with pytest.raises(ValueError, match='no numeric token'):
        integrity.revision(empty)


def test_an_unchanged_results_sequence_is_reported_identical(tmp_path):
    before = submission(tmp_path / 'before', {'main.tex': MONOLITHIC})
    after = submission(tmp_path / 'after', {'main.tex': MONOLITHIC})
    report = integrity.compare(before, after)
    assert report['sequence_identical'] and report['status'] == 'identical'
    assert report['edits'] == [] and report['first_divergence'] is None
    assert report['retained_tokens_in_order'] == 2
    assert report['tokens_only_in_previous'] == report['tokens_only_in_current'] == 0


def test_a_changed_or_dropped_token_is_named_with_its_file_and_context(tmp_path):
    before = submission(tmp_path / 'before', {'main.tex': MONOLITHIC})
    after = submission(tmp_path / 'after', {
        'main.tex': r'\begin{document}\input{sections/results-first}',
        'sections/results-first.tex': 'ProGen3-3B gains 0.0151 TM.'})
    report = integrity.compare(before, after)
    assert not report['sequence_identical'] and report['status'] == 'diverged'
    first = report['first_divergence']
    assert first['edit'] == 'replace'
    # The changed estimate and the dropped domain count are one replacement of
    # the whole run: both old tokens are named, against the one that replaced
    # them, so nothing is silently omitted.
    assert first['previous_tokens'] == ['0.0142', '8']
    assert first['current_tokens'] == ['0.0151']
    assert first['previous_file'] == 'main.tex'
    assert first['current_file'] == 'sections/results-first.tex'
    assert '0.0142' in first['previous_context'] and '0.0151' in first['current_context']
    assert report['edits'] == [first]
    assert report['tokens_only_in_previous'] == 2 and report['tokens_only_in_current'] == 1
    assert report['retained_tokens_in_order'] == 0
    # The revisions partition the Results differently, so a per-file verdict
    # would be an invention; the reason says so instead.
    assert report['per_file']['comparable'] is False
    assert 'monolithic' in report['per_file']['reason']


def test_per_file_verdicts_are_given_where_the_files_correspond(tmp_path):
    main = r'\begin{document}\input{sections/results-a}\input{sections/results-b}'
    before = submission(tmp_path / 'before', {'main.tex': main,
                                              'sections/results-a.tex': 'A says 11.',
                                              'sections/results-b.tex': 'B says 22.'})
    after = submission(tmp_path / 'after', {'main.tex': main,
                                            'sections/results-a.tex': 'A says 11.',
                                            'sections/results-b.tex': 'B says 23.'})
    per_file = integrity.compare(before, after)['per_file']
    assert per_file['comparable'] is True
    assert per_file['files'] == [
        {'file': 'sections/results-a.tex', 'identical': True,
         'previous_tokens': 1, 'current_tokens': 1},
        {'file': 'sections/results-b.tex', 'identical': False,
         'previous_tokens': 1, 'current_tokens': 1}]


def test_the_artifact_is_written_and_the_exit_code_follows_the_verdict(tmp_path):
    before = submission(tmp_path / 'before', {'main.tex': MONOLITHIC})
    after = submission(tmp_path / 'after', {'main.tex': MONOLITHIC})
    output = tmp_path / 'review/source-integrity.json'
    arguments = ['--previous', str(before), '--current', str(after), '--output', str(output)]
    assert integrity.main(arguments) == 0
    report = json.loads(output.read_text())
    assert report['schema'] == 'manuscript_results_source_integrity_v1'
    assert report['previous_revision']['tokens'] == report['current_revision']['tokens'] == 2
    assert '_sequence' not in report['previous_revision']

    (after / 'main.tex').write_text(MONOLITHIC.replace('0.0142', '0.0151'))
    assert integrity.main(arguments) == 1
    assert json.loads(output.read_text())['status'] == 'diverged'


def test_the_two_real_revisions_are_located_and_add_up():
    current = ROOT / 'manuscript/submission'
    previous = ROOT / integrity.PREVIOUS
    if not (current / 'main.tex').is_file() or not (previous / 'main.tex').is_file():
        pytest.skip('the manuscript or its pre-rewrite snapshot is not in this checkout')
    report = integrity.compare(previous, current)
    for key in ('previous_revision', 'current_revision'):
        record = report[key]
        assert record['tokens'] == sum(entry['tokens'] for entry in record['files'])
        assert record['results_files'] == len(record['files']) >= 1
        assert record['tokens'] > 0
    assert report['current_revision']['layout'] == 'modular'
    assert [entry['file'] for entry in report['current_revision']['files']] == [
        'sections/results-mutation.tex', 'sections/results-phenotypes-interactions.tex',
        'sections/results-context-generation.tex', 'sections/results-guidance-selection.tex']
    assert report['previous_revision']['layout'] == 'monolithic'
    assert report['sequence_identical'] == (report['status'] == 'identical')
    assert report['sequence_identical'] == (report['edits'] == [])
