#!/usr/bin/env python3
"""Compare the Results numeric tokens of two manuscript revisions.

The manuscript README states that the rewrite kept every numeric token of the
Results in its original order. That is a checkable statement about two
revisions of the source, and this is the check. It reads each revision's
Results prose, extracts the numbers with the rule the evidence checker reads
the manuscript by, and reports whether the two ordered sequences are the same,
where they first differ and every edit between them.

A revision keeps its Results wherever it keeps them. The current submission
inputs one file per Results group from ``sections/``; the revision before the
rewrite held the whole document in ``main.tex``, so its Results are the region
between ``\\section{Results}`` and the next ``\\section``. Neither layout is
assumed and neither is guessed at: a revision whose Results cannot be located
is a refusal rather than an empty sequence, which would otherwise compare equal
to nothing and report a preserved order that was never read.

Numbers are what ``check_manuscript_evidence`` calls numbers, so the two
checkers cannot disagree about what a numeric token is. Checkpoint names carry
digits --- ProGen3-3B, Qwen2.5-0.5b --- and those digits are not quantities, so
they are blanked before the numbers are read.

    python scripts/capability/reporting/check_source_integrity.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import re
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.capability.reporting import check_manuscript_evidence as evidence  # noqa: E402
from scripts.capability.reporting import publication_paths  # noqa: E402
from scripts.capability.reporting.validate_manuscript import without_comments  # noqa: E402

#: Where the rewrite's own review artifacts live, and what this one is called.
REVIEW = Path('archive/reviews/evolutionary-memory-20261102')
OUTPUT = REVIEW / 'source-integrity.json'
#: The revision the rewrite replaced, snapshotted before the submission ZIP was
#: rebuilt. ``manuscript/`` is outside Git, so a snapshot is the only record of
#: what the Results said before.
PREVIOUS = Path('archive/manuscript-snapshots/20261102T1105-before-zip-replacement/submission')

RESULTS_INPUT = re.compile(r'\\input\{(sections/results-[^}]*)\}')
RESULTS_HEADING = r'\section{Results}'
#: How much of the stripped prose to quote around a token. Enough to find the
#: sentence in the source, short enough that an edit list stays readable.
CONTEXT = 90


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def results_region(main: str) -> str:
    """The Results of a single-file revision, as one source."""
    start = main.find(RESULTS_HEADING)
    if start < 0:
        raise ValueError(f'no {RESULTS_HEADING} to read the Results prose from')
    rest = main[start + len(RESULTS_HEADING):]
    end = rest.find('\\section')
    return rest if end < 0 else rest[:end]


def results_sources(submission: Path) -> tuple[str, list[tuple[str, str]]]:
    """One revision's Results layout and its prose, as (name, text) in reading order."""
    main = without_comments((submission / 'main.tex').read_text())
    names = RESULTS_INPUT.findall(main)
    if names:
        sources = []
        for name in names:
            path = submission / name
            if not path.suffix:
                path = path.with_suffix('.tex')
            sources.append((path.relative_to(submission).as_posix(), path.read_text()))
        return 'modular', sources
    return 'monolithic', [('main.tex', results_region(main))]


def numeric_tokens(text: str) -> list[tuple[str, str]]:
    """The ordered numeric tokens of one source, each with its surrounding prose."""
    body = evidence.strip_markup(text)
    numeric = evidence.NAME.sub(lambda match: ' ' * len(match.group(0)), body)
    tokens = []
    for match in evidence.NUMBER.finditer(numeric):
        token = match.group(1).rstrip(',')
        if not token or token in '+-':
            continue
        window = body[max(0, match.start() - CONTEXT):match.start() + CONTEXT]
        tokens.append((token, ' '.join(window.split())))
    return tokens


def revision(submission: Path) -> dict:
    """One revision's Results token sequence, per file and in total."""
    layout, sources = results_sources(submission)
    files = []
    sequence: list[tuple[str, str, str]] = []
    for name, text in sources:
        tokens = numeric_tokens(text)
        path = submission / name
        files.append({'file': name, 'tokens': len(tokens),
                      'sha256': digest(path) if path.is_file() else None,
                      'sequence': [token for token, _ in tokens]})
        sequence.extend((token, context, name) for token, context in tokens)
    if not sequence:
        raise ValueError(f'{submission}: the Results carry no numeric token to compare')
    return {'submission': submission.as_posix(), 'layout': layout,
            'results_files': len(files), 'tokens': len(sequence), 'files': files,
            '_sequence': sequence}


def per_file_comparison(previous: dict, current: dict) -> dict:
    """File-by-file verdicts, or why the two revisions cannot be compared that way."""
    before = {entry['file']: entry['sequence'] for entry in previous['files']}
    after = {entry['file']: entry['sequence'] for entry in current['files']}
    shared = sorted(set(before) & set(after))
    if not shared:
        return {'comparable': False,
                'reason': f"the revisions partition the Results differently: "
                          f"{previous['layout']} {sorted(before)} against "
                          f"{current['layout']} {sorted(after)}, so no file has a counterpart "
                          f"and only the whole Results sequence can be compared",
                'previous_only': sorted(set(before) - set(after)),
                'current_only': sorted(set(after) - set(before))}
    return {'comparable': True,
            'files': [{'file': name, 'identical': before[name] == after[name],
                       'previous_tokens': len(before[name]), 'current_tokens': len(after[name])}
                      for name in shared],
            'previous_only': sorted(set(before) - set(after)),
            'current_only': sorted(set(after) - set(before))}


def edits(previous: list[tuple[str, str, str]], current: list[tuple[str, str, str]]) -> list[dict]:
    """Every difference between the two sequences, in order, with its context."""
    before = [token for token, _, _ in previous]
    after = [token for token, _, _ in current]
    found = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=before, b=after,
                                                       autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        record = {'edit': tag, 'previous_index': i1, 'current_index': j1,
                  'previous_tokens': before[i1:i2], 'current_tokens': after[j1:j2]}
        if i1 < len(previous):
            record['previous_file'] = previous[i1][2]
            record['previous_context'] = previous[i1][1]
        if j1 < len(current):
            record['current_file'] = current[j1][2]
            record['current_context'] = current[j1][1]
        found.append(record)
    return found


def compare(previous_submission: Path, current_submission: Path) -> dict:
    """The comparison report, as the artifact records it."""
    previous, current = revision(previous_submission), revision(current_submission)
    difference = edits(previous['_sequence'], current['_sequence'])
    retained = sum(block.size for block in
                   difflib.SequenceMatcher(a=[token for token, _, _ in previous['_sequence']],
                                           b=[token for token, _, _ in current['_sequence']],
                                           autojunk=False).get_matching_blocks())
    first = difference[0] if difference else None
    return {
        'schema': 'manuscript_results_source_integrity_v1',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'claim': 'every numeric token of the Results files retains its original order',
        'number_rule': 'check_manuscript_evidence.strip_markup with checkpoint names blanked; '
                       'LaTeX geometry and template markup are not evidence and are removed '
                       'before any number is read',
        'previous_revision': {key: value for key, value in previous.items() if key != '_sequence'},
        'current_revision': {key: value for key, value in current.items() if key != '_sequence'},
        'sequence_identical': not difference,
        'retained_tokens_in_order': retained,
        'tokens_only_in_previous': sum(len(entry['previous_tokens']) for entry in difference),
        'tokens_only_in_current': sum(len(entry['current_tokens']) for entry in difference),
        'first_divergence': first,
        'edits': difference,
        'per_file': per_file_comparison(previous, current),
        'status': 'identical' if not difference else 'diverged',
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--current', type=Path, default=REPO / publication_paths.SUBMISSION,
                        help='the submission directory of the current revision')
    parser.add_argument('--previous', type=Path, default=REPO / PREVIOUS,
                        help='the submission directory of the revision it replaced')
    parser.add_argument('--output', type=Path, default=REPO / OUTPUT,
                        help='where to write the comparison artifact')
    options = parser.parse_args(argv)
    report = compare(options.previous, options.current)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: report[key] for key in
                      ('status', 'sequence_identical', 'retained_tokens_in_order',
                       'tokens_only_in_previous', 'tokens_only_in_current',
                       'first_divergence')}, indent=2, ensure_ascii=False))
    print(f'previous: {report["previous_revision"]["tokens"]} tokens in '
          f'{report["previous_revision"]["results_files"]} '
          f'{report["previous_revision"]["layout"]} Results source(s)')
    print(f'current:  {report["current_revision"]["tokens"]} tokens in '
          f'{report["current_revision"]["results_files"]} '
          f'{report["current_revision"]["layout"]} Results source(s)')
    print(f'wrote {options.output}')
    return 0 if report['status'] == 'identical' else 1


if __name__ == '__main__':
    raise SystemExit(main())
