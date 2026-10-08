"""Locating the frozen single-mutant stability gate's own bytes.

The gate's inputs -- the cohort, the control qualification frozen before any
model quantity was read, the mutation-local profiles, the extraction plan and the
published panel -- exist in two directory layouts at once, and both are in
current use.  The lane that produced them kept them flat in one measurement
directory, which is what the archived dispatch logs and the 2026-09-27 likelihood
replay both read; the published result tree groups them into ``cohort/``,
``controls/``, ``endpoint/`` and ``fits/`` subdirectories.

A stage that hardcodes one layout works on one side of that and fails on the
other, and the failure lands at the start of a multi-hour cell rather than
anywhere useful.  Resolving instead of assuming is therefore not convenience: the
declared digest still binds the bytes, so a resolver that found the wrong file
fails exactly as loudly as a wrong literal path would, while a resolver that
found the right one in the other layout simply works.

Resolution is deliberately shallow -- a named directory and one level below it,
nothing recursive -- and a name that matches twice is a refusal rather than a
choice, because two cohorts under one gate directory is a staging fault and
picking one of them silently is how an incomparable number gets published.

Several candidate directories may be offered, and ones that do not exist are
passed over.  That is what lets one campaign manifest name both layouts without
anybody having to inspect the cluster first; it is not a licence to guess,
because the digest gate downstream still binds the bytes and two candidates that
both exist are still a refusal.
"""
from __future__ import annotations

from pathlib import Path

#: Every frozen gate input a stability stage reads, by basename.
GATE_INPUTS: tuple[str, ...] = (
    'cohort.json', 'controls_qualification.json', 'profile_features.npz',
    'extraction_plan.json', 'panel.json',
)


def resolve(gate_dirs, names=GATE_INPUTS) -> dict[str, Path]:
    """Each named input, found exactly once across the candidate directories."""

    if isinstance(gate_dirs, (str, Path)):
        gate_dirs = [gate_dirs]
    directories = [Path(item) for item in gate_dirs]
    if not directories:
        raise ValueError('no gate directory was declared')
    present = [directory for directory in directories if directory.is_dir()]
    if not present:
        raise ValueError('none of the declared gate directories exists: '
                         f'{[str(item) for item in directories]}')
    unknown = [name for name in names if name not in GATE_INPUTS]
    if unknown:
        raise ValueError(f'not declared gate inputs: {unknown}')
    resolved: dict[str, Path] = {}
    for name in names:
        matches = sorted({path.resolve() for directory in present for path in
                          [*directory.glob(name), *directory.glob(f'*/{name}')]
                          if path.is_file()})
        if len(matches) != 1:
            raise ValueError(
                f'{name}: {len(matches)} candidates at or one level below '
                f'{[str(item) for item in present]}; expected exactly one. '
                f'Found: {[str(path) for path in matches]}')
        resolved[name] = matches[0]
    return resolved


__all__ = ['GATE_INPUTS', 'resolve']
