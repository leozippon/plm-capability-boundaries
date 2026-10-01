#!/usr/bin/env python3
"""Resolve a stage uniquely in a current tree or a retained flat snapshot."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


def stage_path(root: Path, name: str, revision: str | None = None) -> Path:
    if Path(name).name != name or '..' in name or not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError(f'invalid stage basename: {name}')
    names = {name, re.sub(r'^\d\d_', '', name)}
    if revision:
        paths = [root / line for line in subprocess.check_output(
            ['git', '-C', str(root), 'ls-tree', '-r', '--name-only', revision, '--', 'scripts/capability'],
            text=True).splitlines()]
    else:
        base = root / 'scripts/capability'
        paths = [path for pattern in ['*', '*/*'] for path in base.glob(pattern) if path.is_file()]
    matches = [path for path in paths if path.name in names or re.sub(r'^\d\d_', '', path.name) in names]
    if len(matches) != 1:
        raise ValueError(f'stage {name} has {len(matches)} matches in {root}; expected exactly one')
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--revision')
    args = parser.parse_args()
    try:
        print(stage_path(args.root, args.name, args.revision))
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
