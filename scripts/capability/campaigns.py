#!/usr/bin/env python3
"""List or extract a historical campaign manifest without changing its contents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = REPO_ROOT / 'h200/history/campaigns.jsonl'


def records(archive: Path = ARCHIVE) -> dict[str, dict[str, str]]:
    if not archive.is_file():
        raise SystemExit(
            f'no historical recipe archive at {archive}. It records queue slots and their '
            'cluster paths, so it is host-local under h200/ and is not distributed with the '
            'repository; a clone cannot recover a historical campaign from this entry point.')
    result = {}
    for line in archive.read_text().splitlines():
        record = json.loads(line)
        name = Path(record['original_path']).name
        if name in result:
            raise ValueError(f'duplicate archived manifest: {name}')
        result[name] = record
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    extract = commands.add_parser('extract')
    extract.add_argument('name', help='original campaign TSV basename')
    extract.add_argument('--out', type=Path, required=True, help='new output file; never overwrite an existing file')
    args = parser.parse_args()
    archive = records()
    if args.command == 'list':
        print('\n'.join(sorted(archive)))
        return
    if args.name not in archive:
        parser.error(f'unknown historical manifest: {args.name}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(archive[args.name]['content'])
    print(args.out)


if __name__ == '__main__':
    main()
