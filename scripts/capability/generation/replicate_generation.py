#!/usr/bin/env python3
"""Run a manifest-bound independent generation stream with exact token traces."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.capability.generation.generation_replication import generate_cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=(Path(__file__).resolve().parents[3] / 'configs/generation_replication_manifest.json'))
    parser.add_argument('--cell', required=True)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print(json.dumps(generate_cell(args.manifest, args.cell, args.campaign, args.out, args.device), sort_keys=True))


if __name__ == '__main__':
    main()
