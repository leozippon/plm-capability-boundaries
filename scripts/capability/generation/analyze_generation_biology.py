#!/usr/bin/env python3
"""Analyze complete generation/structure records without loading a model."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.generation.generation_biology_analysis import analyze, merge_reference_annotations, read_jsonl, write_figures, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "main"), required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--reference-sidecar", type=Path)
    parser.add_argument("--primary-condition", default="requested")
    parser.add_argument("--uncertainty-unit", choices=("class", "sequence_cluster"), default="class")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    attempts, subset = read_jsonl(args.attempts), read_jsonl(args.subset)
    if args.reference_sidecar:
        annotations = read_jsonl(args.reference_sidecar)
        attempts = merge_reference_annotations(attempts, annotations)
        subset = merge_reference_annotations(subset, annotations)
    result = analyze(attempts, subset, read_jsonl(args.predictions),
                     phase=args.phase, calibration=calibration,
                     primary_condition=args.primary_condition, uncertainty_unit=args.uncertainty_unit)
    sources = [args.attempts, args.subset, args.predictions]
    if args.calibration:
        sources.append(args.calibration)
    if args.reference_sidecar:
        sources.append(args.reference_sidecar)
    result["input_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    write_report(result, args.out)
    write_figures(result, args.out)
    print(json.dumps({"phase": args.phase, "arms": result["arms"], "out": str(args.out)}, allow_nan=False))


if __name__ == "__main__":
    main()
