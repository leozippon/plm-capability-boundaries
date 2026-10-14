"""Calibrate the E20 primary endpoint against its two fixed points.

A TM-score is only interpretable against the scale it lives on, and E20's
primary endpoint is a TM-score to one wild-type reference computed at the
identity residue correspondence. Two fixed points pin that scale, and both
come from the frozen reference set itself rather than from any generated
sequence:

*Upper* -- a reference scored against itself must return exactly 1, which
checks that the identity correspondence and the superposition are what the
endpoint claims.

*Lower* -- two references of equal length are unrelated proteins, so their
mutual score is what "no structural relationship" measures on this
instrument at this chain length. Nothing in the experiment can be read as
evidence of fold recapitulation unless it clears that floor by a margin
worth the name, and the floor is emphatically not zero: TM-scores of
unrelated structures sit well above it, which is why a generated sequence
scoring 0.3 is closer to unrelated than to same-fold.

The comparison is only defined between equal-length references because the
endpoint uses a known correspondence and fits no alignment.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.ladder.fold_geometry import tm_score  # noqa: E402

# TM-score convention: 0.5 is the conventional same-fold threshold.
SAME_FOLD_THRESHOLD = 0.5


def ca_trace(pdb: Path) -> list[list[float]]:
    """Alpha-carbon coordinates in file order."""
    return [
        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        for line in pdb.read_text().splitlines()
        if line.startswith("ATOM") and line[12:16].strip() == "CA"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True, help="frozen wild_type_references.json")
    parser.add_argument("--structures", type=Path, required=True, help="directory holding the reference fold objects")
    parser.add_argument("--out", type=Path, required=True, help="destination json")
    args = parser.parse_args()

    if not args.references.is_file():
        parser.error(f"--references does not exist: {args.references}")
    if not args.structures.is_dir():
        parser.error(f"--structures is not a directory: {args.structures}")

    declared = json.loads(args.references.read_text())["domains"]
    traces: dict[str, list[list[float]]] = {}
    for domain, record in declared.items():
        pdb = args.structures / record["source_object"] / "prediction.pdb"
        if not pdb.is_file():
            raise SystemExit(f"{domain}: reference structure missing at {pdb}")
        trace = ca_trace(pdb)
        if len(trace) != record["length"]:
            raise SystemExit(
                f"{domain}: reference trace has {len(trace)} alpha carbons but the frozen "
                f"record declares length {record['length']}"
            )
        traces[domain] = trace

    self_scores = {}
    for domain, trace in sorted(traces.items()):
        scored = tm_score(trace, trace)
        self_scores[domain] = scored["tm_score"]
        if abs(scored["tm_score"] - 1.0) > 1e-9:
            raise SystemExit(
                f"{domain}: a reference scored against itself returned "
                f"{scored['tm_score']!r}, so the endpoint's correspondence is not the identity"
            )

    by_length: dict[int, list[str]] = {}
    for domain, record in declared.items():
        by_length.setdefault(record["length"], []).append(domain)

    unrelated = []
    for length, domains in sorted(by_length.items()):
        for left, right in itertools.combinations(sorted(domains), 2):
            scored = tm_score(traces[left], traces[right])
            unrelated.append(
                {
                    "length": length,
                    "pair": [left, right],
                    "tm_score": scored["tm_score"],
                    "identity_superposition_tm_score": scored["identity_superposition_tm_score"],
                }
            )

    if not unrelated:
        raise SystemExit(
            "no two references share a length, so the no-relationship floor cannot be "
            "measured on this reference set"
        )

    floor = [entry["tm_score"] for entry in unrelated]
    summary = {
        "schema": "e20_fold_endpoint_calibration_v1",
        "references": str(args.references),
        "endpoint": "tm_score at the identity residue correspondence, against one wild-type reference",
        "same_fold_threshold": SAME_FOLD_THRESHOLD,
        "self_scores": self_scores,
        "unrelated_equal_length_pairs": unrelated,
        "no_relationship_floor": {
            "min": min(floor),
            "max": max(floor),
            "note": (
                "Mutual TM-score of unrelated references of equal length. A generated "
                "sequence must clear this by a meaningful margin before any structural "
                "resemblance can be claimed, and must approach the same-fold threshold "
                "before fold recapitulation can be."
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"self-scores: all {len(self_scores)} references return exactly 1.0")
    print(f"no-relationship floor: {min(floor):.4f} to {max(floor):.4f} over {len(unrelated)} equal-length pairs")
    for entry in unrelated:
        print(
            f"  L={entry['length']:3d}  {entry['pair'][0]} vs {entry['pair'][1]}  "
            f"tm={entry['tm_score']:.4f}  identity_superposition={entry['identity_superposition_tm_score']:.4f}"
        )
    print(f"same-fold threshold for reference: {SAME_FOLD_THRESHOLD}")


if __name__ == "__main__":
    main()
