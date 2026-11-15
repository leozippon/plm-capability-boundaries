"""Length matching for pLDDT contrasts: the retained estimator, the materiality rule and the checks.

ESMFold2 mean CA-pLDDT depends on sequence length, so a gap in confidence between
two sets whose lengths differ is partly a gap in length. The objective-alignment
campaign reads such gaps with an exact sixteen-residue-bin estimator, defined in
``scripts/capability/generation/length_adjusted_contrasts_20261110/analyse.py`` and
rebuilt here so that a producer and a test can import it: lengths are binned by
``floor(length / 16)``, each tuned bin's mean difference is weighted by its tuned
count, and a tuned bin with no base row is dropped.

This module holds only what the length audit shares between its producers: that
estimator, the length summary reported for every compared set, the working rule
that calls a length gap material, the share of a gap that matching removes, and the
comparison of recomputed values against retained ones that licenses all of it.
None of it is a project criterion except the estimator; the thresholds are the
audit's own and are declared here, before any cell is read.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.io import sha256_file

#: Width, in residues, of the bins the retained estimator matches within.
BIN_WIDTH = 16

#: Two sets differ materially in length when their medians are this far apart ...
MATERIAL_MEDIAN_GAP = 20.0
#: ... or when the two-sample Kolmogorov-Smirnov distance between their lengths is this large.
MATERIAL_KS = 0.20
#: A cell with fewer rows than this in either arm is thin and gets no recomputed reading.
MIN_ROWS = 30
#: The length-attributable share is reported only where the raw gap is at least this many points.
MIN_RAW_FOR_SHARE = 1.0


def bin_match(base_lengths, base_values, tuned_lengths, tuned_values,
              width: int = BIN_WIDTH) -> tuple[float, int, int]:
    """The retained sixteen-bin estimator.

    Returns ``(estimate, tuned rows kept, base rows in the kept bins)``. The
    estimate is NaN when no tuned bin has a base row, and is never imputed.
    """

    base_lengths = np.asarray(base_lengths)
    tuned_lengths = np.asarray(tuned_lengths)
    base_values = np.asarray(base_values, dtype=float)
    tuned_values = np.asarray(tuned_values, dtype=float)
    tuned_bins, base_bins = tuned_lengths // width, base_lengths // width
    numerator, denominator, tuned_kept, base_kept = 0.0, 0, 0, 0
    for key in np.unique(tuned_bins):
        in_base = base_bins == key
        if in_base.sum() == 0:
            continue
        in_tuned = tuned_bins == key
        numerator += in_tuned.sum() * (tuned_values[in_tuned].mean() - base_values[in_base].mean())
        denominator += int(in_tuned.sum())
        tuned_kept += int(in_tuned.sum())
        base_kept += int(in_base.sum())
    return (numerator / denominator if denominator else float("nan")), tuned_kept, base_kept


def length_summary(lengths: Iterable[float]) -> dict:
    """Count, median, quartiles, minimum and the number of sequences below 50 residues."""

    lengths = np.asarray(list(lengths), dtype=float)
    if lengths.size == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None, "min": None, "below_50": 0}
    return {"n": int(lengths.size), "median": float(np.median(lengths)),
            "q1": float(np.percentile(lengths, 25)), "q3": float(np.percentile(lengths, 75)),
            "min": int(lengths.min()), "below_50": int((lengths < 50).sum())}


def is_material(median_gap: float, ks: float) -> bool:
    """Whether a length gap is material under the audit's working rule."""

    return abs(median_gap) >= MATERIAL_MEDIAN_GAP or ks >= MATERIAL_KS


def attributable_share(raw: float | None, matched: float | None) -> float | None:
    """(raw - matched) / raw, or None where the raw gap is too small for a share to mean anything."""

    if raw is None or matched is None or math.isnan(matched) or abs(raw) < MIN_RAW_FOR_SHARE:
        return None
    return float((raw - matched) / raw)


def check_values(rows: Sequence[Mapping[str, float]], tolerance: float,
                 expected: int | None = None) -> dict:
    """Compare recomputed against retained values; every one must be within ``tolerance``.

    ``rows`` carry ``abs_difference``. An empty list, a non-finite difference or a
    count other than ``expected`` fails the check, so a check that compared
    nothing cannot pass.
    """

    differences = [float(r["abs_difference"]) for r in rows]
    finite = all(math.isfinite(d) for d in differences)
    worst = max(differences) if differences else float("inf")
    passed = bool(differences) and finite and worst <= tolerance
    if expected is not None:
        passed = passed and len(differences) == expected
    return {"tolerance": tolerance, "n_values": len(differences), "expected_values": expected,
            "max_abs_difference": worst, "passed": passed}


#: The retained evaluation record of the unconditional-generation cohort (E14 and E17), and the
#: two fold-shard indexes that hold one structure per record of its 4,716-record cohort. The
#: indexes live in the audit recovery tree, not where the evaluation record says the original run
#: wrote them; their digests must equal the ones that record carries.
EVALUATION_RECORD = "results/audit_20261011_h200/e17_gap/generation_evaluation.json"
EVALUATION_FOLD_INDEXES = (
    "results/audit_20261012/e17-recovery/geneval_e14_fold_shard0_gpu2/index-000-of-002.jsonl",
    "results/audit_20261012/e17-recovery/geneval_e14_fold_shard1_gpu3/index-001-of-002.jsonl",
)


def read_evaluation_folds(repo: Path) -> pd.DataFrame:
    """One row per record of the E14/E17 cohort with its length and mean CA-pLDDT.

    Refuses an index whose digest is not the one the retained evaluation record carries
    for the shard of the same name, and refuses a row whose structure length is not the
    record's length, so a substituted or truncated index cannot be read as the retained one.
    """

    record = json.loads((repo / EVALUATION_RECORD).read_text(encoding="utf-8"))
    carried = {Path(r["directory"]).name: r["summary"]["index_sha256"]
               for r in record["structure_receipts"]}
    rows = []
    for relative in EVALUATION_FOLD_INDEXES:
        path = repo / relative
        shard = path.parent.name
        digest = sha256_file(path)
        if carried.get(shard) != digest:
            raise SystemExit(f"{relative}: sha256 {digest} is not the {carried.get(shard)} "
                             "the retained evaluation record carries for this shard")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            structure = row["structure"]
            if structure["length"] != row["length"]:
                raise SystemExit(f"{row['id']}: structure length is not the record's length")
            rows.append({"id": row["id"], "arm": row["arm"], "role": row["role"],
                         "roles": tuple(row["roles"]), "stratum": row["stratum"],
                         "condition": row["condition"], "length": int(row["length"]),
                         "sequence_sha256": row["sequence_sha256"],
                         "mean_ca_plddt": float(structure["mean_ca_plddt"])})
    frame = pd.DataFrame(rows)
    if frame["id"].nunique() != len(frame) or len(frame) != 4716:
        raise SystemExit("the fold indexes do not hold the 4,716 distinct records of the cohort")
    return frame
