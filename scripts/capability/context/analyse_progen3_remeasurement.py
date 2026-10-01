#!/usr/bin/env python3
"""The fifteen-checkpoint homologue-concordance/recognition correlation, remeasured.

The withdrawn coefficients (pooled -0.6107, low-overlap -0.5512) were computed
by ``manuscript_cross_result_analysis.py`` from the stage-46 expansion aggregate
and the unconditional-generation diversity census. Only the two ProGen3 rows of
that aggregate were affected by the sequence-ID defect, so this script replaces
exactly those two rows with the corrected stage-46 analyse output and recomputes
the same statistic over the same fifteen checkpoints.

It is a replacement, not a new analysis, so it reproduces the withdrawn
coefficients from the historical inputs first and refuses to report a corrected
value if that reproduction misses the recorded numbers. Everything else -- the
lineage map, the leave-one-lineage-out set, the recognition-rate definition and
the absence of a model-population interval -- is carried over unchanged.

CPU only. No model is loaded and nothing is resampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr

#: What ``manuscript_cross_result_analysis.py`` recorded on 2026-09-21 and the
#: audit withdrew on 2026-09-23. Reproducing these from the historical inputs is
#: what makes the corrected coefficients a like-for-like replacement.
WITHDRAWN = {"pooled": -0.6107, "decisive_stratum": -0.5512}
REPRODUCTION_TOLERANCE = 5e-5

#: The two axes of the correlation, named as the manuscript analysis names them.
STRATA = ("pooled", "decisive_stratum")


def lineage(arm: str) -> str:
    """The manuscript analysis's lineage map, verbatim."""

    if arm.startswith("progen2"):
        return "progen2"
    if arm.startswith("progen3"):
        return "progen3"
    if arm in ["llama-2-7b", "prollama-stage-1", "prollama"]:
        return "llama2-prollama"
    if arm.startswith("galactica"):
        return "galactica"
    return arm


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_census(paths: list[Path]) -> dict[str, Any]:
    """The generation census, merged in the manuscript analysis's order."""

    census: dict[str, Any] = {}
    for path in paths:
        census.update(json.loads(path.read_text(encoding="utf-8"))["checkpoints"])
    return census


def panel(
    homologue_arms: dict[str, Any], census: dict[str, Any]
) -> list[dict[str, Any]]:
    """One row per checkpoint that has both a homologue verdict and a census."""

    rows = []
    for arm, block in homologue_arms.items():
        if arm not in census:
            continue
        if block.get("status") != "scored":
            raise ValueError(f"{arm} carries status {block.get('status')!r}, not a verdict")
        row = {
            "arm": arm,
            "lineage": lineage(arm),
            "recognition_rate": census[arm]["n_any_profile_hits"] / census[arm]["n_attempts"],
        }
        for stratum in STRATA:
            row[stratum] = block[stratum]["auroc"]["mean"]
        rows.append(row)
    return rows


def coefficients(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Spearman against recognition rate, pooled and with each lineage left out."""

    out: dict[str, Any] = {}
    for stratum in STRATA:
        block: dict[str, Any] = {
            "n_checkpoints": len(rows),
            "spearman": float(
                spearmanr(
                    [row[stratum] for row in rows],
                    [row["recognition_rate"] for row in rows],
                ).statistic
            ),
            "leave_one_lineage_out": {},
        }
        for name in sorted({row["lineage"] for row in rows}):
            kept = [row for row in rows if row["lineage"] != name]
            if len(kept) < 3:
                continue
            block["leave_one_lineage_out"][name] = float(
                spearmanr(
                    [row[stratum] for row in kept],
                    [row["recognition_rate"] for row in kept],
                ).statistic
            )
        out[stratum] = block
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--historical-aggregate",
        type=Path,
        required=True,
        help="the withdrawn sixteen-arm stage-46 expansion aggregate",
    )
    parser.add_argument(
        "--corrected-aggregate",
        type=Path,
        required=True,
        help="the stage-46 analyse output over the corrected ProGen3 scores",
    )
    parser.add_argument("--census", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None, help="accepted and ignored; CPU only")
    args = parser.parse_args()

    historical = json.loads(args.historical_aggregate.read_text(encoding="utf-8"))
    corrected = json.loads(args.corrected_aggregate.read_text(encoding="utf-8"))
    census = load_census(args.census)

    if corrected["cohort_digest"] != historical["cohort_digest"]:
        raise SystemExit(
            "the corrected analyse ran against a different cohort digest "
            f"({corrected['cohort_digest']}) than the historical aggregate "
            f"({historical['cohort_digest']}); this is not a replacement"
        )
    replaced = sorted(corrected["arms"])
    if not replaced:
        raise SystemExit("the corrected analyse carries no arms")
    for arm in replaced:
        if arm not in historical["arms"]:
            raise SystemExit(f"{arm} is not in the historical panel")
        if corrected["arms"][arm]["plan_digest"] != historical["arms"][arm]["plan_digest"]:
            raise SystemExit(
                f"{arm}: the corrected analyse used plan digest "
                f"{corrected['arms'][arm]['plan_digest']}, not the historical "
                f"{historical['arms'][arm]['plan_digest']}"
            )

    historical_rows = panel(historical["arms"], census)
    reproduced = coefficients(historical_rows)
    drift = {
        stratum: reproduced[stratum]["spearman"] - WITHDRAWN[stratum]
        for stratum in STRATA
    }
    if max(abs(value) for value in drift.values()) > REPRODUCTION_TOLERANCE:
        raise SystemExit(
            "this script does not reproduce the withdrawn coefficients from the "
            f"historical inputs (differences {drift}); it cannot be read as a "
            "replacement measurement of them"
        )

    # Copied, so the retained historical panel stays readable beside the new one.
    corrected_rows = [dict(row) for row in historical_rows]
    for row in corrected_rows:
        if row["arm"] in replaced:
            block = corrected["arms"][row["arm"]]
            for stratum in STRATA:
                row[stratum] = block[stratum]["auroc"]["mean"]
    measured = coefficients(corrected_rows)

    payload = {
        "written_utc": None,
        "scope": (
            "replacement of the withdrawn fifteen-checkpoint "
            "homologue-concordance/recognition coefficients"
        ),
        "replaced_arms": replaced,
        "n_checkpoints": len(corrected_rows),
        "cohort_digest": historical["cohort_digest"],
        "withdrawn": WITHDRAWN,
        "reproduced_from_historical_inputs": {
            stratum: reproduced[stratum]["spearman"] for stratum in STRATA
        },
        "reproduction_difference": drift,
        "historical": reproduced,
        "remeasured": measured,
        "panel": {
            "historical": historical_rows,
            "remeasured": corrected_rows,
        },
        "source_sha256": {
            str(args.historical_aggregate): digest(args.historical_aggregate),
            str(args.corrected_aggregate): digest(args.corrected_aggregate),
            **{str(path): digest(path) for path in args.census},
        },
        "limitations": [
            "descriptive rank correlation across a fixed checkpoint panel; no "
            "model-population interval is estimated, as in the withdrawn analysis",
            "the thirteen unaffected checkpoints carry their retained historical "
            "concordance means; only the two ProGen3 rows are remeasured",
            "recognition rate is a profile-hit rate over 800 retained attempts, "
            "not a measure of function",
        ],
    }
    from datetime import datetime, timezone

    payload["written_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "progen3_association.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {destination}")
    for stratum in STRATA:
        print(
            f"{stratum}: withdrawn {WITHDRAWN[stratum]:+.4f} "
            f"(reproduced {reproduced[stratum]['spearman']:+.4f}) -> remeasured "
            f"{measured[stratum]['spearman']:+.4f} over {len(corrected_rows)} checkpoints"
        )


if __name__ == "__main__":
    main()
