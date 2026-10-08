#!/usr/bin/env python3
"""Four-state likelihood interactions in a model's own generated proteins.

E16. On natural proteins this project found nothing: over 8,189 measured
double-mutant cycles in 64 family groups, no checkpoint's four-state likelihood
term carried a predictive increment beyond the single-mutation terms once the
measured singles were adjusted for. The sharper question is whether a model is
different on sequences *it produced* -- a model can be internally consistent
about its own output while failing on natural proteins -- and that is what this
stage measures.

What is measured, said plainly. The endpoint is
``epsilon = log p(AB) - log p(Ab) - log p(aB) + log p(ab)`` in nats, formed from
the arm's own retained scalars, where the wild-type term cancels. On a generated
protein there is **no experimental measurement**, so epsilon is the model's
*internal* non-additivity: the gap between its joint likelihood and the sum of
its own single-mutation likelihoods. A nonzero epsilon says the model is not
additive. It says nothing about whether the model predicts real epistasis, and no
number here may be read that way.

Two contrasts and one adjustment:

* the structural contrast -- contacting residue pairs minus non-contacting pairs,
  matched one-to-one on sequence separation inside a declared stratum at
  selection time, nested so that no long product and no densely sampled protein
  carries the panel, with the residual separation imbalance reported beside the
  estimate;
* the adjustment that makes "beyond the contributions of individual mutations"
  concrete -- a leave-one-group-out 20-bin curve of epsilon on the arm's own
  additive prediction, which is the residualizer R3 declared for the measured
  endpoint, applied here to the model's own additive term. The structural
  contrast is reported on both the raw and the adjusted endpoint;
* the magnitude reading -- mean absolute epsilon against the mean absolute
  single-mutation effect of the same cycles, which is the quantity the frozen
  result expressed as the single-mutation terms being several times the
  interaction term.

Every unresolved interval is accompanied by its own half-width and the effect it
could have resolved. An interval that includes zero is unresolved, not zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.interactions import generated_mutation as gm  # noqa: E402
from src.capability.interactions.contact_enrichment import ADDITIVE_RESPONSE_BINS, cross_fit_binned_residuals  # noqa: E402
from src.capability.position.contact_response import stratified_contact_contrast  # noqa: E402

COMPLETION = "generated_interaction_analysis.json"
SCHEMA = "generated_interaction_analysis_v1"

ENDPOINTS = ("epsilon_nats", "epsilon_adjusted_nats")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cycle_table(extraction: gm.Extraction, assays: dict) -> list[dict]:
    """One row per selected residue pair of every generated product."""

    rows: list[dict] = []
    present, absent = extraction.covered(sorted(assays))
    unexplained = [row for row in absent if "without a recorded reason" in row["reason"]]
    if unexplained:
        raise SystemExit(
            f"{extraction.arm}: {len(unexplained)} cohort products are absent from the "
            f"extraction without a recorded reason, first "
            f"{[row['assay'] for row in unexplained][:5]}"
        )
    for assay in present:
        assay_row = assays[assay]
        payload = extraction.payload(assay)
        mutants = [str(value) for value in payload["mutants"]]
        if mutants != list(assay_row["mutants"]):
            raise SystemExit(f"{assay}: archive mutant order differs from the cohort")
        likelihoods = gm.state_likelihoods(payload)
        for cycle in assay_row["cycles"]:
            epsilon = gm.four_state_interaction(likelihoods, cycle)
            additive = gm.additive_prediction(likelihoods, cycle)
            rows.append(
                {
                    "family": assay_row["group"],
                    "assay": assay,
                    "mutation": "pair-design",
                    "direction": "pair",
                    "stratum": cycle["stratum"],
                    "i": int(cycle["i"]),
                    "j": int(cycle["j"]),
                    "separation": int(cycle["separation"]),
                    "contact": bool(cycle["contact"]),
                    "structure_distance_angstrom": float(cycle["structure_distance_angstrom"]),
                    "rsa": None,
                    "stage": assay_row["stage"],
                    "degenerate": bool(assay_row["degenerate"]),
                    "epsilon_nats": epsilon,
                    "additive_nats": additive,
                    "single_low_nats": float(likelihoods[cycle["single_low"]]),
                    "single_high_nats": float(likelihoods[cycle["single_high"]]),
                    "double_nats": float(likelihoods[cycle["double"]]),
                }
            )
    return rows, absent


def adjust(rows: list[dict]) -> dict:
    """Leave-one-group-out binned residuals of epsilon on the model's own additive."""

    groups = np.asarray([row["family"] for row in rows])
    additive = np.asarray([row["additive_nats"] for row in rows], dtype=np.float64)
    epsilon = np.asarray([row["epsilon_nats"] for row in rows], dtype=np.float64)
    distinct = sorted(set(groups.tolist()))
    if len(distinct) < 2:
        for row in rows:
            row["epsilon_adjusted_nats"] = None
        return {"fitted": False, "reason": "a cross-fit curve needs at least two groups"}
    residual = cross_fit_binned_residuals(additive, epsilon, groups, ADDITIVE_RESPONSE_BINS)
    for row, value in zip(rows, residual):
        row["epsilon_adjusted_nats"] = float(value)
    return {
        "fitted": True,
        "bins": ADDITIVE_RESPONSE_BINS,
        "groups": len(distinct),
        "method": (
            "epsilon minus the mean epsilon of its additive-prediction quantile bin, the "
            "curve fit on every family group except the row's own"
        ),
        "removed_variance_share": float(
            1.0 - (residual.var() / epsilon.var()) if epsilon.var() > 0 else 0.0
        ),
    }


def _shaped(rows: list[dict], endpoint: str) -> list[dict]:
    return [
        dict(row, response=float(row[endpoint]), absolute_response=abs(float(row[endpoint])))
        for row in rows
        if row.get(endpoint) is not None and np.isfinite(float(row[endpoint]))
    ]


def magnitude(rows: list[dict], *, draws: int, seed: int) -> dict:
    """Interaction magnitude against the single-mutation magnitude of the same cycles."""

    per_group: dict[str, list[tuple[str, float]]] = {}
    ratio: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        singles = abs(row["single_low_nats"]) + abs(row["single_high_nats"])
        per_group.setdefault(row["family"], []).append((row["assay"], abs(row["epsilon_nats"])))
        if singles > 0:
            ratio.setdefault(row["family"], []).append(
                (row["assay"], abs(row["epsilon_nats"]) / singles)
            )
    absolute = [value for family in sorted(per_group)
                if (value := gm.nested_mean(per_group[family])) is not None]
    shares = [value for family in sorted(ratio)
              if (value := gm.nested_mean(ratio[family])) is not None]
    return {
        "mean_absolute_epsilon_nats": gm.interval(absolute, draws=draws, seed=seed),
        "mean_absolute_epsilon_over_absolute_singles": gm.interval(
            shares, draws=draws, seed=seed
        ),
        "reading": (
            "a ratio well below one says the model's joint likelihood is close to the sum of "
            "its own single-mutation likelihoods; it is an internal-consistency statistic and "
            "not an accuracy statistic"
        ),
    }


def analyse_arm(extraction: gm.Extraction, assays: dict, *, draws: int, seed: int) -> dict:
    rows, absent = cycle_table(extraction, assays)
    adjustment = adjust(rows)

    def block(selected: list[dict]) -> dict:
        record: dict = {
            "cycles": len(selected),
            "contact_cycles": sum(1 for row in selected if row["contact"]),
            "control_cycles": sum(1 for row in selected if not row["contact"]),
            "products": len({row["assay"] for row in selected}),
            "groups": len({row["family"] for row in selected}),
        }
        for endpoint in ENDPOINTS:
            shaped = _shaped(selected, endpoint)
            if not shaped:
                record[endpoint] = {"undefined": f"{endpoint} is not defined on this support"}
                continue
            contrast = stratified_contact_contrast(
                shaped, outcome="absolute_response", draws=draws, seed=seed
            )
            contrast["precision"] = gm.precision_record(contrast["estimate_nats"])
            signed = stratified_contact_contrast(
                shaped, outcome="response", draws=draws, seed=seed
            )
            record[endpoint] = {"absolute": contrast, "signed": signed}
        record["magnitude"] = magnitude(selected, draws=draws, seed=seed)
        return record

    descriptive = {}
    if len(rows) > 2:
        for name, left, right in (
            ("epsilon_vs_additive", "epsilon_nats", "additive_nats"),
            ("epsilon_vs_separation", "epsilon_nats", "separation"),
            ("abs_epsilon_vs_structure_distance", "epsilon_nats", "structure_distance_angstrom"),
        ):
            first = np.asarray([abs(float(row[left])) if name.startswith("abs_")
                                else float(row[left]) for row in rows])
            second = np.asarray([float(row[right]) for row in rows], dtype=np.float64)
            descriptive[name] = _spearman(first, second)

    non_degenerate = [row for row in rows if not row["degenerate"]]
    return {
        "arm": extraction.arm,
        "paradigm": extraction.paradigm,
        "dtype": extraction.completion["identity"]["dtype"],
        "retention_max_abs_nats": float(extraction.completion["totals"]["retention_max_abs_nats"]),
        "products_absent": absent,
        "adjustment": adjustment,
        "primary": block(rows),
        "non_degenerate": block(non_degenerate) if non_degenerate else {
            "undefined": "every product in this cohort falls in the degeneracy stratum"
        },
        "by_stage": dict(
            {
                stage: block([row for row in rows if row["stage"] == stage])
                for stage in sorted({row["stage"] for row in rows})
            },
            status="exploratory",
            reading=(
                "a split of the primary support by generating stage, with no multiplicity "
                "control over the arms, contact definitions and endpoints already reported; "
                "a nominal 95% exclusion here is not a finding"
            ),
        ),
        "descriptive_spearman": descriptive,
    }


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    from scipy.stats import rankdata

    if first.size < 3 or np.allclose(first, first[0]) or np.allclose(second, second[0]):
        return None
    left, right = rankdata(first), rankdata(second)
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=Path, required=True,
                        help="the pairs cohort this extraction was run on")
    parser.add_argument("--archives", type=Path, nargs="+", required=True)
    parser.add_argument("--draws", type=int, default=gm.BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=gm.BOOTSTRAP_SEED)
    parser.add_argument("--device", default="cpu",
                        help="accepted because the campaign queue injects it; unused")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws is positive")

    out = gm.prepare_output_directory(args.out, COMPLETION)
    cohort = json.loads(Path(args.cohort).read_text())
    if cohort.get("schema") != gm.COHORT_SCHEMA or cohort.get("mode") != "pairs":
        raise SystemExit(f"{args.cohort}: not a pairs cohort of {gm.COHORT_SCHEMA}")
    assays = {row["assay"]: row for row in cohort["assays"]}

    arms = [
        analyse_arm(gm.open_extraction(root), assays, draws=args.draws, seed=args.seed)
        for root in args.archives
    ]
    if len({record["arm"] for record in arms}) != len(arms):
        raise SystemExit("two extraction directories report the same arm")

    write_json(out / COMPLETION, {
        "status": "complete",
        "schema": SCHEMA,
        "created_utc": _now(),
        "question": (
            "does a four-state likelihood interaction beyond the single-mutation terms appear "
            "in a model's own generated proteins?"
        ),
        "endpoint": (
            "log p(AB) - log p(Ab) - log p(aB) + log p(ab) in nats, from the arm's own "
            "retained scalars; the model's internal non-additivity, with no experimental "
            "comparator"
        ),
        "design": cohort["design"],
        "cohort": {"path": str(args.cohort), "sha256": sha256_file(args.cohort)},
        "bootstrap": {"draws": args.draws, "seed": args.seed},
        "arms": arms,
        "limitations": [
            "There is no measured double-mutant effect for a generated protein. Internal "
            "non-additivity is not evidence that the model predicts real epistasis.",
            "Contacts come from a predicted structure of a sequence with no experimental "
            "structure, so a contact call inherits the folding instrument's error; the "
            "confidence filter and the admitted-position count are recorded with the cohort.",
            "The support is bounded by how confidently these products fold, not by the "
            "sampling: a product whose positions are mostly below the confidence floor "
            "cannot supply a separation-matched contact pair at all. The cohort's "
            "refused-product count and the admitted-position distribution are the "
            "measurement of that bound and must be read with every estimate here, because "
            "the retained products are the better-folding tail of the generated set.",
            "The separation match is exact by construction inside a stratum, but burial, "
            "composition and local repetitiveness are not matched.",
            "Intervals condition on the extracted likelihoods and on the selected pair design.",
        ],
        "code_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "scripts/capability/interactions/analyse_generated_interaction.py",
                "src/capability/interactions/generated_mutation.py",
                "src/capability/interactions/contact_enrichment.py",
                "src/capability/position/contact_response.py",
            )
        },
    })


if __name__ == "__main__":
    main()
