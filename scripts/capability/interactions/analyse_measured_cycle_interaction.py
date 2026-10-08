#!/usr/bin/env python3
"""Structural contacts and double-mutant interactions in natural proteins: the
model side.

E07. The measured side of this question is already on record. The frozen
MegaScale double-mutant cohort's 217 annotated site pairs were contrasted,
contact against separation-, burial- and hydrophobicity-matched non-contact, and
the enrichment did not resolve: the CB-contact contrast of mean absolute cycle
epsilon is about +0.054 kcal/mol with a group-equal interval of roughly
[-0.167, +0.245] on 120 contact and 57 control site pairs. That is an
*unresolved* contrast, not a zero one, and the half-width is around a third of
the endpoint's own scale.

What remains, and what this stage does, is the model side: with the same
annotation, the same coarsened-exact matching, the same weighting and the same
two-stage group-then-site-pair bootstrap, does an arm's own four-state likelihood
interaction concentrate at structural contacts? The endpoint is
``epsilon = log p(AB) - log p(Ab) - log p(aB) + log p(ab)`` in nats, summarised
per site pair with cycles weighted equally, exactly as the measured endpoint is.

Two readings are produced and must not be confused. The *contrast* asks whether
the model's interaction term is larger at contacts. The *agreement* asks whether
it tracks the measured interaction cycle by cycle, which is the only accuracy
question this experiment can pose -- and whose interpretation is bounded above by
the measured side's own precision. Where the measured contrast is unresolved, a
model contrast that matches it is not a success and a model contrast that differs
from it is not a failure; both are reported with the precision each design
carries.

The measured numbers are quoted from the published artefact rather than
recomputed, because the published estimate is the project's single source for
them and its adjusted endpoint needs the pinned per-channel parquet rows that
this stage does not open.
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
from src.capability.interactions.contact_enrichment import (  # noqa: E402
    ADDITIVE_RESPONSE_BINS,
    cross_fit_binned_residuals,
)
from scripts.capability.interactions import measure_contact_epsilon_enrichment as enrichment  # noqa: E402
from scripts.capability.interactions import reaudit_contact_endpoint as reaudit  # noqa: E402

COMPLETION = "measured_cycle_interaction_analysis.json"
SCHEMA = "measured_cycle_interaction_v1"

#: Model-side endpoints, in nats per site pair, cycles weighted equally.
ENDPOINTS = (
    "mean_abs_model_epsilon",
    "mean_model_epsilon",
    "mean_abs_model_epsilon_adjusted",
)

#: Weightings reported, named as the published measured artefact names them.
WEIGHTINGS = ("site_pair_equal", "group_equal")

#: The measured endpoints quoted for comparison, with the support they are read on.
QUOTED = ("mean_abs_epsilon", "mean_abs_epsilon_adjusted")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def published_measured(published: dict, *, support: str, definition: str) -> dict:
    """Quote the published measured contrast, failing loudly on a shape change."""

    try:
        endpoints = published["supports"][support]["definitions"][definition]["endpoints"]
    except KeyError as error:
        raise SystemExit(
            f"the published artefact has no supports/{support}/definitions/{definition}: {error}"
        ) from error
    out = {}
    for endpoint in QUOTED:
        if endpoint not in endpoints:
            raise SystemExit(f"the published artefact has no endpoint {endpoint!r}")
        out[endpoint] = {
            weighting: {
                key: endpoints[endpoint][weighting][key]
                for key in (
                    "difference", "interval", "excludes_zero", "contact_mean", "control_mean",
                    "contact_site_pairs", "control_site_pairs", "contact_groups",
                    "control_groups", "effective_contact_site_pairs",
                    "effective_control_site_pairs", "unit",
                )
            }
            for weighting in WEIGHTINGS
            if weighting in endpoints[endpoint]
        }
        for weighting, record in out[endpoint].items():
            record["precision"] = gm.precision_record(record)
    return out


def excluded_states(declaration: dict) -> set[str]:
    """Sequences the declared indel exclusion removes entirely."""

    return {
        str(row["sequence"])
        for row in declaration["states"]
        if row.get("value_without_indel_rows_kcal_mol") is None
    }


def cycle_table(extraction: gm.Extraction, assays: dict, measured: dict,
                drop: set[str]) -> list[dict]:
    """One row per cohort cycle with the model term, the measured term and the key."""

    rows: list[dict] = []
    present, absent = extraction.covered(sorted(assays))
    unexplained = [row for row in absent if "without a recorded reason" in row["reason"]]
    if unexplained:
        raise SystemExit(
            f"{extraction.arm}: {len(unexplained)} backgrounds are absent from the extraction "
            f"without a recorded reason, first {[row['assay'] for row in unexplained][:5]}"
        )
    for assay in present:
        assay_row = assays[assay]
        payload = extraction.payload(assay)
        mutants = [str(value) for value in payload["mutants"]]
        if mutants != list(assay_row["mutants"]):
            raise SystemExit(f"{assay}: archive mutant order differs from the cohort")
        states = dict(zip(mutants, (str(value) for value in assay_row["sequences"])))
        likelihoods = gm.state_likelihoods(payload)
        for cycle in assay_row["cycles"]:
            labels = (cycle["single_low"], cycle["single_high"], cycle["double"])
            if any(states[label] in drop for label in labels):
                continue
            key = cycle["site_pair"]
            rows.append(
                {
                    "site_pair": key,
                    "group": assay_row["group"],
                    "background": assay,
                    "separation": int(cycle["separation"]),
                    "model_epsilon": gm.four_state_interaction(likelihoods, cycle),
                    "model_additive": gm.additive_prediction(likelihoods, cycle),
                    "measured_epsilon": measured.get((assay, cycle["double"])),
                }
            )
    return rows, absent


def site_pair_endpoints(rows: list[dict]) -> dict[str, dict]:
    """Per-site-pair model endpoints, cycles weighted equally inside a site pair."""

    groups = np.asarray([row["group"] for row in rows])
    additive = np.asarray([row["model_additive"] for row in rows], dtype=np.float64)
    epsilon = np.asarray([row["model_epsilon"] for row in rows], dtype=np.float64)
    adjusted = cross_fit_binned_residuals(additive, epsilon, groups, ADDITIVE_RESPONSE_BINS)
    collected: dict[str, dict] = {}
    for row, residual in zip(rows, adjusted):
        entry = collected.setdefault(
            row["site_pair"],
            {"group": row["group"], "background": row["background"], "cycles": 0,
             "epsilon": [], "adjusted": [], "measured": []},
        )
        entry["cycles"] += 1
        entry["epsilon"].append(float(row["model_epsilon"]))
        entry["adjusted"].append(float(residual))
        if row["measured_epsilon"] is not None:
            entry["measured"].append(float(row["measured_epsilon"]))
    out = {}
    for key, entry in collected.items():
        values = np.asarray(entry["epsilon"])
        residuals = np.asarray(entry["adjusted"])
        out[key] = {
            "group": entry["group"],
            "background": entry["background"],
            "cycles": entry["cycles"],
            "mean_abs_model_epsilon": float(np.abs(values).mean()),
            "mean_model_epsilon": float(values.mean()),
            "mean_abs_model_epsilon_adjusted": float(np.abs(residuals).mean()),
            "mean_abs_measured_epsilon": (
                float(np.abs(entry["measured"]).mean()) if entry["measured"] else None
            ),
        }
    return out


def analyse_arm(extraction: gm.Extraction, assays: dict, measured: dict, annotation: dict,
                drop: set[str], *, declared_boundary: float) -> dict:
    rows, absent = cycle_table(extraction, assays, measured, drop)
    if not rows:
        raise SystemExit(f"{extraction.arm}: no cycle survived the declared exclusions")
    pairs = site_pair_endpoints(rows)
    support, boundary = reaudit.support_rows(annotation, pairs)
    if not support:
        raise SystemExit("no annotated site pair is present in this extraction")
    if boundary != declared_boundary:
        raise SystemExit(
            f"the burial matching boundary derived from this annotation ({boundary!r}) is not "
            f"the one the published measured contrast used ({declared_boundary!r}); the model "
            "side would not be matched the same way and the two are not comparable"
        )

    record: dict = {
        "arm": extraction.arm,
        "paradigm": extraction.paradigm,
        "dtype": extraction.completion["identity"]["dtype"],
        "retention_max_abs_nats": float(extraction.completion["totals"]["retention_max_abs_nats"]),
        "backgrounds_absent": absent,
        "cycles": len(rows),
        "site_pairs": len(pairs),
        "annotated_site_pairs": len(support),
        "groups": len({row["group"] for row in pairs.values()}),
        "rsa_boundary": boundary,
        "definitions": {},
        "agreement": {},
    }
    for definition in ("heavy_atom", "cb"):
        treated = enrichment.treated_mask(support, definition)
        block: dict = {
            "contact_site_pairs": int(np.asarray(treated, dtype=bool).sum()),
            "control_site_pairs": int((~np.asarray(treated, dtype=bool)).sum()),
            "endpoints": {},
        }
        for endpoint in ENDPOINTS:
            values = np.asarray(
                [pairs[row["site_pair"]][endpoint] for row in support], dtype=np.float64
            )
            block["endpoints"][endpoint] = {}
            for weighting in WEIGHTINGS:
                point, bounds, weight = reaudit.matched_contrast(
                    support, values, treated, boundary,
                    group_equal=(weighting == "group_equal"),
                )
                shaped = reaudit.published_shaped(point, bounds, support, pairs, treated, weight)
                shaped["unit"] = "nats"
                shaped["precision"] = gm.precision_record(shaped)
                block["endpoints"][endpoint][weighting] = shaped
        record["definitions"][definition] = block

    cycle_measured = [row for row in rows if row["measured_epsilon"] is not None]
    pair_measured = [
        key for key, entry in pairs.items() if entry["mean_abs_measured_epsilon"] is not None
    ]
    record["agreement"] = {
        "cycles_with_measurement": len(cycle_measured),
        "site_pairs_with_measurement": len(pair_measured),
        "cycle_spearman_signed": _spearman(
            [row["model_epsilon"] for row in cycle_measured],
            [row["measured_epsilon"] for row in cycle_measured],
        ),
        "cycle_spearman_absolute": _spearman(
            [abs(row["model_epsilon"]) for row in cycle_measured],
            [abs(row["measured_epsilon"]) for row in cycle_measured],
        ),
        "site_pair_spearman_absolute": _spearman(
            [pairs[key]["mean_abs_model_epsilon"] for key in pair_measured],
            [pairs[key]["mean_abs_measured_epsilon"] for key in pair_measured],
        ),
        "reading": (
            "a rank agreement between the model's own interaction term and the measured one "
            "is the only accuracy statement this experiment supports, and its interpretation "
            "is bounded by the measured side's own precision"
        ),
    }
    return record


def _spearman(first, second) -> float | None:
    from scipy.stats import rankdata

    left = np.asarray(list(first), dtype=np.float64)
    right = np.asarray(list(second), dtype=np.float64)
    if left.size < 3 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return None
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=Path, required=True, help="the cycles cohort")
    parser.add_argument("--archives", type=Path, nargs="+", required=True)
    parser.add_argument("--annotation", type=Path, default=reaudit.ANNOTATION)
    parser.add_argument("--measured", type=Path, default=reaudit.COHORT,
                        help="the frozen pairwise cohort, read for its cycle epsilon only")
    parser.add_argument("--published", type=Path, default=reaudit.PUBLISHED,
                        help="the published measured contact endpoint, quoted not recomputed")
    parser.add_argument("--indel-declaration", type=Path,
                        help="declared indel exclusion; its fully absent states are dropped")
    parser.add_argument("--support", default="all_cycles",
                        choices=("all_cycles", "indel_excluded"),
                        help="which published measured support to quote beside the model side")
    parser.add_argument("--device", default="cpu",
                        help="accepted because the campaign queue injects it; unused")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = gm.prepare_output_directory(args.out, COMPLETION)
    cohort = json.loads(Path(args.cohort).read_text())
    if cohort.get("schema") != gm.COHORT_SCHEMA or cohort.get("mode") != "cycles":
        raise SystemExit(f"{args.cohort}: not a cycles cohort of {gm.COHORT_SCHEMA}")
    assays = {row["assay"]: row for row in cohort["assays"]}
    annotation = json.loads(Path(args.annotation).read_text())
    published = json.loads(Path(args.published).read_text())
    if published.get("schema") != enrichment.SCHEMA:
        raise SystemExit(f"{args.published}: unexpected schema {published.get('schema')!r}")

    # The per-cycle join is on the double state's own substitution label, which names
    # both positions and both replacement residues and is therefore unique inside a
    # background. A site pair carries many cycles and a join on the position pair
    # alone would collapse them.
    frozen = json.loads(Path(args.measured).read_text())
    measured: dict[tuple[str, str], float] = {}
    for background in frozen["backgrounds"]:
        wildtype = background["cycles"][0]["sequences"][0]
        for cycle in background["cycles"]:
            label = gm.state_label(wildtype, cycle["sequences"][3])
            if ":" not in label:
                raise SystemExit(
                    f"{background['name']}: the fourth state of a cycle is not a double "
                    f"substitution ({label})"
                )
            key = (str(background["name"]), label)
            if key in measured:
                raise SystemExit(f"{key}: the frozen cohort repeats a double state")
            measured[key] = float(cycle["epsilon"])

    drop: set[str] = set()
    declaration = None
    if args.indel_declaration is not None:
        declaration = json.loads(Path(args.indel_declaration).read_text())
        drop = excluded_states(declaration)

    declared_boundary = float(published["declaration"]["rsa_boundary"])
    arms = [
        analyse_arm(gm.open_extraction(root), assays, measured, annotation, drop,
                    declared_boundary=declared_boundary)
        for root in args.archives
    ]
    if len({record["arm"] for record in arms}) != len(arms):
        raise SystemExit("two extraction directories report the same arm")

    write_json(out / COMPLETION, {
        "status": "complete",
        "schema": SCHEMA,
        "created_utc": _now(),
        "question": (
            "does a model's own four-state likelihood interaction concentrate at structural "
            "contacts of natural proteins, where a measured non-additivity exists?"
        ),
        "design": cohort["design"],
        "cohort": {"path": str(args.cohort), "sha256": sha256_file(args.cohort)},
        "inputs": {
            "annotation": {"path": str(args.annotation), "sha256": sha256_file(args.annotation)},
            "measured_cohort": {"path": str(args.measured), "sha256": sha256_file(args.measured)},
            "published": {"path": str(args.published), "sha256": sha256_file(args.published)},
            "indel_declaration": (
                None if declaration is None
                else {"path": str(args.indel_declaration),
                      "sha256": sha256_file(args.indel_declaration),
                      "states_dropped": sorted(drop)}
            ),
        },
        "estimator": {
            "matching": "coarsened exact matching on the published separation, burial and "
                        "hydrophobicity cells, with the published RSA boundary rule",
            "bootstrap": "group then site pair inside the drawn group",
            "declaration": published["declaration"],
        },
        "measured_quoted": {
            "support": args.support,
            "heavy_atom": published_measured(published, support=args.support,
                                             definition="heavy_atom"),
            "cb": published_measured(published, support=args.support, definition="cb"),
        },
        "arms": arms,
        "limitations": [
            "The measured contact enrichment on this support is unresolved, so the model side "
            "cannot be scored against it as accuracy; both sides are reported with their own "
            "precision instead.",
            "The source study selected these site pairs as coupling tables, so contacts are the "
            "majority arm and the control arm is small; the effective control site-pair count "
            "is reported with every estimate.",
            "The model endpoint is in nats and the measured one in kcal/mol; only ranks and "
            "the sign of a contrast are comparable between them.",
            "The cycle-level agreement is formed on position pairs carrying exactly one cycle, "
            "because the frozen cohort indexes a cycle by its position pair alone.",
        ],
        "code_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "scripts/capability/interactions/analyse_measured_cycle_interaction.py",
                "src/capability/interactions/generated_mutation.py",
                "src/capability/interactions/contact_enrichment.py",
                "scripts/capability/interactions/reaudit_contact_endpoint.py",
            )
        },
    })


if __name__ == "__main__":
    main()
