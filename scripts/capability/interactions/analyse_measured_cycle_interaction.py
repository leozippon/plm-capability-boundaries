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
it tracks the measured interaction cycle by cycle, and it is a descriptive point
estimate: no group-resampled correlation uncertainty is computed for it, so it
licenses no prediction-accuracy statement and no absence claim. Where the
measured contrast is unresolved, a model contrast that matches it is not a
success and a model contrast that differs from it is not a failure; both are
reported with the precision each design carries.

The published measured numbers are still quoted from the published artefact
rather than recomputed, because the published estimate is the project's single
source for them and its adjusted endpoint needs the pinned per-channel parquet
rows that this stage does not open. Beside that historical comparator this stage
does recompute one measured contact contrast of its own: on the declared
state-level repair of the frozen cohort, on the model's own matched site pairs
and with the model's own weighting and bootstrap, in kcal/mol. The two are never
added together or rescaled into one number.
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
from src.capability.interactions.pairwise_epistasis import CYCLE_IDENTITY_TOLERANCE  # noqa: E402
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

#: The policy when no declaration is given: the frozen cohort's own cycle epsilon is
#: carried exactly, which is the historical raw behaviour, while the raw four-state
#: identity is still validated and refused beyond the declared tolerance.
RAW_MEASURED_POLICY = {
    "declaration_schema": None,
    "epsilon": "the frozen cohort's own cycle epsilon is carried exactly, which is the "
               "historical raw behaviour",
    "raw_identity_checked": "the frozen epsilon is still checked against its own four state "
                            "medians and refused if it disagrees beyond the declared "
                            "tolerance",
}

#: The declared state-level repair of the frozen measured cohort. A cycle's frozen
#: epsilon is the raw four-state combination of the cohort's state medians, and
#: those medians still contain insertion- and deletion-construct rows, so the
#: repair has to happen at the state level before any measured epsilon is read.
DECLARED_REPAIR_POLICY = {
    "declaration_schema": "pairwise_indel_exclusion_v1",
    "scope": "one measured state, keyed by its background and its sequence; a position pair "
             "is never the unit",
    "absent_state": "a state whose rows are all insertion/deletion constructs has no "
                    "remaining value, so every cycle whose four states include it is removed",
    "corrected_state": "a state whose remaining rows give a different median carries that "
                       "declared value in place of its frozen median",
    "epsilon": "each retained cycle's epsilon is recomputed from its four corrected state "
               "values as AB - A - B + WT, then joined on the background and the double "
               "state's own substitution label",
}


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


def declared_repairs(declaration: dict
                     ) -> tuple[set[tuple[str, str]], dict[tuple[str, str], float]]:
    """The declared absent and corrected measured states, keyed by background and sequence.

    The declaration repairs a state, never a position pair, so every key carries
    its own background. A state with no replacement value is absent and takes
    every cycle that indexes it out of the support; a state with one carries that
    value in place of its frozen median.
    """

    if declaration.get("schema") != DECLARED_REPAIR_POLICY["declaration_schema"]:
        raise SystemExit(
            f"unexpected indel-exclusion schema {declaration.get('schema')!r}, expected "
            f"{DECLARED_REPAIR_POLICY['declaration_schema']!r}"
        )
    if "states" not in declaration:
        raise SystemExit(
            "the indel-exclusion declaration must carry the state-level 'states' shape, not "
            f"the cycle-level one: found keys {sorted(declaration)}"
        )
    absent: set[tuple[str, str]] = set()
    corrected: dict[tuple[str, str], float] = {}
    for row in declaration["states"]:
        key = (str(row["background"]), str(row["sequence"]))
        value = row.get("value_without_indel_rows_kcal_mol")
        if value is None:
            absent.add(key)
            continue
        number = float(value)
        if not np.isfinite(number):
            raise SystemExit(f"{key}: the declared replacement value is not finite ({value!r})")
        corrected[key] = number
    return absent, corrected


def _site_pair_name(background: str, positions) -> str:
    ordered = sorted(int(value) for value in positions)
    return f"{background}:" + "-".join(str(value) for value in ordered)


def corrected_measured(frozen: dict, declaration: dict | None, *, cohort_sha256: str
                       ) -> tuple[dict, dict, set[tuple[str, str]]]:
    """Per-cycle measured epsilon after the declared state-level repair.

    The frozen cohort carries one median per state and one epsilon per four-state
    cycle, and its epsilon is the raw combination of the uncorrected medians. A
    declaration is applied at the state level: an absent state drops every cycle
    that indexes it, a corrected state carries its declared median, and each
    retained cycle's epsilon is recomputed as ``AB - A - B + WT`` before the join
    on the background and the double state's own substitution label. The raw
    four-state identity is checked first, so a cohort that disagrees with its own
    states is refused rather than repaired. With no declaration the measured map
    is the frozen one exactly, which is the historical raw behaviour.

    Returns the measured map, its accounting, and the absent state keys the model
    side must drop cycles on.
    """

    absent: set[tuple[str, str]] = set()
    corrected: dict[tuple[str, str], float] = {}
    if declaration is not None:
        if declaration.get("cohort_sha256") != cohort_sha256:
            raise SystemExit(
                "the indel-exclusion declaration was declared over a different cohort: "
                f"declaration {declaration.get('cohort_sha256')!r} against measured "
                f"{cohort_sha256!r}"
            )
        absent, corrected = declared_repairs(declaration)

    measured: dict[tuple[str, str], float] = {}
    removed: list[dict] = []
    changed: list[float] = []
    changed_pairs: dict[str, int] = {}
    retained_pairs: set[str] = set()
    retained_groups: set[str] = set()
    for background in frozen["backgrounds"]:
        name = str(background["name"])
        group = str(background["group"])
        values: dict[str, float] = {}
        for sequence, measurement in background["measurements"].items():
            number = float(measurement["value"])
            if not np.isfinite(number):
                raise SystemExit(f"{name}: state {sequence!r} carries a non-finite value")
            values[str(sequence)] = number
        repaired_values = dict(values)
        for (state_background, sequence), number in corrected.items():
            if state_background == name:
                repaired_values[sequence] = number
        for cycle in background["cycles"]:
            states = [str(value) for value in cycle["sequences"]]
            if any(sequence not in values for sequence in states):
                raise SystemExit(f"{name}: a cycle indexes a state with no measured value")
            raw = (values[states[3]] - values[states[1]]
                   - values[states[2]] + values[states[0]])
            frozen_epsilon = float(cycle["epsilon"])
            if not np.isfinite(frozen_epsilon):
                raise SystemExit(f"{name}: a cycle carries a non-finite epsilon")
            if abs(raw - frozen_epsilon) > CYCLE_IDENTITY_TOLERANCE:
                raise SystemExit(
                    f"{name}: the cycle at {cycle['positions']} disagrees with its own four "
                    f"state values by {abs(raw - frozen_epsilon):.3e} kcal/mol"
                )
            pair = _site_pair_name(name, cycle["positions"])
            if any((name, sequence) in absent for sequence in states):
                removed.append({"site_pair": pair, "background": name,
                                "positions": sorted(int(value)
                                                    for value in cycle["positions"])})
                continue
            repaired = (repaired_values[states[3]] - repaired_values[states[1]]
                        - repaired_values[states[2]] + repaired_values[states[0]])
            label = gm.state_label(states[0], states[3])
            if ":" not in label:
                raise SystemExit(
                    f"{name}: the fourth state of a cycle is not a double substitution "
                    f"({label})"
                )
            key = (name, label)
            if key in measured:
                raise SystemExit(f"{key}: the frozen cohort repeats a double state")
            # Exact historical identity: with no declaration the frozen epsilon is carried
            # unchanged, even where it differs from the recomputed four-state combination by
            # roundoff inside the checked tolerance. A declaration carries the value
            # recomputed from its corrected states instead.
            measured[key] = frozen_epsilon if declaration is None else repaired
            retained_pairs.add(pair)
            retained_groups.add(group)
            if repaired != raw:
                changed.append(float(repaired - raw))
                changed_pairs[pair] = changed_pairs.get(pair, 0) + 1

    corrected_states = []
    if declaration is not None:
        for row in declaration["states"]:
            value = row.get("value_without_indel_rows_kcal_mol")
            if value is None:
                continue
            frozen_value = float(row["value_kcal_mol"])
            corrected_states.append({
                "background": str(row["background"]),
                "sequence": str(row["sequence"]),
                "frozen_value_kcal_mol": frozen_value,
                "value_without_indel_rows_kcal_mol": float(value),
                "change_kcal_mol": float(value) - frozen_value,
            })
        corrected_states.sort(key=lambda row: (row["background"], row["sequence"]))

    accounting = {
        "repair_applied": declaration is not None,
        "policy": (DECLARED_REPAIR_POLICY if declaration is not None else RAW_MEASURED_POLICY),
        "measured_cohort_sha256": cohort_sha256,
        "declaration_cohort_sha256": (
            None if declaration is None else str(declaration.get("cohort_sha256"))
        ),
        "absent_states": [{"background": background, "sequence": sequence}
                          for background, sequence in sorted(absent)],
        "corrected_states": corrected_states,
        "removed_cycles": sorted(removed, key=lambda row: (row["background"],
                                                           row["positions"])),
        "removed_cycles_count": len(removed),
        "changed_cycles_count": len(changed),
        "changed_site_pairs": {pair: changed_pairs[pair] for pair in sorted(changed_pairs)},
        "changed_epsilon_kcal_mol": sorted(set(changed)),
        "largest_change_kcal_mol": (max(abs(value) for value in changed) if changed else 0.0),
        "retained": {"cycles": len(measured), "site_pairs": len(retained_pairs),
                     "groups": len(retained_groups)},
    }
    return measured, accounting, absent


def measured_contrast(support: list[dict], pairs: dict, boundary: float, *,
                      correction: dict) -> dict:
    """The measured contact contrast on the model's own matched support.

    The measured term is read on exactly the annotated site pairs, the coarsened
    exact match on separation, burial and hydrophobicity, the group and site-pair
    weighting and the two-stage bootstrap the model contrast uses, so the two are
    estimated on one support rather than on two. The unit is kcal/mol and it is
    never added to the model's nats. A support pair without a measured value is
    refused rather than silently dropped.
    """

    values = reaudit.values_for(support, pairs, "mean_abs_measured_epsilon")
    block: dict = {
        "endpoint": "mean_abs_measured_epsilon",
        "unit": "kcal_mol",
        "policy": (DECLARED_REPAIR_POLICY if correction["repair_applied"]
                   else RAW_MEASURED_POLICY),
        "repair_applied": bool(correction["repair_applied"]),
        "definitions": {},
    }
    for definition in ("heavy_atom", "cb"):
        treated = enrichment.treated_mask(support, definition)
        definition_block: dict = {
            "contact_site_pairs": int(np.asarray(treated, dtype=bool).sum()),
            "control_site_pairs": int((~np.asarray(treated, dtype=bool)).sum()),
        }
        for weighting in WEIGHTINGS:
            point, bounds, weight = reaudit.matched_contrast(
                support, values, treated, boundary,
                group_equal=(weighting == "group_equal"),
            )
            shaped = reaudit.published_shaped(point, bounds, support, pairs, treated, weight)
            shaped["unit"] = "kcal_mol"
            shaped["precision"] = gm.precision_record(shaped)
            definition_block[weighting] = shaped
        block["definitions"][definition] = definition_block
    return block


def cycle_table(extraction: gm.Extraction, assays: dict, measured: dict,
                absent: set[tuple[str, str]]) -> tuple[list[dict], list[dict]]:
    """One row per cohort cycle with the model term, the measured term and the key."""

    rows: list[dict] = []
    present, missing = extraction.covered(sorted(assays))
    unexplained = [row for row in missing if "without a recorded reason" in row["reason"]]
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
            if any((assay, states[label]) in absent for label in labels):
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
    return rows, missing


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
                absent: set[tuple[str, str]], *, declared_boundary: float,
                correction: dict) -> dict:
    rows, backgrounds_absent = cycle_table(extraction, assays, measured, absent)
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
        "backgrounds_absent": backgrounds_absent,
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

    # The measured contrast recomputed on the model's own matched support, beside the
    # historical published quote and never merged with it.
    record["measured_recomputed"] = measured_contrast(
        support, pairs, boundary, correction=correction
    )

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
            "is a descriptive point estimate: no group-resampled correlation uncertainty is "
            "computed here, and neither prediction accuracy nor the absence of a relation "
            "may be inferred from the unresolved contact enrichment"
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
    # alone would collapse them. The measured epsilon itself is rebuilt from the
    # declared state-level repair, never read raw, whenever a declaration is given.
    frozen = json.loads(Path(args.measured).read_text())
    measured_sha256 = sha256_file(args.measured)
    declaration = None
    declaration_sha256 = None
    if args.indel_declaration is not None:
        declaration = json.loads(Path(args.indel_declaration).read_text())
        declaration_sha256 = sha256_file(args.indel_declaration)
    measured, correction, absent = corrected_measured(
        frozen, declaration, cohort_sha256=measured_sha256
    )

    declared_boundary = float(published["declaration"]["rsa_boundary"])
    arms = [
        analyse_arm(gm.open_extraction(root), assays, measured, annotation, absent,
                    declared_boundary=declared_boundary, correction=correction)
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
                      "sha256": declaration_sha256,
                      "cohort_sha256": measured_sha256,
                      "states_dropped": sorted(
                          {row["sequence"] for row in correction["absent_states"]}),
                      "states_corrected": sorted(
                          {row["sequence"] for row in correction["corrected_states"]})}
            ),
        },
        "measured_correction": correction,
        "estimator": {
            "matching": "coarsened exact matching on the published separation, burial and "
                        "hydrophobicity cells, with the published RSA boundary rule",
            "bootstrap": "group then site pair inside the drawn group",
            "declaration": published["declaration"],
        },
        "measured_quoted": {
            "label": ("historical published comparator, quoted from the published artefact "
                      "and not recomputed here; it is not the state-level-repaired estimate "
                      "reported under each arm's measured_recomputed block"),
            "support": args.support,
            "unit": "kcal_mol",
            "heavy_atom": published_measured(published, support=args.support,
                                             definition="heavy_atom"),
            "cb": published_measured(published, support=args.support, definition="cb"),
        },
        "arms": arms,
        "limitations": [
            "The measured contact enrichment on this support is unresolved, and the model-side "
            "contact enrichment is a distinct estimand carrying its own precision: the "
            "unresolved measured enrichment neither scores the model contrast as accuracy nor "
            "shows the model contrast absent.",
            "The rank agreement between the model and measured interaction terms is computed "
            "descriptively without biological-group resampling, so it carries no group-level "
            "inference and licenses no absence or equivalence claim.",
            "The source study selected these site pairs as coupling tables, so contacts are the "
            "majority arm and the control arm is small; the effective control site-pair count "
            "is reported with every estimate.",
            "The model endpoint is in nats and the measured one in kcal/mol; only ranks and "
            "the sign of a contrast are comparable between them.",
            "The cycle-level agreement joins the measured cohort on the background and on the "
            "double state's own substitution label, which is unique inside a background; a "
            "site pair carries many cycles and cycles within a background are not independent "
            "units.",
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
