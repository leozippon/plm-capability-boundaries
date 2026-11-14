#!/usr/bin/env python3
"""Probe: does refusing the seven degenerate GEMME families move a ladder number?

`docs/explicit-pair-ablation.md` applies the statistical-baseline-ladder
declaration's own `refused_degenerate` condition to the published R1 GEMME
table, which left it unapplied: 896 of 25,728 rows over seven references, each
the only assay of its family cluster, so seven of the ladder's 163 equal-weighted
families carry a GEMME column with no within-assay variance. That reading could
state the exposure exactly and could not state whether any published ladder
number moves, because establishing it needs a refit.

This is the cheap evidence instead of either refitting everything or leaving the
question open. It refits one G-bearing control set and its joint arm on the
restricted support and measures the shift against the published simultaneous band
half-width. It decides whether a full refit is worth commissioning; it is not a
replacement for one and publishes no new ladder result.

What is compared, declared here before anything is fitted
=========================================================

Everything below is a constant of this file, fixed before the first fit.

Designs: ``PEG``, the P+E+G control arm, and ``MPEG``, its joint arm. ``PEG``
carries no model block, so one fit of it serves every arm and the control arm's
own shift is measured for all 33 arms at once; only the forward increment
``MPEG - PEG`` needs a fit per arm.

Seed: 20260923, the first of the three declared split seeds. One seed, so every
published quantity this compares against is recomputed at that seed from
``full/assay-metrics.json`` rather than taken from the three-seed-averaged
published hypothesis.

Arms: five, chosen to span the published P+E+G forward increment rather than for
convenience -- the arms nearest its minimum, lower quartile, median, upper
quartile and maximum over the 33 (``PROBE_ARMS``). The choice reads the published
fit's increment and nothing this probe produces.

Three comparisons, in this order, so the two mechanisms do not hide each other:

``aggregation_only``
    the published fit's own family-equal mean over the 156 retained families
    minus its mean over all 163. No refit: this is what dropping seven families
    from the average does on its own.
``fit_effect``
    this probe's restricted-support fit over its 156 families minus the published
    fit over the same 156. This is what removing the 896 rows does to the fit.
``total``
    this probe over 156 minus the published fit over 163, which is what a reader
    of the published number would see change.

Reproduction gate: before the restricted fit is read, the same designs are
refitted on the *unchanged* 25,728-row support at the same seed and each
prediction's digest must equal the published ``prediction_sha256`` of that cell.
An archived fit is used for nothing else. If the gate fails, the comparison is
void and that is the whole report.

Decision rule, also fixed here
==============================

The published Spearman family's native simultaneous half-width is
:data:`BAND_HALF_WIDTH`. Let ``worst`` be the largest absolute shift over every
comparison, the control arm and all five arms' increments.

* ``worst < 0.10 * BAND_HALF_WIDTH`` -- **immaterial**: the defect is real,
  bounded and far below the resolution any published verdict rests on.
* ``worst >= BAND_HALF_WIDTH`` -- **material**: a full refit is justified, and
  this probe's own timings give its cost.
* otherwise -- **intermediate**: reported as a fraction of the half-width with no
  label, because neither statement above would be true.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "2")

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from scripts.capability.extensions import fit_statistical_ladder as runner  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.extensions import statistical_ladder as sl  # noqa: E402

PROBE_ID = "gemme_refusal_impact_probe_20261114"
SCHEMA = "gemme_refusal_impact_probe"

#: The published panel this probe reproduces and then perturbs.
PUBLISHED = ROOT / "results/extensions/statistical_ladder_20261106/full"
SCORES = ROOT / "results/extensions/statistical_ladder_20261106/accepted-scores"
OUT = ROOT / "results/R1/explicit_pair_ablation_20261114/gemme-refusal-probe"

#: The two designs. PEG carries no model block and is fitted once.
CONTROL_DESIGN = "PEG"
JOINT_DESIGN = "MPEG"

SEED = 20260923

#: Five arms spanning the published P+E+G forward increment: the arms nearest its
#: minimum, lower quartile, median, upper quartile and maximum over the 33, with
#: the published point each was chosen by. Nothing this probe computes enters the
#: choice.
PROBE_ARMS = {
    "bygpt5-medium-en": {"quantile": 0, "published_increment": -0.0006515910651231596},
    "galactica-125m": {"quantile": 25, "published_increment": -0.00010256246856966989},
    "gpt2-large": {"quantile": 50, "published_increment": 0.00023937362792966521},
    "protgpt2": {"quantile": 75, "published_increment": 0.009725177738190368},
    "progen3-3b": {"quantile": 100, "published_increment": 0.019960436565210904},
}

#: The seven references the ladder declaration's own `refused_degenerate`
#: condition names on this support, and the assay each one contributes.
REFUSED_ASSAYS = (
    "A0A247D711_LISMN_Stadelmann_2021",
    "DN7A_SACS2_Tsuboyama_2023_1JIC",
    "ISDH_STAAW_Tsuboyama_2023_2LHR",
    "PPM1D_HUMAN_Miller_2022",
    "SBI_STAAM_Tsuboyama_2023_2JVG",
    "SCIN_STAAR_Tsuboyama_2023_2QFF",
    "VG08_BPP22_Tsuboyama_2023_2GP8",
)

#: The published Spearman family's native simultaneous half-width, from
#: `full/contrasts.json`: 396 columns, 10,000 draws, seed 20261106.
BAND_HALF_WIDTH = 0.05303511894416825
IMMATERIAL_FRACTION = 0.10

#: BLAS threads the published full panel ran at, per its `runtime-start.json`.
THREADS = 2

LIMITS = (
    "One control set, one joint arm, one seed and five arms are a probe and not the "
    "ladder: no published number is replaced and no new ladder result is published here.",
    "The probe measures the shift in the family-equal mean Spearman and in the forward "
    "increment. It does not re-form the simultaneous band, which would need every column "
    "of the family refitted, so a shift below the published half-width is read against "
    "that half-width and is not a statement that the band itself is unchanged.",
    "Spearman only. The rank-MSE family is not probed, and no claim is made for it.",
    "The five arms span the published increment, so the largest shift among them bounds "
    "the arms between them only under the assumption that the shift varies smoothly with "
    "the increment, which this probe does not test.",
)


def plan() -> dict:
    """The declaration, written before the first fit."""

    return {
        "schema": SCHEMA,
        "probe_id": PROBE_ID,
        "question": (
            "does refusing the seven degenerate GEMME families move the P+E+G control arm's "
            "performance, or any arm's forward increment, by enough to matter against the "
            "published simultaneous band half-width"
        ),
        "designs": {"control": CONTROL_DESIGN, "joint": JOINT_DESIGN,
                    "note": "PEG carries no model block, so its shift holds for all 33 arms"},
        "seed": SEED,
        "arms": PROBE_ARMS,
        "arm_choice": (
            "the arms nearest the minimum, lower quartile, median, upper quartile and maximum "
            "of the published P+E+G forward increment over the 33; chosen on the published fit "
            "and on nothing this probe produces"
        ),
        "refused_assays": list(REFUSED_ASSAYS),
        "comparisons": {
            "aggregation_only": "published fit over the 156 retained families minus over all 163",
            "fit_effect": "this probe over its 156 families minus the published fit over the same 156",
            "total": "this probe over 156 minus the published fit over 163",
        },
        "reproduction_gate": (
            "both designs refitted on the unchanged 25,728-row support at this seed must "
            "reproduce the published prediction digest of every probed cell; an archived fit is "
            "used for nothing else, and a gate failure voids the comparison"
        ),
        "band_half_width": BAND_HALF_WIDTH,
        "decision_rule": {
            "immaterial": f"largest absolute shift < {IMMATERIAL_FRACTION} * {BAND_HALF_WIDTH}",
            "material": f"largest absolute shift >= {BAND_HALF_WIDTH}; a full refit is justified",
            "intermediate": "neither; reported as a fraction of the half-width with no label",
        },
        "threads": THREADS,
        "limits": list(LIMITS),
    }


def published_rows() -> list[dict]:
    """The published per-assay metric rows, this seed and these arms only."""

    rows = json.loads((PUBLISHED / "assay-metrics.json").read_text(encoding="utf-8"))
    kept = [row for row in rows if int(row["seed"]) == SEED and row["model"] in PROBE_ARMS]
    if len(kept) != len(PROBE_ARMS) * 201:
        raise SystemExit(
            f"expected {len(PROBE_ARMS) * 201} published rows at seed {SEED}, found {len(kept)}"
        )
    return kept


def family_means(rows, *, designs) -> dict[str, dict[str, float]]:
    """Family-equal mean Spearman per arm and design, through the declared aggregation."""

    record = sl.absolute_matrix(
        rows, "spearman", designs=designs, models=sorted(PROBE_ARMS), seeds=(SEED,)
    )
    if record["undefined"]:
        raise SystemExit(f"undefined per-assay correlation: {record['undefined'][:3]}")
    matrix = np.asarray(record["matrix"], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise SystemExit("the family matrix is not finite; the aggregation is undefined")
    point = matrix.mean(axis=0)
    means: dict[str, dict[str, float]] = {}
    for (model, design), value in zip(record["labels"], point, strict=True):
        means.setdefault(model, {})[design] = float(value)
    means["_families"] = len(record["families"])
    return means


def control_arm(means: dict) -> float:
    """PEG's value, which must be identical across arms because it carries no model."""

    values = {round(means[arm][CONTROL_DESIGN], 12) for arm in PROBE_ARMS}
    if len(values) != 1:
        raise SystemExit(f"the model-independent control arm differs across arms: {values}")
    return float(next(iter(values)))


def increments(means: dict) -> dict[str, float]:
    return {arm: means[arm][JOINT_DESIGN] - means[arm][CONTROL_DESIGN] for arm in PROBE_ARMS}


def fit_cell(prepared, measured, aids, families, signature, design, features=None):
    """One design fitted through the ladder's own declared path."""

    blocks = dict(prepared)
    if features is not None:
        blocks["M"] = sl.prepare_block("M", features[:, None], aids)
    started = time.monotonic()
    prediction, records, design_sha256 = sl.fit_design(
        blocks, measured, aids, families, signature, design
    )
    return {
        "prediction": prediction,
        "records": records,
        "design_sha256": design_sha256,
        "prediction_sha256": sl.array_digest(prediction),
        "seconds": time.monotonic() - started,
    }


def gate(fits: dict[tuple[str, str], dict]) -> dict:
    """Every probed cell's refit on the unchanged support must match the published digest."""

    checked, failures = [], []
    for (arm, design), fit in sorted(fits.items()):
        path = PUBLISHED / f"cells/{arm}-seed{SEED}-checks.json"
        declared = json.loads(path.read_text(encoding="utf-8"))["designs"]
        if design not in declared:
            failures.append({"arm": arm, "design": design, "reason": "not in the published cell"})
            continue
        expected = declared[design]
        row = {
            "arm": arm,
            "design": design,
            "published_prediction_sha256": expected["prediction_sha256"],
            "refitted_prediction_sha256": fit["prediction_sha256"],
            "identical": fit["prediction_sha256"] == expected["prediction_sha256"],
            "published_design_sha256": expected["design_sha256"],
            "refitted_design_sha256": fit["design_sha256"],
            "design_identical": fit["design_sha256"] == expected["design_sha256"],
            "published_rows": int(expected["rows"]),
        }
        checked.append(row)
        if not (row["identical"] and row["design_identical"]):
            failures.append(row)
    return {
        "status": "passed" if not failures else "failed",
        "cells": checked,
        "failures": failures,
        "reading": (
            "the probe reproduces the published fit bit for bit on the unchanged support, so a "
            "difference on the restricted support is the restriction and not the refit"
            if not failures
            else "the probe does not reproduce the published fit on the unchanged support, so it "
                 "cannot isolate the restriction; the comparison is void"
        ),
    }


def run() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"{OUT} must be new or empty; a result directory is never overwritten")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write_json(OUT / "plan.json", plan())

    with threadpool_limits(limits=THREADS):
        loading = time.monotonic()
        data = runner.load_support(ROOT, runner.SOURCE)
        sample_ids = np.asarray([row["sample_id"] for row in data["samples"]])
        tables = {
            name: runner.read_baseline_table(SCORES, name, column)
            for name, column in (("potts", sl.POTTS_COLUMN), ("gemme", sl.GEMME_COLUMN))
        }
        joined = {}
        for name, column in (("potts", sl.POTTS_COLUMN), ("gemme", sl.GEMME_COLUMN)):
            joined.update(sl.join_scores(sample_ids, tables[name], columns=(column,)))
        data["blocks"]["E"] = joined[sl.POTTS_COLUMN][:, None]
        data["blocks"]["G"] = joined[sl.GEMME_COLUMN][:, None]
        loading = time.monotonic() - loading

        aids, families = data["aids"], data["families"]
        measured = data["measured"]
        signature = data["signatures"][SEED]
        keep = ~np.isin(aids, np.asarray(REFUSED_ASSAYS))
        if int((~keep).sum()) != 896 or len(set(families[~keep])) != 7:
            raise SystemExit(
                f"the refusal removes {int((~keep).sum())} rows over "
                f"{len(set(families[~keep]))} families, not 896 over 7"
            )

        results: dict[str, dict] = {}
        timings: list[dict] = []
        for stage, mask in (("full", np.ones(len(aids), dtype=bool)), ("restricted", keep)):
            prepared = {
                name: sl.prepare_block(name, data["blocks"][name][mask], aids[mask])
                for name in ("P", "E", "G")
            }
            control = fit_cell(prepared, measured[mask], aids[mask], families[mask],
                              signature, CONTROL_DESIGN)
            timings.append({"stage": stage, "design": CONTROL_DESIGN, "arm": None,
                            "seconds": control["seconds"]})
            fits = {}
            rows = []
            for arm in sorted(PROBE_ARMS):
                joint = fit_cell(prepared, measured[mask], aids[mask], families[mask],
                                 signature, JOINT_DESIGN, features=data["features"][arm][mask])
                timings.append({"stage": stage, "design": JOINT_DESIGN, "arm": arm,
                                "seconds": joint["seconds"]})
                fits[(arm, CONTROL_DESIGN)] = control
                fits[(arm, JOINT_DESIGN)] = joint
                rows.extend(sl.assay_rows(
                    {CONTROL_DESIGN: control["prediction"], JOINT_DESIGN: joint["prediction"]},
                    measured[mask], aids[mask], families[mask],
                    model=arm, seed=SEED, designs=(CONTROL_DESIGN, JOINT_DESIGN)))
            results[stage] = {"fits": fits, "rows": rows}

    gate_record = gate(results["full"]["fits"])
    write_json(OUT / "gate.json", gate_record)
    if gate_record["status"] != "passed":
        write_json(OUT / "probe.json", {
            "schema": SCHEMA, "probe_id": PROBE_ID, "status": "void",
            "outcome": "not_comparable",
            "reason": "the reproduction gate failed; see gate.json",
            "gate": gate_record, "plan": plan(),
        })
        print("gate FAILED; the comparison is void")
        return 1

    designs = (CONTROL_DESIGN, JOINT_DESIGN)
    published = published_rows()
    retained = set(aids[keep].tolist())
    published_163 = family_means(published, designs=designs)
    published_156 = family_means([row for row in published if row["assay"] in retained],
                                 designs=designs)
    probe_156 = family_means(results["restricted"]["rows"], designs=designs)
    if probe_156["_families"] != 156 or published_156["_families"] != 156:
        raise SystemExit(
            f"the restricted family counts are {probe_156['_families']} and "
            f"{published_156['_families']}, not 156"
        )

    control = {
        "published_163": control_arm(published_163),
        "published_156": control_arm(published_156),
        "probe_156": control_arm(probe_156),
    }
    control["aggregation_only"] = control["published_156"] - control["published_163"]
    control["fit_effect"] = control["probe_156"] - control["published_156"]
    control["total"] = control["probe_156"] - control["published_163"]

    published_increments = {"163": increments(published_163), "156": increments(published_156)}
    probe_increments = increments(probe_156)
    per_arm = {}
    for arm in sorted(PROBE_ARMS):
        per_arm[arm] = {
            "quantile_of_published_increment": PROBE_ARMS[arm]["quantile"],
            "published_three_seed_increment": PROBE_ARMS[arm]["published_increment"],
            "published_163": published_increments["163"][arm],
            "published_156": published_increments["156"][arm],
            "probe_156": probe_increments[arm],
            "aggregation_only": published_increments["156"][arm] - published_increments["163"][arm],
            "fit_effect": probe_increments[arm] - published_increments["156"][arm],
            "total": probe_increments[arm] - published_increments["163"][arm],
        }

    shifts = {f"control/{name}": abs(control[name]) for name in
              ("aggregation_only", "fit_effect", "total")}
    for arm, record in per_arm.items():
        for name in ("aggregation_only", "fit_effect", "total"):
            shifts[f"{arm}/{name}"] = abs(record[name])
    worst_label = max(shifts, key=shifts.get)
    worst = shifts[worst_label]
    if worst < IMMATERIAL_FRACTION * BAND_HALF_WIDTH:
        outcome = "immaterial"
    elif worst >= BAND_HALF_WIDTH:
        outcome = "material"
    else:
        outcome = "intermediate"

    fit_seconds = [row["seconds"] for row in timings]
    per_fit = float(np.mean(fit_seconds))
    # A full refit refits every G-bearing design on the restricted support: the
    # three model-independent control arms once per seed, and the three joint arms
    # once per arm and seed.
    full_fits = 3 * 3 + 3 * 33 * 3
    write_json(OUT / "probe.json", {
        "schema": SCHEMA, "probe_id": PROBE_ID, "status": "complete",
        "outcome": outcome,
        "plan": plan(),
        "gate": {"status": gate_record["status"], "cells": len(gate_record["cells"])},
        "support": {
            "full_rows": int(len(aids)), "full_families": int(len(set(families.tolist()))),
            "restricted_rows": int(keep.sum()),
            "restricted_families": int(len(set(families[keep].tolist()))),
            "rows_refused": int((~keep).sum()), "families_refused": 7,
        },
        "control_arm": control,
        "forward_increments": per_arm,
        "shifts": {
            "largest_absolute": worst,
            "largest_absolute_at": worst_label,
            "band_half_width": BAND_HALF_WIDTH,
            "largest_as_fraction_of_half_width": worst / BAND_HALF_WIDTH,
            "all": shifts,
        },
        "cost": {
            "mean_seconds_per_fit": per_fit,
            "fits_in_this_probe": len(fit_seconds),
            "probe_seconds": time.monotonic() - started,
            "support_load_seconds": loading,
            "full_refit_fits": full_fits,
            "full_refit_cpu_hours_at_this_rate": full_fits * per_fit / 3600.0,
            "note": (
                "a full refit of every G-bearing design on the restricted support: three "
                "model-independent control arms per seed plus three joint arms per arm and seed, "
                "at three seeds, excluding the support load and the band recomputation"
            ),
        },
        "timings": timings,
        "inputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (PUBLISHED / "assay-metrics.json", PUBLISHED / "contract.json",
                         SCORES / "potts/scores.json", SCORES / "gemme/scores.json")
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (Path(__file__).resolve(),
                         ROOT / "scripts/capability/extensions/fit_statistical_ladder.py",
                         ROOT / "src/capability/extensions/statistical_ladder.py")
        },
        "limits": list(LIMITS),
    })
    print(f"outcome={outcome} worst={worst:.3e} at {worst_label} "
          f"({worst / BAND_HALF_WIDTH:.2%} of the half-width) in "
          f"{time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
