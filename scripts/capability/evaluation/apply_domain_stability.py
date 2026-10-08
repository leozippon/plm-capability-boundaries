#!/usr/bin/env python3
"""Apply a validated stability head to the generated cohort, or refuse to.

The gate comes first. This stage reads the fit stage's verdict and, if an
instrument did not clear both declared conditions on the cross-dataset check, it
writes a completion record saying so and scores nothing with that instrument. A
refusal is a result and is recorded as one; it is not a gap in a table.

What is reported when the gate does pass: predicted absolute folding free energy
in kcal/mol for the generated sequences of the frozen generation-evaluation
cohort that fall inside the licensed length band, against the whole natural
records that cohort already length-matched to them, per arm and per termination
stratum, with the support of each stratum beside its estimate.

Two limitations are carried in the artefact rather than left to be noticed. The
licensed band excludes the budget-censored stratum entirely -- a continuation
stopped at the token budget is several times longer than any domain in the
training data -- so this says nothing about truncation. And predicted free energy
is kept in its own block, with its own unit, separate from the structural
confidences, which :func:`domain_stability.require_comparable` enforces rather
than recommends.

CPU only; ``--device`` is accepted because the campaign queue injects it.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.evaluation import domain_stability as ds  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402

COMPLETION = "domain_stability_evaluation.json"
SCHEMA_VERSION = "d1_domain_stability_evaluation_v1"


def run(args: argparse.Namespace) -> dict[str, Any]:
    gp.require_fresh_out(args.out, COMPLETION)
    fit = json.loads((args.fit / "domain_stability_fit.json").read_text(encoding="utf-8"))
    band = tuple(int(value) for value in fit["licensed_band"])
    gates = fit["gates"]

    # The guard this whole experiment exists to install, exercised here so the
    # artefact carries evidence that it is live rather than a claim that it is.
    # A prediction against the measurement of the same physical quantity is the
    # validation and must be allowed; a structural confidence or an undocumented
    # thermostability score against a free energy must not be.
    ds.require_comparable("predicted_delta_g", "measured_delta_g")
    separation_check: dict[str, Any] = {
        "enforced": True,
        "allowed": "predicted_delta_g vs measured_delta_g, both folding_free_energy in kcal/mol",
        "refused": {},
    }
    for other in ("esmfold2_mean_ca_plddt", "esmfold2_ptm", "prime_value_head"):
        try:
            ds.require_comparable("predicted_delta_g", other)
        except ValueError as error:
            separation_check["refused"][other] = str(error)
        else:  # pragma: no cover - the guard must raise
            raise RuntimeError(
                f"require_comparable accepted {other!r} against a free energy; the "
                "separation this experiment depends on is not enforced"
            )

    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    apply_rows = [row for row in rows if row.get("split") == "apply"]
    if not apply_rows:
        raise SystemExit(f"{args.rows} carries no applicable row")
    in_band = [row for row in apply_rows if band[0] <= int(row["length"]) <= band[1]]
    kept = {str(row["id"]) for row in in_band}
    out_of_band = [row for row in apply_rows if str(row["id"]) not in kept]
    # A pair survives only if both members are in band; a one-sided pair would
    # break the matching that makes the contrast a matched one.
    by_pair: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in in_band:
        by_pair[str(row["pair_id"])][str(row["role"])] = row
    pairs = {
        pair_id: members
        for pair_id, members in by_pair.items()
        if "generated" in members and "natural" in members
    }

    passed = {name: bool(gate["passed"]) for name, gate in gates.items()}
    applied: dict[str, Any] = {}
    if passed.get("plm") and pairs:
        model = np.load(args.fit / "model_plm.npz")
        loaded = {
            "coefficients": model["coefficients"],
            "intercept": float(model["intercept"][0]),
            "mean": model["mean"],
            "scale": model["scale"],
        }
        index = json.loads((args.embeddings / "embedding_index.json").read_text(encoding="utf-8"))
        embeddings = np.load(args.embeddings / "embeddings.npy", mmap_mode="r")
        position = {identifier: i for i, identifier in enumerate(index["ids"])}
        ordered = sorted(pairs)
        members = [pairs[pair_id][role] for pair_id in ordered for role in ("generated", "natural")]
        ds.require_in_band([int(row["length"]) for row in members], band, label="generated cohort")
        cheap = np.stack([ds.composition_features(str(row["sequence"])) for row in members])
        block = np.asarray(embeddings[[position[row["id"]] for row in members]], dtype=np.float64)
        predicted = ds.predict_with(loaded, np.hstack([cheap, block]))
        values = {str(row["id"]): float(value) for row, value in zip(members, predicted)}

        per_arm: dict[str, Any] = {}
        arms = sorted({pairs[pair_id]["generated"]["arm"] for pair_id in ordered})
        for arm in arms:
            strata: dict[str, Any] = {}
            for stratum in sorted(gp.STRATA):
                selected = [
                    pair_id
                    for pair_id in ordered
                    if pairs[pair_id]["generated"]["arm"] == arm
                    and pairs[pair_id]["generated"]["stratum"] == stratum
                ]
                if not selected:
                    strata[stratum] = {
                        "status": "no_support_in_licensed_band",
                        "n_pairs": 0,
                        "unit_floor": gp.bootstrap_unit_floor(0),
                    }
                    continue
                generated = [values[pairs[pair_id]["generated"]["id"]] for pair_id in selected]
                natural = [values[pairs[pair_id]["natural"]["id"]] for pair_id in selected]
                contrast = gp.matched_contrast(
                    generated, natural, selected, seed=args.seed, n_bootstrap=args.draws
                )
                contrast["status"] = "measured"
                contrast["unit"] = ds.QUANTITIES["predicted_delta_g"]["unit"]
                contrast["generated_length_mean"] = float(
                    np.mean([int(pairs[p]["generated"]["length"]) for p in selected])
                )
                strata[stratum] = contrast
            per_arm[arm] = strata
        applied["plm"] = {
            "quantity": "predicted_delta_g",
            "quantity_declaration": dict(ds.QUANTITIES["predicted_delta_g"]),
            "n_pairs": len(ordered),
            "per_arm": per_arm,
            "pool_summary": {
                "generated_mean": float(
                    np.mean([values[pairs[p]["generated"]["id"]] for p in ordered])
                ),
                "natural_mean": float(
                    np.mean([values[pairs[p]["natural"]["id"]] for p in ordered])
                ),
                "unit": ds.QUANTITIES["predicted_delta_g"]["unit"],
            },
        }
        sidecar = args.out / "generated_stability.jsonl"
        sidecar.write_text(
            "".join(
                json.dumps(
                    {
                        "id": str(row["id"]),
                        "pair_id": row["pair_id"],
                        "role": row["role"],
                        "arm": row["arm"],
                        "stratum": row["stratum"],
                        "length": int(row["length"]),
                        "predicted_delta_g_kcal_per_mol": values[str(row["id"])],
                    },
                    sort_keys=True,
                )
                + "\n"
                for row in members
            ),
            encoding="utf-8",
        )
        applied["plm"]["predictions_jsonl"] = str(sidecar)
        applied["plm"]["predictions_sha256"] = sha256_file(sidecar)
    elif pairs:
        applied["plm"] = {
            "status": "refused_by_validation_gate",
            "gate": gates.get("plm"),
            "consequence": (
                "no generated sequence was scored for free energy. The predictor did "
                "not clear its declared cross-dataset conditions, so a number from it "
                "would not be evidence of stability"
            ),
        }

    if "prime" in gates:
        applied["prime"] = {
            "status": "second_opinion_withheld"
            if not passed.get("prime")
            else "second_opinion_available",
            "gate": gates["prime"],
            "quantity_declaration": dict(ds.QUANTITIES["prime_value_head"]),
            "note": (
                "the third-party checkpoint is read as rank corroboration only, and "
                "only if it cleared the same gate. It is never converted to kcal/mol "
                "and never averaged with the fitted head"
            ),
        }

    stratum_counts = collections.Counter(
        (pairs[pair_id]["generated"]["arm"], pairs[pair_id]["generated"]["stratum"])
        for pair_id in pairs
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "licensed_band": list(band),
        "gates": gates,
        "gate_conditions": dict(ds.GATE_CONDITIONS),
        "quantity_separation": {
            **separation_check,
            "statement": (
                "predicted folding free energy and predicted structural confidence are "
                "different quantities with different units and are never substituted. "
                "The guard above refuses a comparison between them at runtime"
            ),
            "quantities": dict(ds.QUANTITIES),
        },
        "support": {
            "n_apply_rows": len(apply_rows),
            "n_in_licensed_band": len(in_band),
            "n_outside_licensed_band_unscored": len(out_of_band),
            "n_matched_pairs_in_band": len(pairs),
            "per_arm_stratum": {f"{arm}::{stratum}": count for (arm, stratum), count in sorted(stratum_counts.items())},
            "usable_above_unit_floor": sorted(
                f"{arm}::{stratum}"
                for (arm, stratum), count in stratum_counts.items()
                if not gp.bootstrap_unit_floor(count)["degenerate"]
            ),
        },
        "applied": applied,
        "limitations": {
            "censored_stratum_absent": (
                "the budget-censored stratum has no support inside the licensed band, "
                "because a continuation stopped at the token budget is several times "
                "longer than any domain in the training data. This result therefore "
                "says nothing about truncation, which the structural result does cover"
            ),
            "same_assay_technology": (
                "the training and cross-dataset measurements share an assay technology, "
                "so a passing gate shows transfer across sequence populations and not "
                "across measurement technology"
            ),
            "not_a_measurement": (
                "this is a prediction for a sequence, in kcal/mol, from a linear head on "
                "frozen embeddings. No generated sequence has been measured"
            ),
            "small_domains_only": (
                f"the licensed band is {band[0]}-{band[1]} residues. Most generated "
                "sequences in the cohort are longer and are deliberately unscored"
            ),
        },
        "sources": {
            "fit": str(args.fit),
            "rows": str(args.rows),
            "rows_sha256": sha256_file(args.rows),
            "embeddings": str(args.embeddings),
        },
        "n_bootstrap": int(args.draws),
        "seed": int(args.seed),
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True, help="the fit stage's out dir")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.draws < 1000:
        parser.error("--draws below 1000 gives percentile intervals this package will not publish")
    record = run(args)
    print(
        json.dumps(
            {
                "status": record["status"],
                "licensed_band": record["licensed_band"],
                "n_matched_pairs_in_band": record["support"]["n_matched_pairs_in_band"],
                "gates": {name: gate["passed"] for name, gate in record["gates"].items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
