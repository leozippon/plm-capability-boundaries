#!/usr/bin/env python3
"""E10 refit: the remote-homology gate's increments, re-read under a named corpus (CPU).

The prior gate banded its family groups by searching a UniRef50 snapshot with no
recorded release. Re-banding the same groups against UniRef90 2026_03 -- the
corpus the registry names for the arms that read positive on the remote stratum --
moves 34 of its 179 groups. This stage asks what that does to the numbers.

**Nothing about the fit changes, and that is the point.** The gate's own design
already separates the fit from the readout: the same folds, the same training
groups and the same held-out predictions are read out again over whichever groups
a stratum holds, and ``group_errors`` renormalises the nesting inside the retained
rows. So the increments under a second banding need no new estimator and no new
cohort -- only a second group-to-stratum map, which
:func:`~src.capability.context.remote_homology.stratum_keep` already takes as an
argument. Every recipe here is imported: the panel assembly, the control set, the
nested folds, the ridge, the nonlinear-additive nuisance, the group-equal squared
error and the group bootstrap all come from the gate's own modules, and the
per-arm block assembly comes from the gate's own fit entry point rather than a
second copy of it.

**The fit is reproduced, not assumed.** The prior gate retains per-seed summaries
but no per-row predictions, so the predictions have to be recomputed before any
stratum can be re-read. A recomputation that does not reproduce the frozen numbers
would be a different fit wearing the same name, so this stage recomputes the
*original* banding as well and refuses to report anything unless its per-seed
full-support and per-stratum points match the retained ``fit_<arm>.json`` to
:data:`REPRODUCTION_TOLERANCE`.

**What a band is and is not.** A band is a property of one search against one
corpus release. UniRef90 2026_03 post-dates every checkpoint here; representative
churn moves entries in both directions; and for the UniRef-plus-BFD and
UniRef-plus-ColabFoldDB mixtures the unsearched component means a *remote* call is
not conservative. Those directions are recorded per arm by
:mod:`~src.capability.context.corpus_distance` and are carried into this artefact.
Reference-database similarity is not pretraining exposure, and this stage never
reports it as such.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.capability.entrypoints import stage_path  # noqa: E402
from src.capability.context import corpus_distance as CD  # noqa: E402
from src.capability.context import homology_context as H  # noqa: E402
from src.capability.context import remote_homology as RH  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.mutation.external_confirmation import (  # noqa: E402
    ARM_CANDIDATE_BLOCKS,
    MODEL_BLOCKS,
    SPLIT_SEEDS,
    build_panel,
    fold_predictions,
    group_errors,
    interval,
    load_cohort,
    paired_increment,
    qualify,
    require_blas_threads,
    row_identity,
    secondary_control_set,
)
from src.capability.stability.stability_gate import load_profiles  # noqa: E402

EXPECT = "remote_strata_refit.json"

#: The gate's own measurement directory.
GATE = "results/R5/remote_homology_20260924"

#: Largest admissible disagreement between a recomputed per-seed point estimate
#: and the retained one. The recomputation runs the same code on the same bytes,
#: so the only expected difference is floating-point summation order.
REPRODUCTION_TOLERANCE = 1e-9

#: The two bandings read off one fit.
FROZEN_BANDING = "uniref50_snapshot_frozen"
REVISED_BANDING = "uniref90_2026_03"


def runtime() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "device": "cpu",
        "peak_rss_bytes": usage.ru_maxrss * 1024,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "threads_requested": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def gate_module():
    """The gate's own fit entry point, for its per-arm block assembly."""

    path = stage_path(ROOT, "fit_nested_singles.py")
    spec = importlib.util.spec_from_file_location("fit_nested_singles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def revised_assignment(contexts: dict) -> tuple[dict[str, str], dict]:
    """The group-to-stratum map implied by this run's own corpus search.

    Built with the prior gate's own grouping rule -- close only if every member
    bands at or above 70% identity, remote only if every member falls below it --
    imported rather than restated, so a group that moves cannot be moving because
    a second convention was applied.
    """

    gate = contexts.get("remote_gate")
    if not gate:
        raise SystemExit(
            "the retrieval artefact carries no remote-gate re-band; run the retrieval "
            "stage without --no-remote-gate"
        )
    labels = {row["background"]: row["group"] for row in gate["backgrounds"]}
    bands = {row["background"]: row["new_band"] for row in gate["backgrounds"]}
    assignment, report = RH.group_strata(labels, bands)
    return assignment, report


def stratum_records(panel, predictions, baselines, assignment, *, names=("close", "remote")):
    """One banding's per-stratum readout of one seed's held-out predictions."""

    out = {}
    for name in names:
        keep = RH.stratum_keep(panel["group"], assignment, name)
        groups = sorted(set(panel["group"][keep].tolist()))
        record = {
            "groups": len(groups),
            "rows": int(keep.sum()),
            "clears_the_unit_floor": RH.unit_floor_cleared(len(groups)),
        }
        for role, baseline in baselines.items():
            record[f"{role}_likelihood"] = paired_increment(
                panel, predictions, f"{baseline}_M", baseline, keep=keep
            )
            record[f"{role}_representation"] = paired_increment(
                panel, predictions, f"{baseline}_R", baseline, keep=keep
            )
        out[name] = record
    return out


def power_of(record: dict, *, groups: int) -> dict:
    """The group count this increment would need to resolve its own estimate.

    An unresolved stratum is not a null one. The E08 audit measured the gate's
    close control as underpowered by one to two orders of magnitude, so every
    stratum here reports what it would take to resolve rather than leaving a
    spanning interval to be read as absence.
    """

    point = record.get("point")
    span = record.get("interval")
    if point is None or span is None or groups < 1:
        return {"resolvable": None, "reason": "no estimate on this stratum"}
    half = max(abs(span[1] - point), abs(point - span[0]))
    if half <= 0:
        return {"resolvable": None, "reason": "degenerate interval"}
    # The percentile interval is what this estimator delivers, so its own
    # half-width is the yardstick rather than a normal quantile imported from
    # elsewhere: at a fixed effect the half-width falls as 1/sqrt(groups).
    ratio = half / abs(point) if point else None
    return {
        "half_width": half,
        "implied_half_width_over_point": ratio,
        "groups": groups,
        "groups_required_to_resolve": (
            None if ratio is None else int(np.ceil(groups * ratio * ratio))
        ),
        "basis": "half-width of the group-bootstrap percentile interval; 1/sqrt(groups)",
    }


def verify_reproduction(per_seed: dict, reference: dict, *, tolerance: float) -> dict:
    """Refuse a recomputation that does not reproduce the retained fit."""

    checks = []
    for seed, record in per_seed.items():
        frozen = reference["per_seed"][seed]
        for key in ("primary_likelihood", "secondary_likelihood", "primary_baseline_mse"):
            observed = record[key]["point"]
            expected = frozen[key]["point"]
            checks.append(
                {
                    "seed": seed,
                    "quantity": key,
                    "recomputed": observed,
                    "retained": expected,
                    "absolute_difference": abs(observed - expected),
                }
            )
        for name in ("close", "remote"):
            observed = record["strata"][FROZEN_BANDING][name]["primary_likelihood"]["point"]
            expected = frozen[f"stratum_{name}"]["primary_likelihood"]["point"]
            checks.append(
                {
                    "seed": seed,
                    "quantity": f"stratum_{name}_primary_likelihood",
                    "recomputed": observed,
                    "retained": expected,
                    "absolute_difference": abs(observed - expected),
                }
            )
    worst = max(checks, key=lambda row: row["absolute_difference"])
    if worst["absolute_difference"] > tolerance:
        raise RuntimeError(
            "the recomputed fit does not reproduce the retained one: "
            f"{worst['quantity']} on seed {worst['seed']} differs by "
            f"{worst['absolute_difference']:.3e} (tolerance {tolerance:.1e}). No stratum "
            "may be re-read off a fit that is not the frozen one."
        )
    return {
        "tolerance": tolerance,
        "checks": len(checks),
        "worst": worst,
        "reproduced": True,
    }


def fit_arm(args, arm: str, loaded: dict, panel: dict, controls: dict, revised: dict) -> dict:
    import torch

    gate = gate_module()
    blocks_arm, manifest = gate.load_arm(
        args.gate / "extractions" / f"rh-{arm}", arm, loaded["units"], loaded["sha256"]
    )
    blocks = dict(panel["blocks"], **blocks_arm)
    qualified = tuple(controls["qualified_control_set"])
    secondary, secondary_record = secondary_control_set(controls)
    designs = {}
    for prefix, columns in (("S", qualified), ("S2", secondary)):
        designs[prefix] = tuple(columns)
        for extra in ARM_CANDIDATE_BLOCKS:
            designs[f"{prefix}_{extra}"] = (*columns, extra)
        for block in MODEL_BLOCKS:
            designs[f"{prefix}_{block}"] = (*columns, block)
            for extra in ARM_CANDIDATE_BLOCKS:
                designs[f"{prefix}_{extra}_{block}"] = (*columns, extra, block)
    first_stage = tuple(block for block in qualified if block != "G")

    started = time.monotonic()
    main_pass = {
        seed: fold_predictions(
            panel, blocks, designs, seed=seed, first_stage=first_stage, device="cpu"
        )
        for seed in args.seeds
    }
    verdict = qualify(
        {
            seed: paired_increment(panel, main_pass[seed]["predictions"], "S_T", "S")["point"]
            for seed in args.seeds
        }
    )
    suffix = "_T" if verdict["qualified"] else ""
    baselines = {"primary": f"S{suffix}", "secondary": f"S2{suffix}"}

    frozen_assignment = loaded["strata"]
    per_seed: dict[str, dict] = {}
    for seed in args.seeds:
        predictions = main_pass[seed]["predictions"]
        record: dict = {}
        for role, baseline in baselines.items():
            record[f"{role}_likelihood"] = paired_increment(
                panel, predictions, f"{baseline}_M", baseline
            )
            record[f"{role}_representation"] = paired_increment(
                panel, predictions, f"{baseline}_R", baseline
            )
            _, errors = group_errors(
                panel["target"],
                predictions[baseline],
                panel["group"],
                panel["domain"],
                panel["site"],
            )
            record[f"{role}_baseline_mse"] = interval(errors)
        record["strata"] = {
            FROZEN_BANDING: stratum_records(
                panel, predictions, baselines, frozen_assignment
            ),
            REVISED_BANDING: stratum_records(panel, predictions, baselines, revised),
        }
        per_seed[str(seed)] = record

    reference = json.loads((args.gate / "fits" / f"fit_{arm}.json").read_text())
    reproduction = verify_reproduction(per_seed, reference, tolerance=args.tolerance)

    outcomes = {}
    for banding in (FROZEN_BANDING, REVISED_BANDING):
        resolved = {}
        for name in ("close", "remote"):
            for role in baselines:
                points = [
                    per_seed[str(seed)]["strata"][banding][name][f"{role}_likelihood"]
                    for seed in args.seeds
                ]
                resolved[(name, role)] = bool(
                    all(p["excludes_zero"] and p["point"] > 0 for p in points)
                )
        outcomes[banding] = {
            "rule": (
                "an arm resolves on a stratum when all three per-seed intervals of its "
                "paired increment over that stratum exclude zero above it"
            ),
            "outcomes": {
                role: RH.outcome_record(resolved[("close", role)], resolved[("remote", role)])
                for role in baselines
            },
            "power": {
                name: {
                    role: power_of(
                        per_seed[str(args.seeds[0])]["strata"][banding][name][
                            f"{role}_likelihood"
                        ],
                        groups=per_seed[str(args.seeds[0])]["strata"][banding][name]["groups"],
                    )
                    for role in baselines
                }
                for name in ("close", "remote")
            },
        }
    return {
        "arm": arm,
        "baselines": baselines,
        "tokenisation_verdict": verdict,
        "secondary_control_derivation": secondary_record,
        "extraction_identity": manifest["identity"],
        "reference_fit_sha256": sha256_file(args.gate / "fits" / f"fit_{arm}.json"),
        "reproduction": reproduction,
        "per_seed": per_seed,
        "stratified_outcome": outcomes,
        "elapsed_seconds": time.monotonic() - started,
        "torch_version": torch.__version__,
    }


def run(args: argparse.Namespace) -> None:
    numeric = require_blas_threads()
    import torch

    torch.set_num_threads(numeric["pinned_threads"])

    contexts = json.loads(args.homologs.read_text(encoding="utf-8"))
    H.require_declaration(contexts, scope="retrieval")
    corpus = contexts["corpus_identity"]
    if not corpus.get("named_release"):
        raise SystemExit(
            f"corpus {corpus['corpus_id']!r} carries no published release record; the "
            "whole point of this refit is to band against a corpus that can be named"
        )
    revised, revised_report = revised_assignment(contexts)

    loaded = load_cohort(args.gate / "cohort.json")
    controls = json.loads((args.gate / "controls_qualification.json").read_bytes())
    if controls["cohort_sha256"] != loaded["sha256"]:
        raise SystemExit("the control qualification was run against a different cohort")
    wildtypes = {row["name"]: row["wildtype"] for row in loaded["units"]}
    profiles, _ = load_profiles(args.gate / "profile_features.npz", wildtypes)
    panel = build_panel(loaded["units"], profiles)
    identity = row_identity(panel["group"], panel["site"], panel["target"])
    if identity != controls["row_identity_sha256"]:
        raise SystemExit("the panel rows differ from the ones the controls were qualified on")
    missing = sorted(set(loaded["strata"]) - set(revised))
    if missing:
        raise SystemExit(f"{len(missing)} cohort groups carry no revised band: {missing[:5]}")

    trace = CD.traceability(args.arms)
    eligibility = CD.arms_for_corpus(trace, corpus["corpus_id"])
    refused = [row["arm"] for row in eligibility["refused"]]
    if refused:
        raise SystemExit(
            f"these arms are not traceable to {corpus['corpus_id']}: {refused}. This refit "
            "reads only arms whose declared corpus the searched corpus can stand for."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for number, arm in enumerate(args.arms, start=1):
        records.append(fit_arm(args, arm, loaded, panel, controls, revised))
        print(
            f"{number}/{len(args.arms)} {arm}: reproduced "
            f"{records[-1]['reproduction']['reproduced']}, "
            f"{records[-1]['elapsed_seconds']:.0f}s",
            flush=True,
        )

    migration = contexts["remote_gate"]
    write_json(
        args.out / EXPECT,
        {
            "schema_version": "remote_strata_refit_v1",
            "stage": "refit_remote_strata",
            "status": "complete",
            **H.declaration_digests(),
            "question": (
                "do the remote-homology gate's model increments change when its family "
                "groups are banded against the corpus the registry names for the arms "
                "being read, rather than against an unidentified UniRef50 snapshot?"
            ),
            "corpus_identity": corpus,
            "bandings": {
                FROZEN_BANDING: {
                    "source": "the gate's own frozen cohort",
                    "groups": {
                        name: sum(1 for value in loaded["strata"].values() if value == name)
                        for name in sorted(set(loaded["strata"].values()))
                    },
                },
                REVISED_BANDING: {
                    "source": "this run's search, under the gate's own grouping rule",
                    "groups": {
                        "close": revised_report["close_groups"],
                        "remote": revised_report["remote_groups"],
                        "mixed": revised_report["mixed_groups"],
                    },
                    "group_report": revised_report,
                },
            },
            "group_migration": {
                key: migration[key]
                for key in (
                    "group_migration_counts",
                    "groups_that_moved",
                    "prior_remote_groups_now_close_or_mixed",
                    "prior_remote_groups_now_close_or_mixed_fraction",
                )
            },
            "traceability": trace,
            "corpus_eligibility": eligibility,
            "corpus_relations": CD.CORPUS_RELATIONS,
            "cohort": {
                "path": str((args.gate / "cohort.json").relative_to(ROOT)),
                "sha256": loaded["sha256"],
                "endpoint": loaded["endpoint"],
                "groups": len(set(panel["group"].tolist())),
                "rows": int(len(panel["target"])),
            },
            "seeds": list(args.seeds),
            "arms": records,
            "does_not_license": (
                "a band is reference-database similarity, not pretraining exposure. The "
                "searched release post-dates every checkpoint, representative churn moves "
                "entries both ways, and where a declared corpus has an unsearched component "
                "a remote call is not conservative. Remote generalization is not claimed "
                "from a stratum that does not resolve, nor from one that resolves while its "
                "own close positive control does not."
            ),
            "limitations": list(H.LIMITATIONS)
            + [
                "The fit is the gate's own and is reproduced, not improved: the endpoint, "
                "the control set, the folds and the squared-error target are unchanged, so "
                "the close stratum's power is unchanged too.",
                "The anchor cohort cannot be asked this question at all: 163 of its 174 "
                "wild types are near-duplicates of the searched corpus and none is remote, "
                "which is why this gate's metagenome-derived panel is the only support.",
            ],
            "numeric_environment": numeric,
            "runtime": runtime(),
        },
    )
    print(f"refit {len(records)} arms under both bandings", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gate", type=Path, default=ROOT / GATE)
    parser.add_argument("--homologs", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SPLIT_SEEDS))
    parser.add_argument("--tolerance", type=float, default=REPRODUCTION_TOLERANCE)
    args = parser.parse_args()
    if tuple(args.seeds) != tuple(SPLIT_SEEDS):
        raise SystemExit(
            f"the gate's tokenisation qualification is defined over all of "
            f"{list(SPLIT_SEEDS)} and refuses a subset; a partial-seed refit would read "
            "its strata off a different baseline than the frozen fit did"
        )
    if args.device != "cpu":
        raise SystemExit(
            f"this stage is CPU-only and was given --device {args.device!r}; it refits from "
            "retained extractions and loads no checkpoint"
        )
    run(args)


if __name__ == "__main__":
    main()
