#!/usr/bin/env python3
"""Descriptive ProGen2 medium→large→xlarge comparison. Not a causal scale claim.

This stage is CPU-only. It consumes existing stage-20 and stage-29 artefacts
and a full stage-41 report. It does not load a model, does not download
weights, and does not synthesise a cross-task total.

The comparison is fixed as progen2-medium → progen2-large → progen2-xlarge.
Checkpoint differences are descriptive of those checkpoints. They are not a
parameter-count causal effect: corpus identification is a model-favouring bound
on UniRef90+BFD30, and nothing here is a claim about biological knowledge.

A run that cannot qualify all three rungs on the same stage-41 blocks does not
emit DMS or MegaScale gates.

Two cohorts here are narrower than the queues they come from, and both are
recorded rather than implied. ProteinGym declares 217 substitution assays over
174 families; 16 of them render longer than the 1024-position context every
ProGen2 rung shares, so the ladder is read on the 201 assays over 163 families
that every rung can score. MegaScale scores all 146 designs; the design census
is the 130 certified zero-hit designs, and that flag is carried by the stage-29
cohort rather than by the per-arm payloads.

The freeze lives here and the arithmetic lives in
:mod:`src.capability.models.scale_comparison`. Every rung name, census count, seed and
resample count below is this campaign's declaration and is checked here; the
paired resampling, the cohort alignment and the stage-41 qualification are one
implementation shared with the second-stage campaign, because those two must not
be free to drift apart on exactly the operations that make them comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.arms import (  # noqa: E402
    REPO,
    STAGED_SCALE_ARMS,
    UNIREF90_BFD30_INCOMPLETE_SEARCH,
    arm_spec,
)
from src.capability.mutation import designed_referent as D  # noqa: E402
from src.capability.models import scale_comparison as C  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.models.scale_comparison import (  # noqa: E402
    compound_verdict,
    lower_bound_positive,
    # Re-exported under the stage's own spelling. Nothing here calls it; the
    # stage's tests do, to state the property the design census rests on -- that
    # the unrestricted design side of a stage-29 payload is 146 wild types, so
    # the 130 certified zero-hit designs cannot be read off that file and must
    # come from the cohort.
    side_keys as _side_keys,  # noqa: F401
)

SCHEMA_VERSION = "r2_transfer_scale_capability_v2"
DEFAULT_OUT = REPO / "results/R1/progen2_scale_20260826"
SCALE_RUNGS = ("progen2-medium", "progen2-large", "progen2-xlarge")
ADJACENT_PAIRS = (
    ("progen2-medium", "progen2-large"),
    ("progen2-large", "progen2-xlarge"),
)
BOOTSTRAP_RESAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260825
HYDROPATHY_BASELINE = "hydropathy_change"
FRAGMENT_MAX_ORDER = 7

#: Fixed EXP-R2-224 census, from the stage-20/29 artefacts this comparison reads.
#:
#: The DMS cohort has two counts and they are not the same number. ProteinGym's
#: substitution queue is 217 assays over 174 wild-type families and is what the
#: freeze declares. Every rung of this ladder is a 1024-position ProGen2 with one
#: shared residue tokenizer, so the 16 assays whose rendered variant exceeds that
#: context cannot be scored on any of them; the ladder's analysis set is the 201
#: assays over 163 families that remain. Both are recorded, and the exclusion is
#: a property of the shared interface rather than of a checkpoint or of a score.
DMS_DECLARED_ASSAYS = 217
DMS_DECLARED_CLUSTERS = 174
DMS_ANALYSIS_ASSAYS = 201
DMS_ANALYSIS_CLUSTERS = 163
PROGEN2_CONTEXT = 1024
DMS_CONTEXT_EXCLUSION_REASON = "exceeds this arm's context"
DMS_CONTEXT_EXCLUDED_ASSAYS = (
    "A0A140D2T1_ZIKV_Sourisseau_2019",
    "BRCA1_HUMAN_Findlay_2018",
    "BRCA2_HUMAN_Erwood_2022_HEK293T",
    "CAR11_HUMAN_Meitlis_2020_gof",
    "CAR11_HUMAN_Meitlis_2020_lof",
    "CAS9_STRP1_Spencer_2017_positive",
    "ERBB2_HUMAN_Elazar_2016",
    "KCNH2_HUMAN_Kozek_2020",
    "NPC1_HUMAN_Erwood_2022_HEK293T",
    "NPC1_HUMAN_Erwood_2022_RPE1",
    "POLG_CXB3N_Mattenberger_2021",
    "POLG_HCVJF_Qi_2014",
    "SCN5A_HUMAN_Glazer_2019",
    "SPIKE_SARS2_Starr_2020_binding",
    "SPIKE_SARS2_Starr_2020_expression",
    "UBE4B_MOUSE_Starita_2013",
)
MEGASCALE_DESIGN_WILDTYPES = 130
MEGASCALE_DESIGN_SERIES = 40
MEGASCALE_NATURAL_WILDTYPES = 266
MEGASCALE_NATURAL_CLUSTERS = 124

#: The same two censuses as objects, which is the form
#: :mod:`src.capability.models.scale_comparison` checks them in. The numbers above stay
#: the declaration; these bind them to this campaign's label so a refusal names
#: the freeze it came from, and are the only route by which the library learns a
#: count at all.
DMS_CENSUS = C.DmsCensus(
    label="EXP-R2-224",
    declared_assays=DMS_DECLARED_ASSAYS,
    declared_clusters=DMS_DECLARED_CLUSTERS,
    analysis_assays=DMS_ANALYSIS_ASSAYS,
    analysis_clusters=DMS_ANALYSIS_CLUSTERS,
    context_excluded_assays=DMS_CONTEXT_EXCLUDED_ASSAYS,
)
MEGASCALE_CENSUS = C.MegascaleCensus(
    design_wildtypes=MEGASCALE_DESIGN_WILDTYPES,
    design_series=MEGASCALE_DESIGN_SERIES,
    natural_wildtypes=MEGASCALE_NATURAL_WILDTYPES,
    natural_clusters=MEGASCALE_NATURAL_CLUSTERS,
)

DESCRIPTIVE_NOT_CAUSAL = (
    "checkpoint differences on this ladder are descriptive of the named "
    "checkpoints. They are not identified as a causal effect of parameter count"
)
NO_BIOLOGICAL_KNOWLEDGE_CLAIM = (
    "no biological-knowledge claim is licensed. Beating a sequence baseline is "
    "not evidence that a model has learned biology"
)
DMS_CONTEXT_EXCLUSION_NOTE = (
    "these assays render longer than the 1024-position context every rung of "
    "this ladder shares, so no rung can score them and truncating would score a "
    "sequence that may not contain the mutated position. The exclusion is a "
    "cohort definition fixed by the interface, not a filter on any score; the "
    "ladder's DMS reading covers the 201 assays and 163 families that remain, "
    "not all of ProteinGym's 217 and 174"
)
ZERO_HIT_DESIGN_NOTE = (
    "the design side is the certified zero-hit design census. "
    "baselines.json and model_<arm>.json score all 146 designs and carry no "
    "zero_hit flag, so the flag is read from the stage-29 cohort itself, whose "
    "SHA-256 is the cohort_sha256 those payloads already declare"
)
FRAGMENT_INCOMPLETE_NOTE = (
    "the stage-29 fragment_order pass carries margins only for the rungs in "
    "rungs_present. 3-7-mer margins are reported and never gated, so a rung "
    "missing from that pass does not stop a gate; it is named here rather than "
    "silently omitted"
)
TRANSITION_MEANS = (
    "the larger rung clears its own baseline contrast and is paired-"
    "significantly better than the smaller; this is not a claim that the "
    "smaller rung failed"
)
PRECISION_NOTE = (
    "one precision per endpoint across all three rungs. A rung scored at a "
    "different precision from the rung it is paired against would make the "
    "paired difference partly a difference of arithmetic"
)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_rung_order(names: list[str]) -> None:
    """Refuse any comparison whose rung list is not medium, large, xlarge."""

    C.require_rungs(names, SCALE_RUNGS)


def require_frozen_bootstrap(resamples: int, seed: int) -> None:
    """Refuse CLI changes to the pre-data EXP-R2-224 bootstrap freeze."""

    if resamples != BOOTSTRAP_RESAMPLES or seed != DEFAULT_BOOTSTRAP_SEED:
        raise ValueError(
            "EXP-R2-224 freezes stage 42 at "
            f"resamples={BOOTSTRAP_RESAMPLES}, seed={DEFAULT_BOOTSTRAP_SEED}; "
            f"got resamples={resamples}, seed={seed}"
        )


def _endpoint(
    per_rung: dict[str, dict[str, float]],
    units: dict[str, str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """One endpoint over this ladder's frozen rungs and adjacent pairs."""

    return C.endpoint(
        per_rung,
        units,
        rungs=SCALE_RUNGS,
        pairs=ADJACENT_PAIRS,
        resamples=resamples,
        seed=seed,
    )


def align_dms(
    models: dict[str, dict[str, Any]],
    lookup: dict[str, Any],
    *,
    require_fixed_census: bool = True,
) -> dict[str, Any]:
    """Align the DMS cohort against this campaign's frozen interface and census.

    ``require_fixed_census=False`` drops the four count checks and the frozen
    exclusion set; every structural refusal -- a rung-specific skip, a skip that
    is not a shared-context overflow, a skip against another context, an assay
    set that disagrees with the jointly scored LOOKUP subset -- still applies,
    because those are properties of the pairing rather than of the census.
    """

    return C.align_dms(
        models,
        lookup,
        rungs=SCALE_RUNGS,
        context=PROGEN2_CONTEXT,
        exclusion_reason=DMS_CONTEXT_EXCLUSION_REASON,
        census=DMS_CENSUS if require_fixed_census else None,
    )


def zero_hit_design_cohort(path: Path) -> dict[str, Any]:
    """The certified zero-hit design census, read from the stage-29 cohort that
    carries the flag.

    ``zero_hit`` lives on the cohort's wild types and on neither ``baselines``
    nor ``model_<arm>``, both of which score all 146 designs. Reading it here
    reaches the same 130 certified zero-hit designs through the same
    :meth:`Referent.side` rule the rest of stage 29 uses, and the file's SHA-256
    is the ``cohort_sha256`` those payloads already declare, so the input cannot
    drift without the digest check firing.
    """

    referent = D.load_referent(path)
    names = sorted(wildtype.name for wildtype in referent.side("design"))
    if not names:
        raise ValueError(f"{path} carries no certified zero-hit design")
    return {
        "source": path.name,
        "cohort_sha256": sha256_file(path),
        "zero_hit_designs": names,
        "note": ZERO_HIT_DESIGN_NOTE,
    }


def align_megascale(
    models: dict[str, dict[str, Any]],
    baselines: dict[str, Any],
    *,
    side: str,
    baseline_name: str,
    admissible: frozenset[str] | None = None,
    require_fixed_census: bool = True,
) -> tuple[dict[str, str], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Align one MegaScale side against this campaign's frozen rungs and census."""

    return C.align_megascale(
        models,
        baselines,
        rungs=SCALE_RUNGS,
        side=side,
        baseline_name=baseline_name,
        admissible=admissible,
        census=MEGASCALE_CENSUS if require_fixed_census else None,
    )


def require_uniform_dtype(models: dict[str, dict[str, Any]], *, label: str) -> str:
    """One scoring precision across this campaign's three rungs."""

    return C.require_uniform_dtype(models, rungs=SCALE_RUNGS, label=label)


def _require_same_cohort(*payloads: dict[str, Any], label: str) -> str:
    digests = [payload["cohort_sha256"] for payload in payloads]
    if len(set(digests)) != 1:
        raise ValueError(f"{label} cohort_sha256 disagree: {digests}")
    return digests[0]


def _fragment_margins(fragment_order: dict[str, Any]) -> dict[str, Any]:
    """Report every supported fragment margin up to k=7, without a composite score."""

    admissibility = fragment_order.get("admissibility") or {}
    highest = admissibility.get("highest_supported_order")
    supported = []
    if highest is not None:
        supported = [order for order in range(3, min(int(highest), FRAGMENT_MAX_ORDER) + 1)]
    settings = fragment_order.get("settings") or {}
    schemes = list(settings.get("schemes") or D.FRAGMENT_SMOOTHING)
    arms = fragment_order.get("arms") or {}
    present = [name for name in SCALE_RUNGS if name in arms]
    missing = [name for name in SCALE_RUNGS if name not in arms]
    margins: dict[str, Any] = {"designs": {}, "control": {}}
    for side in ("designs", "control"):
        for order in supported:
            for scheme in schemes:
                key = D.fragment_channel_name(order, scheme)
                per_rung = {}
                for rung in present:
                    block = arms[rung].get(side) or {}
                    if key not in block:
                        raise ValueError(
                            f"fragment_order {rung} {side} lacks {key} though "
                            f"k={order} is supported"
                        )
                    per_rung[rung] = block[key]
                if per_rung:
                    margins[side][key] = {"per_rung": per_rung, "supported_to_k": order}
    return {
        "highest_supported_order": highest,
        "supported_orders": supported,
        "schemes": schemes,
        "margins": margins,
        "reported_not_gated": True,
        "rungs_present": present,
        "rungs_missing": missing,
        "incomplete_note": FRAGMENT_INCOMPLETE_NOTE if missing else None,
        "cohort_sha256": fragment_order.get("cohort_sha256"),
    }


def qualify_stage41(report: dict[str, Any]) -> dict[str, Any]:
    """Qualify this campaign's three rungs from a full stage-41 report."""

    return C.qualify_stage41(report, rungs=SCALE_RUNGS)


def descriptive_gate_transitions(
    dms: dict[str, Any], megascale: dict[str, Any]
) -> dict[str, Any]:
    """The one pre-registered gate family. No per-endpoint copies.

    A transition asks the LARGER rung to clear its own baseline contrast and the
    paired delta to exclude zero. It never asks the smaller rung to fail: a rule
    that required the smaller rung's own contrast to be non-significant would
    make the gate depend on accepting a null, so a well-powered pair in which
    both rungs are competent could not register a transition however large the
    improvement. The smaller rung's contrasts are therefore reported beside each
    gate rather than folded into it.
    """

    dms_gates: dict[str, Any] = {}
    mega_gates: dict[str, Any] = {}
    for smaller, larger in ADJACENT_PAIRS:
        pair = f"{smaller}__{larger}"
        dms_conditions = {
            "larger_model_minus_lookup": lower_bound_positive(
                dms["model_minus_lookup"]["per_rung"][larger]
            ),
            "raw_spearman_delta": lower_bound_positive(
                dms["raw_spearman"]["adjacent_delta_rho"][pair]
            ),
        }
        dms_gates[pair] = {
            "verdict": compound_verdict(dms_conditions),
            "conditions": dms_conditions,
            "blosum62_is_not_a_dms_gate": True,
            "transition_means": TRANSITION_MEANS,
            "reported_not_gated": {
                "smaller_model_minus_lookup": lower_bound_positive(
                    dms["model_minus_lookup"]["per_rung"][smaller]
                )
            },
        }
        mega_conditions = {
            "design_larger_model_minus_hydropathy": lower_bound_positive(
                megascale["designs"]["model_minus_hydropathy"]["per_rung"][larger]
            ),
            "design_larger_model_minus_blosum62": lower_bound_positive(
                megascale["designs"]["model_minus_blosum62"]["per_rung"][larger]
            ),
            "natural_larger_model_minus_hydropathy": lower_bound_positive(
                megascale["control"]["model_minus_hydropathy"]["per_rung"][larger]
            ),
            "natural_larger_model_minus_blosum62": lower_bound_positive(
                megascale["control"]["model_minus_blosum62"]["per_rung"][larger]
            ),
            "design_raw_spearman_delta": lower_bound_positive(
                megascale["designs"]["raw_spearman"]["adjacent_delta_rho"][pair]
            ),
        }
        mega_gates[pair] = {
            "verdict": compound_verdict(mega_conditions),
            "conditions": mega_conditions,
            "transition_means": TRANSITION_MEANS,
            "reported_not_gated": {
                "natural_raw_spearman_delta": lower_bound_positive(
                    megascale["control"]["raw_spearman"]["adjacent_delta_rho"][pair]
                ),
                "smaller_design_model_minus_hydropathy": lower_bound_positive(
                    megascale["designs"]["model_minus_hydropathy"]["per_rung"][smaller]
                ),
                "smaller_design_model_minus_blosum62": lower_bound_positive(
                    megascale["designs"]["model_minus_blosum62"]["per_rung"][smaller]
                ),
            },
        }
    return {"dms": dms_gates, "megascale": mega_gates}


def compare_scale(
    *,
    dms_models: dict[str, dict[str, Any]],
    lookup: dict[str, Any],
    megascale_models: dict[str, dict[str, Any]],
    baselines: dict[str, Any],
    design_cohort: dict[str, Any],
    fragment_order: dict[str, Any] | None,
    qualification_report: dict[str, Any],
    resamples: int,
    seed: int,
    require_fixed_census: bool = True,
) -> dict[str, Any]:
    qualification = qualify_stage41(qualification_report)
    require_rung_order(list(dms_models))
    require_rung_order(list(megascale_models))
    dms_dtype = require_uniform_dtype(dms_models, label="DMS")
    mega_dtype = require_uniform_dtype(megascale_models, label="MegaScale")
    alignment = align_dms(
        dms_models, lookup, require_fixed_census=require_fixed_census
    )
    dms_units = alignment["units"]
    dms_raw = alignment["raw"]
    dms_contrasts = alignment["contrasts"]
    dms = {
        "declared_cohort": {
            "n_assays": len(alignment["declared_assays"]),
            "n_clusters": alignment["declared_clusters"],
            "source": "ProteinGym substitution assays, wild-type families at 50% identity",
        },
        "context_excluded_assays": alignment["context_excluded_assays"],
        "context_exclusion_note": DMS_CONTEXT_EXCLUSION_NOTE,
        "n_assays": len(alignment["analysis_assays"]),
        "n_clusters": len(set(dms_units.values())),
        "unit": "wild-type family at 50% identity",
        "raw_spearman": _endpoint(dms_raw, dms_units, resamples=resamples, seed=seed),
        "model_minus_lookup": _endpoint(
            dms_contrasts["model_minus_lookup"], dms_units, resamples=resamples, seed=seed + 20
        ),
        "model_minus_blosum62": _endpoint(
            dms_contrasts["model_minus_blosum62"], dms_units, resamples=resamples, seed=seed + 40
        ),
    }

    mega_digest = _require_same_cohort(
        *megascale_models.values(), baselines, label="MegaScale model/baseline"
    )
    if design_cohort["cohort_sha256"] != mega_digest:
        raise ValueError(
            "the zero-hit design census was read from a different cohort: "
            f"{design_cohort['cohort_sha256']} against {mega_digest}"
        )
    design_names = frozenset(design_cohort["zero_hit_designs"])
    if fragment_order is not None:
        if "cohort_sha256" not in fragment_order:
            raise ValueError("fragment_order exists but carries no cohort_sha256")
        _require_same_cohort(
            megascale_models[SCALE_RUNGS[0]],
            fragment_order,
            label="MegaScale model/fragment_order",
        )
    megascale: dict[str, Any] = {
        "cohort_sha256": mega_digest,
        "hydropathy_baseline": HYDROPATHY_BASELINE,
        "design_census": {
            "source": design_cohort["source"],
            "n_zero_hit_designs": len(design_names),
            "note": design_cohort["note"],
        },
    }
    for side, label, unit_name, offset in (
        ("design", "designs", "design series", 60),
        ("natural", "control", "WT_cluster", 80),
    ):
        admissible = design_names if side == "design" else None
        units, raw, hydro = align_megascale(
            megascale_models,
            baselines,
            side=side,
            baseline_name=HYDROPATHY_BASELINE,
            admissible=admissible,
            require_fixed_census=require_fixed_census,
        )
        _, _, blosum = align_megascale(
            megascale_models,
            baselines,
            side=side,
            baseline_name="blosum62",
            admissible=admissible,
            require_fixed_census=require_fixed_census,
        )
        megascale[label] = {
            "unit": unit_name,
            "n_wildtypes": len(units),
            "raw_spearman": _endpoint(raw, units, resamples=resamples, seed=seed + offset),
            "model_minus_hydropathy": _endpoint(
                hydro, units, resamples=resamples, seed=seed + offset + 5
            ),
            "model_minus_blosum62": _endpoint(
                blosum, units, resamples=resamples, seed=seed + offset + 10
            ),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "scale_capability_report",
        "created_utc": _timestamp(),
        "rungs": list(SCALE_RUNGS),
        "not_panel_admission": True,
        "staged_scale": {
            "scope": "progen2_training_lineage_medium_to_xlarge",
            "measured_staged_arms": list(STAGED_SCALE_ARMS),
            "scoring_target_alphabet": {
                name: {
                    "size": arm_spec(name).scoring_target_alphabet_size,
                    "source": "arm_spec.scoring_target_alphabet_size",
                }
                for name in STAGED_SCALE_ARMS
            },
        },
        "descriptive_not_causal": True,
        "descriptive_not_causal_note": DESCRIPTIVE_NOT_CAUSAL,
        "corpus_identification_bound": UNIREF90_BFD30_INCOMPLETE_SEARCH,
        "no_biological_knowledge_claim": True,
        "no_biological_knowledge_claim_note": NO_BIOLOGICAL_KNOWLEDGE_CLAIM,
        "no_cross_task_total": True,
        "bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "default_seed": DEFAULT_BOOTSTRAP_SEED,
        },
        "precision": {
            "dms": dms_dtype,
            "megascale": mega_dtype,
            "note": PRECISION_NOTE,
        },
        "qualification": qualification,
        "dms": dms,
        "megascale": megascale,
        "descriptive_gate_transitions": descriptive_gate_transitions(dms, megascale),
        "fragment_order": None if fragment_order is None else _fragment_margins(fragment_order),
    }
    return payload


def _load_rung_models(directory: Path, prefix: str) -> dict[str, dict[str, Any]]:
    models = {}
    for name in SCALE_RUNGS:
        path = directory / f"{prefix}{name}.json"
        payload = _read(path)
        if payload.get("arm") != name:
            raise ValueError(f"{path} declares arm {payload.get('arm')!r}, expected {name}")
        models[name] = payload
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval-bound-dir",
        type=Path,
        required=True,
        help="directory holding model_<arm>.json and lookup.json from stage 20",
    )
    parser.add_argument(
        "--designed-referent-dir",
        type=Path,
        required=True,
        help=(
            "directory holding model_<arm>.json, baselines.json and cohort.json "
            "from stage 29"
        ),
    )
    parser.add_argument(
        "--context-information-summary",
        type=Path,
        required=True,
        help="full stage-41 report with arm_results; summary-only JSON is refused",
    )
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    require_frozen_bootstrap(args.bootstrap, args.seed)

    qualification_report = _read(args.context_information_summary)
    qualify_stage41(qualification_report)

    dms_models = _load_rung_models(args.retrieval_bound_dir, "model_")
    lookup = _read(args.retrieval_bound_dir / "lookup.json")
    megascale_models = _load_rung_models(args.designed_referent_dir, "model_")
    baselines = _read(args.designed_referent_dir / "baselines.json")
    design_cohort = zero_hit_design_cohort(args.designed_referent_dir / "cohort.json")
    fragment_path = args.designed_referent_dir / "fragment_order.json"
    fragment_order = _read(fragment_path) if fragment_path.is_file() else None

    payload = compare_scale(
        dms_models=dms_models,
        lookup=lookup,
        megascale_models=megascale_models,
        baselines=baselines,
        design_cohort=design_cohort,
        fragment_order=fragment_order,
        qualification_report=qualification_report,
        resamples=args.bootstrap,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "scale_capability.json"
    write_json(destination, payload)
    print(f"wrote {destination}")
    gates = payload["descriptive_gate_transitions"]
    for pair, block in gates["dms"].items():
        print(f"DMS gate {pair}: {block['verdict']}")
    for pair, block in gates["megascale"].items():
        print(f"MegaScale gate {pair}: {block['verdict']}")


if __name__ == "__main__":
    main()
