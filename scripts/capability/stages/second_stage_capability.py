#!/usr/bin/env python3
"""EXP-R2-225's descriptive second-stage comparison. Not a causal scale claim.

CPU only. It consumes existing stage-20 and stage-41 artefacts, loads no model,
downloads no weights, and synthesises no cross-task total.

Stage 42's sibling, and a sibling for the same reason
``second_stage_interface_qualification.py`` is one: the *freeze* belongs to the
campaign that made it. Every rung name, wave, census and stratum below is
EXP-R2-225's own declaration and is checked here. The arithmetic underneath --
the paired resampling, the cohort alignment, the stage-41 qualification, the
uniform-precision and uniform-stratum checks -- is
:mod:`src.capability.models.scale_comparison`, one implementation shared with EXP-R2-224,
because the two campaigns must not be free to drift apart on exactly the
operations that make them commensurable. The group bootstrap is not restated
either: ``resamples`` and ``seed`` are imported from ``scale_capability.py``,
where the pre-data amendment that freezes them for both experiments put them.

**Three waves, one label rule.** ``descriptive_gate_transition`` is reserved for
**same-family adjacent rungs** and exists only as the unique compound block of
the EXP-R2-224 pre-data amendment. A cross-family row cannot receive it, and
neither can ProteinGLM or RITA, which report existence only. Auxiliary intervals
-- a per-rung raw Spearman, a MODEL - BLOSUM62, a context-information curve --
are reported and are not gates. :func:`require_label_eligible` is that rule as a
refusal rather than as a sentence.

**What this stage does not build, and why that is the point.** Two endpoints
EXP-R2-225's text licenses have no code path here, and each is declared as
unreachable with its reason rather than given a gate no artefact can feed:

* **MegaScale, on every wave.** EXP-R2-225's freeze delivers no MegaScale row
  on any wave, so this stage builds no gate for one. The code-path obstruction
  this bullet used to describe is gone: ``designed_referent.py`` now admits
  both ProGen3 rungs -- ``arms.PROGEN3_ARMS`` is a door there and
  ``designed_referent.EXCLUDED_ARMS`` no longer names the lineage -- and
  EXP-R2-244 scores them on the single N-to-C direction by decision, which is
  what the exclusion's ground was about. The ground stands and is satisfied
  rather than overridden: that lineage may not be read in the same reading as the
  N-to-C arms, and on this queue its own reading is the N-to-C one, recorded
  under :data:`scale_comparison.STRATUM_N_TO_C`. Its published bidirectional
  convention is not called wrong anywhere -- it is what stage 20 scores and what
  the lineage's ProteinGym reading is on -- and the two readings are separate
  functionals that are never pooled, ranked together or converted into one
  another.
* **ProteinGym and MegaScale on Galactica protein mode.** No code path reaches
  either: Galactica is declared as an arm nowhere and is reachable only through
  ``joint_mode_qualification.py --checkpoint <dir> --rendering galactica``,
  which measures context information rather than fitness. Stages 20 and 29 both
  resolve an ``ArmSpec`` by name and would refuse it.

A wave therefore carries only the endpoints it can actually be fed, and
:data:`NOT_DELIVERABLE` says so in the artefact.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts/capability") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts/capability"))

from src.capability.models import scale_comparison as C  # noqa: E402
from src.capability.core.arms import REPO  # noqa: E402
from src.capability.core.io import write_json  # noqa: E402

#: EXP-R2-224's stage 42, imported for the two numbers its pre-data amendment
#: freezes for **both** experiments: "This stage's new paired group bootstrap
#: inherits the amendment: resamples=2000, seed=20260825." Restating them here
#: would be a second place one frozen number could live and a second place it
#: could be edited. ``importlib`` rather than an ``import`` statement only
#: because a module whose name begins with a digit has no statement spelling.
STAGE_42 = importlib.import_module('scripts.capability.stages.scale_capability')
BOOTSTRAP_RESAMPLES = STAGE_42.BOOTSTRAP_RESAMPLES
DEFAULT_BOOTSTRAP_SEED = STAGE_42.DEFAULT_BOOTSTRAP_SEED
TRANSITION_MEANS = STAGE_42.TRANSITION_MEANS

SCHEMA_VERSION = "r2_transfer_second_stage_capability_v1"
DEFAULT_OUT = REPO / "results/R1/second_stage_capability_20260826"
REPORT_NAME = "second_stage_capability.json"

DESCRIPTIVE_NOT_CAUSAL = (
    "checkpoint differences on these ladders are descriptive of the named "
    "checkpoints. They are not identified as a causal effect of parameter count: "
    "depth, width and parameter count move together, and corpus and training "
    "remain confounded with the rung"
)
NO_BIOLOGICAL_KNOWLEDGE_CLAIM = (
    "no biological-knowledge claim is licensed. Beating a sequence baseline is "
    "not evidence that a model has learned biology"
)
LOOKUP_IS_NOT_A_RETRIEVAL_BOUND = (
    "LOOKUP is stage 20's staged UniRef50 profile on the frozen ProteinGym "
    "profiles. It is a retrieval bound only on an arm with evidenced identity or "
    "containment between that snapshot and its training set; on every arm of this "
    "campaign it is an external UniRef50 profile baseline, so MODEL - LOOKUP is a "
    "capability comparison against that channel and does not exclude, bound or "
    "quantify training-corpus retrieval"
)
CONTEXT_INFORMATION_IS_NOT_A_GATE = (
    "context information is a qualification endpoint. It is not averaged with a "
    "fitness gate, the between-rung separation in this table is not a tested "
    "contrast, and a movement in it is a descriptive curve and never a "
    "descriptive_gate_transition"
)
EXISTENCE_ONLY = (
    "a Wave B single point reports existence or unavailability. It is not "
    "evidence of a large-scale gate flip and is not a substitute for any failed "
    "Wave A arm"
)

#: Every endpoint this stage could name, so that a wave which does not carry one
#: has to say why rather than simply not mention it.
ENDPOINTS = ("dms", "megascale", "context_information")

MEGASCALE_PROGEN3_REFUSAL = (
    "EXP-R2-225's freeze delivers no MegaScale row on any wave, and that is now "
    "the whole of the reason: the code-path obstruction this string used to "
    "describe is gone. designed_referent.py admits both ProGen3 rungs -- "
    "arms.PROGEN3_ARMS is a door there and designed_referent.EXCLUDED_ARMS no "
    "longer names the lineage -- and EXP-R2-244 scores them on the single N-to-C "
    "direction by decision, which is exactly what the exclusion's ground was "
    "about: ProGen3 may not be read in the same reading as the N-to-C arms, and "
    "on that queue its own reading now IS the N-to-C one, recorded under "
    "scale_comparison.STRATUM_N_TO_C. The deviation from the lineage's published "
    "bidirectional convention is declared in the payload rather than silently "
    "resolved, and the two ProGen3 readings -- that N-to-C designed-referent one "
    "and stage 20's bidirectional ProteinGym one -- are separate functionals "
    "that are never pooled, ranked together or converted into one another. What "
    "still stops a row here is the freeze and not the code: no wave of this "
    "campaign carries the endpoint, so this stage builds no gate no artefact of "
    "this campaign can feed. A gate for the stage-29 reading belongs to the "
    "campaign that produces it"
)
GALACTICA_FITNESS_REFUSAL = (
    "EXP-R2-225's freeze delivers no ProteinGym or MegaScale row on the "
    "Galactica wave, and that is now the whole of the reason: the code-path "
    "obstruction this string used to describe is gone. retrieval_bound.py "
    "scores the four rungs through galactica_fitness.GalacticaFitnessScorer, "
    "and designed_referent.py opens the same scorer on MegaScale without "
    "an ArmSpec. What still stops a row here is the freeze and not the code: "
    "no wave of this campaign carries either fitness endpoint, so this stage "
    "builds no gate no artefact of this campaign can feed"
)
QWEN_FITNESS_FORBIDDEN = (
    "EXP-R2-225 forbids it: 'Qwen does not enter them.' A text checkpoint gets "
    "no fabricated protein task, ProteinGym row or MegaScale row, and no text "
    "analogue of either is invented in order to force a transition"
)
PROGEN3_CONTEXT_INFORMATION_UNAVAILABLE = (
    "neither ProGen3 rung has a stage-41 identification record and neither can "
    "have one as staged: ProGen3 has no ArmSpec -- it is reached through "
    "src.capability.models.progen3.load_progen3, which converts the released megablocks "
    "packing -- so cohort_power.py cannot resolve it and writes no sufficient-"
    "statistics sidecar for stage 41 to read. The wave's qualification is the "
    "per-checkpoint ProGen3 self-check instead"
)

#: What EXP-R2-225's text licenses that this repository cannot produce, keyed by
#: ``endpoint/family`` with the refusal that stops it and where that refusal
#: lives. Carried into the artefact so a reader learns what was not run from the
#: report rather than from its absence.
NOT_DELIVERABLE: dict[str, str] = {
    "megascale/progen3": MEGASCALE_PROGEN3_REFUSAL,
    "context_information/progen3": PROGEN3_CONTEXT_INFORMATION_UNAVAILABLE,
    "dms/galactica": GALACTICA_FITNESS_REFUSAL,
    "megascale/galactica": GALACTICA_FITNESS_REFUSAL,
    "dms/qwen2.5": QWEN_FITNESS_FORBIDDEN,
    "megascale/qwen2.5": QWEN_FITNESS_FORBIDDEN,
}


# ------------------------------------------------------------------- the waves


@dataclass(frozen=True)
class Wave:
    """One declared trajectory: its rungs, its adjacent pairs, its endpoints.

    ``family`` is what the label rule turns on. ``descriptive_gate_transition``
    is reserved for same-family adjacent rungs, so a wave that spans families --
    or a single point that is no wave at all -- is refused the label by
    :func:`require_label_eligible` rather than by a reviewer noticing.

    ``endpoints`` is what this wave can be *fed*, not what would be interesting
    to compute. An endpoint absent from it has an entry in
    :data:`NOT_DELIVERABLE` naming the refusal that stops it.
    """

    name: str
    family: str
    rungs: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    endpoints: tuple[str, ...]
    qualification: str
    note: str

    def __post_init__(self) -> None:
        if len(self.rungs) < 2:
            raise ValueError(f"{self.name}: a wave needs at least two rungs")
        if len(set(self.rungs)) != len(self.rungs):
            raise ValueError(f"{self.name}: a rung is repeated")
        expected = tuple(zip(self.rungs, self.rungs[1:]))
        if self.pairs != expected:
            raise ValueError(
                f"{self.name}: pairs must be the adjacent rungs in order, "
                f"{list(expected)}; got {list(self.pairs)}"
            )


#: EXP-R2-225's Wave A, in the prereg's own order. Adjacent pairs only.
#:
#: The joint wave's rungs are checkpoint names and not ``ArmSpec`` names -- there
#: is no ``ArmSpec`` for a Galactica rung and there must not be -- so they name
#: the directory each stage-41 report was produced from. The pure-text wave's
#: rungs are arm names, ``qwen2.5-0.5b`` being a panel member scored on this
#: track without inheriting anything from its membership. The pure-protein wave's
#: rungs are the two ProGen3 names ``retrieval_bound.py`` scores.
WAVES: tuple[Wave, ...] = (
    Wave(
        name="joint_galactica",
        family="galactica",
        rungs=("galactica-1.3b", "galactica-6.7b", "galactica-30b"),
        pairs=(
            ("galactica-1.3b", "galactica-6.7b"),
            ("galactica-6.7b", "galactica-30b"),
        ),
        endpoints=("context_information",),
        qualification="stage41_per_rung",
        note="facebook/galactica-125m is not a rung here. Each mode qualifies "
        "separately through joint_mode_qualification.py, whose report names "
        "the condition rather than the checkpoint, so the three rungs are three "
        "reports and not three rows of one",
    ),
    Wave(
        name="text_qwen",
        family="qwen2.5",
        rungs=("qwen2.5-0.5b", "qwen2.5-7b", "qwen2.5-32b"),
        pairs=(("qwen2.5-0.5b", "qwen2.5-7b"), ("qwen2.5-7b", "qwen2.5-32b")),
        endpoints=("context_information",),
        qualification="stage41_shared_report",
        note="base checkpoints only; an instruct, chat or thinking substitute is "
        "forbidden. Text continuation only: Qwen does not enter ProteinGym or "
        "MegaScale and no text analogue of either is invented in order to force "
        "a descriptive_gate_transition",
    ),
    Wave(
        name="protein_progen3",
        family="progen3",
        rungs=("progen3-112m", "progen3-3b"),
        pairs=(("progen3-112m", "progen3-3b"),),
        endpoints=("dms",),
        qualification="stage20_progen3_self_check",
        note="the lineage's official bidirectional estimand, on both rungs. Its "
        "qualification is src.capability.models.progen3.self_check, a per-checkpoint band "
        "with three recorded corruptions, which retrieval_bound.py runs before "
        "it scores and records in the payload this stage reads. Neither rung has "
        "a stage-41 identification record, because ProGen3 has no ArmSpec and "
        "cohort_power.py cannot reach it",
    ),
)

WAVES_BY_NAME = {wave.name: wave for wave in WAVES}

#: Wave B, which is not a wave: two new-lineage single points that report
#: existence only and can never receive the reserved label.
SINGLE_POINTS: dict[str, str] = {
    "proteinglm-7b-clm": "one descriptive pure-protein decoder point",
    "rita-xl": "a secondary new-architecture single point",
}

#: The scoring stratum each wave with a fitness endpoint is read under, checked
#: against every payload rather than assumed from the arm name.
WAVE_STRATUM: dict[str, str] = {
    "protein_progen3": C.STRATUM_BIDIRECTIONAL,
}

#: The ProGen3 wave's DMS census.
#:
#: **Two counts that coincide, which is a fact about this ladder and not a rule.**
#: EXP-R2-224's ProGen2 ladder analyses 201 of ProteinGym's 217 assays because 16
#: render longer than the 1024-position context all three of its rungs share.
#: ProGen3 declares 65536 positions and ``retrieval_bound.py`` records no
#: context for it at all, so no assay is unscorable for that reason and the
#: analysis set is the declared cohort. The exclusion machinery still runs: a
#: skip taken for any other reason, or by one rung and not the other, is a
#: refusal exactly as it is on the first round.
PROGEN3_DMS_CENSUS = C.DmsCensus(
    label="EXP-R2-225 progen3",
    declared_assays=217,
    declared_clusters=174,
    analysis_assays=217,
    analysis_clusters=174,
    context_excluded_assays=(),
)

#: The position budget ProGen3 declares, and the reason a skip would have to
#: name. Nothing on this ladder is expected to skip; both are passed so that one
#: that does is refused with the same specificity the first round's is.
PROGEN3_CONTEXT = 65536
DMS_CONTEXT_EXCLUSION_REASON = "exceeds this arm's context"


# -------------------------------------------------------------- the label rule


def require_wave(name: str) -> Wave:
    if name not in WAVES_BY_NAME:
        raise ValueError(
            f"{name!r} is not an EXP-R2-225 wave; the waves are "
            f"{list(WAVES_BY_NAME)}"
        )
    return WAVES_BY_NAME[name]


def require_label_eligible(wave: Wave, pair: tuple[str, str]) -> None:
    """``descriptive_gate_transition`` is same-family adjacent rungs, or nothing.

    The rule verbatim from EXP-R2-225: the label "is reserved for **same-family
    adjacent rungs** and exists only as the unique compound block in the
    EXP-R2-224 pre-data amendment. Cross-family rows, ProteinGLM, and RITA cannot
    receive that label. Auxiliary intervals are not gates."

    A refusal rather than a comment because a label is a string, and a string is
    exactly the kind of thing that gets attached to the wrong row by a later
    edit. The two single points cannot reach this function at all -- they are in
    no wave -- and are refused by name as well, so the rule holds even if one is
    ever declared as a one-rung wave.
    """

    smaller, larger = pair
    forbidden = sorted(set(pair) & set(SINGLE_POINTS))
    if forbidden:
        raise ValueError(
            f"{forbidden} report existence only and can never receive "
            "descriptive_gate_transition"
        )
    # Family before adjacency, because a rung from another wave is the
    # cross-family case and must be named as one rather than as a pair that
    # happens not to be adjacent.
    for name in pair:
        _family_of(wave, name)
    if pair not in wave.pairs:
        raise ValueError(
            f"{smaller} -> {larger} is not an adjacent pair of wave "
            f"{wave.name!r}; the label is reserved for adjacent rungs and "
            f"{wave.name} declares {[list(p) for p in wave.pairs]}"
        )


def _family_of(wave: Wave, rung: str) -> str:
    """The family a rung belongs to, which is the wave's or nothing.

    A wave is one family by construction, so a rung this wave does not declare
    belongs to another one -- that is the cross-family case, and it is refused by
    name here rather than reported as an adjacency failure.
    """

    if rung not in wave.rungs:
        raise ValueError(
            f"{rung!r} is not a rung of wave {wave.name!r} (family "
            f"{wave.family!r}); descriptive_gate_transition is reserved for "
            "same-family adjacent rungs and a cross-family row cannot receive it"
        )
    return wave.family


def require_endpoint(wave: Wave, endpoint: str) -> None:
    if endpoint in wave.endpoints:
        return
    if endpoint not in ENDPOINTS:
        raise ValueError(f"{endpoint!r} is not one of {list(ENDPOINTS)}")
    raise ValueError(
        f"wave {wave.name!r} carries no {endpoint!r} endpoint. "
        + NOT_DELIVERABLE[f"{endpoint}/{wave.family}"]
    )


def _check_absent_endpoints_are_declared() -> None:
    """An endpoint a wave does not carry is a decision, never an omission.

    This is the check that keeps the two over-assumptions in EXP-R2-225's text
    visible. Both are cases where the prereg licenses an endpoint and the code
    refuses it, and the failure mode to avoid is not the refusal -- it is a
    report that simply has no MegaScale block and reads as though nobody asked.
    """

    for wave in WAVES:
        for endpoint in ENDPOINTS:
            if endpoint in wave.endpoints:
                continue
            key = f"{endpoint}/{wave.family}"
            if key not in NOT_DELIVERABLE:
                raise AssertionError(
                    f"wave {wave.name!r} carries no {endpoint!r} endpoint and "
                    f"NOT_DELIVERABLE has no {key!r} entry saying why"
                )


_check_absent_endpoints_are_declared()


def require_frozen_bootstrap(resamples: int, seed: int) -> None:
    """Refuse CLI changes to the bootstrap both campaigns inherit."""

    if resamples != BOOTSTRAP_RESAMPLES or seed != DEFAULT_BOOTSTRAP_SEED:
        raise ValueError(
            "EXP-R2-225 inherits EXP-R2-224's pre-data amendment at "
            f"resamples={BOOTSTRAP_RESAMPLES}, seed={DEFAULT_BOOTSTRAP_SEED}; "
            f"got resamples={resamples}, seed={seed}"
        )


# --------------------------------------------------------------- the endpoints


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def progen3_self_check(models: dict[str, dict[str, Any]], *, rungs: tuple[str, ...]) -> dict[str, Any]:
    """This wave's qualification, read from the stage-20 payloads themselves.

    ProGen3 has no stage-41 identification record and cannot have one: it has no
    ``ArmSpec``, so ``cohort_power.py`` cannot reach it. What it does have is
    stronger for the purpose an interface qualification serves -- a per-checkpoint
    NLL band measured on that checkpoint, against three recorded corruptions
    including a strict-clean wrong expert mapping that ``load_state_dict`` cannot
    see -- and ``retrieval_bound.py`` runs it before it scores anything and
    records it in the payload. Requiring it here rather than trusting the run is
    what makes a payload produced without it unusable instead of merely
    undocumented.
    """

    per_rung: dict[str, Any] = {}
    for name in rungs:
        loader = models[name].get("loader") or {}
        check = loader.get("self_check")
        if not isinstance(check, dict):
            raise ValueError(
                f"{name}: the stage-20 payload records no ProGen3 self_check, so "
                "there is no evidence its experts and routers were loaded rather "
                "than freshly initialised"
            )
        if check.get("verdict") != "PASS":
            raise ValueError(
                f"{name}: ProGen3 self-check verdict is {check.get('verdict')!r}, "
                "not PASS"
            )
        per_rung[name] = {
            "checkpoint": check.get("checkpoint"),
            "nll": check.get("nll"),
            "band": check.get("band"),
            "verdict": check.get("verdict"),
        }
    checkpoints = {row["checkpoint"] for row in per_rung.values()}
    if len(checkpoints) != len(rungs):
        raise ValueError(
            "two rungs self-checked against the same declared checkpoint band: "
            f"{sorted(checkpoints)}"
        )
    return {
        "source": "stage20_loader_self_check",
        "passed": True,
        "rungs": per_rung,
        "note": "src.capability.models.progen3.self_check, resolved per checkpoint by "
        "config fingerprint; a band borrowed from another rung is a KeyError "
        "there and never a number",
    }


def dms_endpoint(
    models: dict[str, dict[str, Any]],
    lookup: dict[str, Any],
    *,
    wave: Wave,
    resamples: int,
    seed: int,
    require_fixed_census: bool = True,
) -> dict[str, Any]:
    """This wave's ProteinGym reading: raw, MODEL - LOOKUP and MODEL - BLOSUM62."""

    require_endpoint(wave, "dms")
    C.require_rungs(list(models), wave.rungs)
    dtype = C.require_uniform_dtype(models, rungs=wave.rungs, label="DMS")
    stratum = C.require_uniform_stratum(models, rungs=wave.rungs, label="DMS")
    declared = WAVE_STRATUM[wave.name]
    if stratum != declared:
        raise ValueError(
            f"wave {wave.name!r} is declared on the {declared!r} stratum and its "
            f"payloads record {stratum!r}"
        )
    alignment = C.align_dms(
        models,
        lookup,
        rungs=wave.rungs,
        context=PROGEN3_CONTEXT,
        exclusion_reason=DMS_CONTEXT_EXCLUSION_REASON,
        census=PROGEN3_DMS_CENSUS if require_fixed_census else None,
    )
    units = alignment["units"]

    def endpoint(values: dict[str, dict[str, float]], offset: int) -> dict[str, Any]:
        return C.endpoint(
            values,
            units,
            rungs=wave.rungs,
            pairs=wave.pairs,
            resamples=resamples,
            seed=seed + offset,
        )

    return {
        "declared_cohort": {
            "n_assays": len(alignment["declared_assays"]),
            "n_clusters": alignment["declared_clusters"],
            "source": "ProteinGym substitution assays, wild-type families at 50% identity",
        },
        "context_excluded_assays": alignment["context_excluded_assays"],
        "n_assays": len(alignment["analysis_assays"]),
        "n_clusters": len(set(units.values())),
        "unit": "wild-type family at 50% identity",
        "precision": dtype,
        "scoring_stratum": stratum,
        "lookup_note": LOOKUP_IS_NOT_A_RETRIEVAL_BOUND,
        "corpus": {name: models[name].get("corpus") for name in wave.rungs},
        "raw_spearman": endpoint(alignment["raw"], 0),
        "model_minus_lookup": endpoint(alignment["contrasts"]["model_minus_lookup"], 20),
        "model_minus_blosum62": endpoint(
            alignment["contrasts"]["model_minus_blosum62"], 40
        ),
    }


def dms_gate(wave: Wave, dms: dict[str, Any]) -> dict[str, Any]:
    """The one DMS gate, on same-family adjacent rungs only.

    The compound is the EXP-R2-224 pre-data amendment's, unchanged and not
    re-derived: the larger rung's MODEL - LOOKUP 95% lower bound is > 0, **and**
    the paired delta-rho 95% lower bound on **raw Spearman** is > 0. MODEL -
    BLOSUM62 is reported and is not a condition.
    """

    gates: dict[str, Any] = {}
    for pair in wave.pairs:
        require_label_eligible(wave, pair)
        smaller, larger = pair
        key = f"{smaller}__{larger}"
        conditions = {
            "larger_model_minus_lookup": C.lower_bound_positive(
                dms["model_minus_lookup"]["per_rung"][larger]
            ),
            "raw_spearman_delta": C.lower_bound_positive(
                dms["raw_spearman"]["adjacent_delta_rho"][key]
            ),
        }
        gates[key] = {
            "label": "descriptive_gate_transition",
            "verdict": C.compound_verdict(conditions),
            "conditions": conditions,
            "blosum62_is_not_a_dms_gate": True,
            "transition_means": TRANSITION_MEANS,
            "reported_not_gated": {
                "smaller_model_minus_lookup": C.lower_bound_positive(
                    dms["model_minus_lookup"]["per_rung"][smaller]
                )
            },
        }
    return gates


def context_information(
    wave: Wave,
    *,
    shared_report: dict[str, Any] | None = None,
    per_rung_reports: dict[str, dict[str, Any]] | None = None,
    row_arm: str | None = None,
) -> dict[str, Any]:
    """A wave's qualification endpoint. Never a gate, never a transition."""

    require_endpoint(wave, "context_information")
    if wave.qualification == "stage41_shared_report":
        if shared_report is None:
            raise ValueError(f"{wave.name} needs one stage-41 report over its rungs")
        record = C.qualify_stage41(shared_report, rungs=wave.rungs)
    elif wave.qualification == "stage41_per_rung":
        if per_rung_reports is None or row_arm is None:
            raise ValueError(
                f"{wave.name} needs one stage-41 report per rung and the "
                "condition its rows are read on"
            )
        record = C.qualify_per_rung_stage41(
            per_rung_reports, rungs=wave.rungs, row_arm=row_arm
        )
    else:
        raise ValueError(
            f"{wave.name} declares qualification {wave.qualification!r}, which is "
            "not a context-information route"
        )
    record["is_a_gate"] = False
    record["not_a_gate_note"] = CONTEXT_INFORMATION_IS_NOT_A_GATE
    record["label_eligible"] = False
    return record


# ------------------------------------------------------------------- assembly


def compare_second_stage(
    *,
    wave_name: str,
    dms_models: dict[str, dict[str, Any]] | None = None,
    lookup: dict[str, Any] | None = None,
    shared_qualification: dict[str, Any] | None = None,
    per_rung_qualification: dict[str, dict[str, Any]] | None = None,
    qualification_row_arm: str | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    require_fixed_census: bool = True,
) -> dict[str, Any]:
    wave = require_wave(wave_name)
    require_frozen_bootstrap(resamples, seed)
    if dms_models is not None:
        # Before anything reads a payload: a rung set that is not this wave's is
        # refused as a wrong ladder, never as a missing key inside a check.
        C.require_rungs(list(dms_models), wave.rungs)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "second_stage_capability_report",
        "created_utc": _timestamp(),
        "campaign": "EXP-R2-225",
        "wave": wave.name,
        "family": wave.family,
        "rungs": list(wave.rungs),
        "adjacent_pairs": [list(pair) for pair in wave.pairs],
        "declared_endpoints": list(wave.endpoints),
        "wave_note": wave.note,
        "not_panel_admission": True,
        "descriptive_not_causal": True,
        "descriptive_not_causal_note": DESCRIPTIVE_NOT_CAUSAL,
        "no_biological_knowledge_claim": True,
        "no_biological_knowledge_claim_note": NO_BIOLOGICAL_KNOWLEDGE_CLAIM,
        "no_cross_task_total": True,
        "not_deliverable": dict(NOT_DELIVERABLE),
        "single_points": {
            name: {"scope": scope, "label_eligible": False, "note": EXISTENCE_ONLY}
            for name, scope in SINGLE_POINTS.items()
        },
        "bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "source": "scale_capability.py, the EXP-R2-224 pre-data amendment",
        },
    }
    # One qualification per wave, chosen by the route the wave declares rather
    # than by whichever endpoint block runs last.
    if wave.qualification == "stage20_progen3_self_check":
        if dms_models is None:
            raise ValueError(
                f"{wave.name} qualifies from its stage-20 payloads and was given none"
            )
        payload["qualification"] = progen3_self_check(dms_models, rungs=wave.rungs)
    else:
        payload["qualification"] = context_information(
            wave,
            shared_report=shared_qualification,
            per_rung_reports=per_rung_qualification,
            row_arm=qualification_row_arm,
        )
    if "dms" in wave.endpoints:
        if dms_models is None or lookup is None:
            raise ValueError(f"{wave.name} declares a dms endpoint and was given none")
        dms = dms_endpoint(
            dms_models,
            lookup,
            wave=wave,
            resamples=resamples,
            seed=seed,
            require_fixed_census=require_fixed_census,
        )
        payload["dms"] = dms
        payload["descriptive_gate_transitions"] = {"dms": dms_gate(wave, dms)}
    if "context_information" in wave.endpoints:
        payload["descriptive_gate_transitions"] = {
            "dms": {},
            "note": "this wave carries no fitness endpoint, so no gate exists "
            "for it. A context-information curve is not a descriptive_gate_"
            "transition",
        }
    return payload


def _load_rung_models(directory: Path, wave: Wave) -> dict[str, dict[str, Any]]:
    models = {}
    for name in wave.rungs:
        path = directory / f"model_{name}.json"
        payload = _read(path)
        if payload.get("arm") != name:
            raise ValueError(
                f"{path} declares arm {payload.get('arm')!r}, expected {name}"
            )
        models[name] = payload
    return models


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", required=True, choices=sorted(WAVES_BY_NAME))
    parser.add_argument(
        "--retrieval-bound-dir",
        type=Path,
        default=None,
        help="directory holding model_<rung>.json and lookup.json from stage 20; "
        "required by a wave that declares a dms endpoint",
    )
    parser.add_argument(
        "--context-information-summary",
        type=Path,
        default=None,
        help="full stage-41 report over this wave's rungs, for a wave whose "
        "rungs share one report",
    )
    parser.add_argument(
        "--context-information-report",
        action="append",
        default=[],
        metavar="RUNG=PATH",
        help="one stage-41 report per rung, for a joint wave whose rungs each "
        "wrote their own; repeat once per rung",
    )
    parser.add_argument(
        "--qualification-row-arm",
        default=None,
        help="the condition a per-rung stage-41 report is read on, e.g. "
        "protein_declared or text_declared",
    )
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    wave = require_wave(args.wave)
    require_frozen_bootstrap(args.bootstrap, args.seed)

    dms_models = lookup = None
    if "dms" in wave.endpoints:
        if args.retrieval_bound_dir is None:
            parser.error(f"wave {wave.name} needs --retrieval-bound-dir")
        dms_models = _load_rung_models(args.retrieval_bound_dir, wave)
        lookup = _read(args.retrieval_bound_dir / "lookup.json")

    shared = per_rung = None
    if "context_information" in wave.endpoints:
        if wave.qualification == "stage41_shared_report":
            if args.context_information_summary is None:
                parser.error(f"wave {wave.name} needs --context-information-summary")
            shared = _read(args.context_information_summary)
        else:
            if not args.context_information_report or args.qualification_row_arm is None:
                parser.error(
                    f"wave {wave.name} needs one --context-information-report per "
                    "rung and --qualification-row-arm"
                )
            per_rung = {}
            for item in args.context_information_report:
                rung, _, path = item.partition("=")
                if not path:
                    parser.error(f"--context-information-report expects RUNG=PATH, got {item!r}")
                per_rung[rung] = _read(Path(path))

    payload = compare_second_stage(
        wave_name=wave.name,
        dms_models=dms_models,
        lookup=lookup,
        shared_qualification=shared,
        per_rung_qualification=per_rung,
        qualification_row_arm=args.qualification_row_arm,
        resamples=args.bootstrap,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / REPORT_NAME
    write_json(destination, payload)
    print(f"wrote {destination}")
    for pair, block in payload.get("descriptive_gate_transitions", {}).get("dms", {}).items():
        print(f"DMS gate {pair}: {block['verdict']}")


if __name__ == "__main__":
    main()
