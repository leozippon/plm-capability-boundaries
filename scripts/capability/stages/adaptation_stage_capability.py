#!/usr/bin/env python3
"""Descriptive base → Stage 1 → Stage 2 comparison on the ProLLaMA lineage.

EXP-R2-226. This stage is CPU-only. It consumes existing stage-20 and stage-29
artefacts, this campaign's own per-rung qualification artefacts, and the
per-rung stage-41 reports the lineage was already qualified on. It loads no
model, downloads no weights and synthesises no cross-task total.

**What the axis is, and what it is not.** The ladder is a training *stage* axis:
one set of weights, adapted twice. F14 measured what the same three checkpoints
**lost** on a text task; this measures what they bought on the two frozen
protein-capability queues the repository already owns. Differences are
descriptive of these checkpoints. They are not identified as a causal effect of
"continued pretraining": the step that separates base from Stage 1 changes the
corpus, retrains ``embed_tokens`` and ``lm_head`` outright, and fixes a LoRA
rank, a schedule and a data order at once, so what a transition prices is the
released adaptation **as a whole**.

**The verdict label is** :data:`TRANSITION_LABEL`, deliberately not
``descriptive_gate_transition``. The arithmetic is EXP-R2-224's compound; that
label is reserved for adjacent rungs of a same-family *scale* ladder. The two
must never be pooled into one table of transitions, and a row from here may not
be counted alongside a ProGen2 rung transition.

**The two ladders do not pool, and that is enforced rather than asserted.**
Three independent refusals stand between them. The rung list is fixed at this
lineage's three names. The DMS census is 217 assays over 174 families with an
**empty** context-exclusion set -- EXP-R2-224's ladder analyses 201 over 163
because 16 assays overflow ProGen2's 1024 positions, and this lineage's longest
wild type renders to 2231 tokens against 4096 -- so a ProGen2 payload cannot
satisfy this campaign's census. And every payload must declare the **token** as
its scored symbol unit: this family's summed log-likelihood is over merged
multi-residue SentencePiece pieces and a residue-unit family's is over single
residues, which is a different scoring functional even on a common support.

**The ladder's length is decided by the qualification and not by a result.** If
Stage 2 fails a clause, the reported ladder is base → Stage 1, the Stage 1–Stage
2 pair is not formed, and the failure is reported with the clause that fired.
The instruction rendering is never fallen back to in order to recover it. If
Stage 1 fails, the campaign stops entirely: the base rung alone is not a ladder
and its solitary reading is not published as a capability measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.mutation import designed_referent as D  # noqa: E402
from src.capability.models import joint_lineage as L  # noqa: E402
from src.capability.models import scale_comparison as C  # noqa: E402
from src.capability.core.arms import REPO  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.models.joint_modes import TOKEN_UNIT  # noqa: E402
from src.capability.models.scale_comparison import compound_verdict, lower_bound_positive  # noqa: E402


from scripts.capability.entrypoints import stage_path

def _load_stage(filename: str) -> Any:
    import importlib.util

    path = stage_path(REPO_ROOT, filename)
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


QUALIFICATION = _load_stage("adaptation_stage_qualification.py")

SCHEMA_VERSION = "r2_adaptation_stage_capability_v1"
DEFAULT_OUT = REPO / "results/R1/adaptation_stage_capability"

#: The verdict label. Not ``descriptive_gate_transition``: that one is reserved
#: for adjacent rungs of a same-family scale ladder, and this axis is a training
#: stage.
TRANSITION_LABEL = "adaptation_stage_transition"

#: The paired group bootstrap, frozen before any score on this ladder existed.
#: These two numbers are not revised after a result exists.
BOOTSTRAP_RESAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260826

#: The shared context every rung of this lineage declares. All three are
#: ``LlamaForCausalLM`` at ``max_position_embeddings`` 4096.
LINEAGE_CONTEXT = 4096
DMS_CONTEXT_EXCLUSION_REASON = "exceeds this arm's context"

#: **This campaign's census, and it is not EXP-R2-224's.** Rendering all 187
#: ProteinGym wild types as ``Seq=<...>`` through this lineage's shared tokenizer
#: gives a maximum of 2231 tokens -- assay family q00109, 3418 residues --
#: against 4096, a headroom of 1865 tokens, with a median of 154. A substitution
#: does not change a sequence's length, so every one of the 217 assays over 174
#: families is reachable on all three rungs and the analysis set equals the
#: declared cohort. The construction rule still runs and still refuses: a skip
#: taken by one rung and not another removes that assay from the analysis set on
#: every rung, and a skip against any context other than 4096 is a refusal.
DMS_DECLARED_ASSAYS = 217
DMS_DECLARED_CLUSTERS = 174
DMS_ANALYSIS_ASSAYS = 217
DMS_ANALYSIS_CLUSTERS = 174
DMS_CONTEXT_EXCLUDED_ASSAYS: tuple[str, ...] = ()

MEGASCALE_DESIGN_WILDTYPES = 130
MEGASCALE_DESIGN_SERIES = 40
MEGASCALE_NATURAL_WILDTYPES = 266
MEGASCALE_NATURAL_CLUSTERS = 124
HYDROPATHY_BASELINE = "hydropathy_change"

DMS_CENSUS = C.DmsCensus(
    label="EXP-R2-226",
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

#: The stage-41 condition this ladder is read on. A joint checkpoint's stage-41
#: report names the *condition* rather than the checkpoint, so the declared and
#: the reversed conditions of one checkpoint can never be paired by accident.
STAGE41_ROW_ARM = "protein_declared"

DESCRIPTIVE_NOT_CAUSAL = (
    "differences on this ladder are descriptive of the named checkpoints. They "
    "are not identified as a causal effect of continued pretraining: the step "
    "that separates base from Stage 1 changes the corpus, retrains embed_tokens "
    "and lm_head outright -- 262.1 M of a 582.0 M trainable budget -- and fixes a "
    "LoRA rank, a schedule and a data order at once, so what is priced is this "
    "released adaptation as a whole"
)
NO_BIOLOGICAL_KNOWLEDGE_CLAIM = (
    "no biological-knowledge claim is licensed. Beating a sequence baseline is "
    "not evidence that a model has learned biology, and §7.0 does not gate this "
    "campaign because no knowledge claim is in play"
)
NOT_CONTAMINATION_CONTROLLED = (
    "ProLLaMA's declared continued-pretraining corpus family is UniRef50, which "
    "contains the Swiss-Prot proteins these assays are built on. This campaign "
    "does not exclude retrieval, does not bound it, and must not be read as doing "
    "either"
)
BASE_RUNG_IS_A_FLOOR = (
    "the base rung reads a directional-reversal cost of -0.0013 nats per scored "
    "token, so whatever correlation it returns is what these queues yield from a "
    "checkpoint with no directional reading of sequence. It is a pre-adaptation "
    "reference and never a protein capability"
)
NOT_A_STATEMENT_ABOUT_THE_INSTRUCTION_FORMAT = (
    "Stage 2 is scored under the bare Seq=<...> block and not under its own "
    "declared [Generate by superfamily] instruction form, by design. A Stage 2 "
    "reading at or below Stage 1's is NOT evidence that instruction tuning "
    "removed a capability"
)
TRANSITION_MEANS = (
    "the later rung clears its own baseline contrast and is paired-significantly "
    "better than the earlier one; this is not a claim that the earlier one failed"
)
NEVER_POOLED = (
    "this ladder is never placed in one column with EXP-R2-224's ProGen2 scale "
    "ladder or with any ProtGPT2, ProGen3 or Galactica reading. Two censuses and "
    "two scoring functionals: 217 assays over 174 families here against 201 over "
    "163 there, and a summed log-likelihood over merged multi-residue pieces here "
    "against one over single residues there"
)
SYMBOL_UNIT_NOTE = (
    "one scored symbol on this ladder is one token carrying one or more residues, "
    "so every magnitude is in nats per token and none is commensurable with a "
    "residue-unit family's (Appendix B rule 26, limitation L23). Spearman rho is "
    "invariant under any strictly increasing per-assay transformation of the score "
    "and carries no unit at all, which is why the ranking estimand is admissible "
    "where a per-token one is not; the measured rate is reported beside it anyway"
)
PRECISION_NOTE = (
    "one precision per endpoint across every rung. The released checkpoints "
    "declare float16, so bfloat16 is a declared cast -- identical on all rungs, "
    "which is what the paired difference requires"
)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_frozen_bootstrap(resamples: int, seed: int) -> None:
    """Refuse CLI changes to the pre-data EXP-R2-226 bootstrap freeze."""

    if resamples != BOOTSTRAP_RESAMPLES or seed != DEFAULT_BOOTSTRAP_SEED:
        raise ValueError(
            "EXP-R2-226 freezes this stage at "
            f"resamples={BOOTSTRAP_RESAMPLES}, seed={DEFAULT_BOOTSTRAP_SEED}; "
            f"got resamples={resamples}, seed={seed}"
        )


# -------------------------------------------------------------- the ladder


def resolve_ladder(qualification_dir: Path) -> dict[str, Any]:
    """How long this ladder is, decided by the qualification and never by a score.

    The base rung and Stage 1 are required: the base rung alone is not a ladder
    and its solitary reading is not published as a capability measurement, so a
    Stage 1 that did not qualify stops the campaign here. Stage 2 is optional by
    declaration -- its failure shortens the ladder to base → Stage 1 and is
    reported with the clause that fired.
    """

    base, stage1, stage2 = L.LINEAGE_RUNGS
    verdicts: dict[str, Any] = {}
    for name in (base, stage1):
        verdicts[name] = QUALIFICATION.read_verdict(
            qualification_dir, name, dtype=QUALIFICATION.CAMPAIGN_DTYPE
        )
    try:
        verdicts[stage2] = QUALIFICATION.read_verdict(
            qualification_dir, stage2, dtype=QUALIFICATION.CAMPAIGN_DTYPE
        )
    except (FileNotFoundError, ValueError) as failure:
        return {
            "rungs": (base, stage1),
            "pairs": ((base, stage1),),
            "record": {
                "rungs": [base, stage1],
                "fallback": "base_to_stage_1_only",
                "stage_2_refusal": str(failure),
                "stage_2_clause": _failed_clause(qualification_dir, stage2),
                "note": (
                    "Stage 2 did not qualify, so the Stage 1-Stage 2 pair is not "
                    "formed and the instruction rendering is NOT fallen back to in "
                    "order to recover it. This is EXP-R2-226's declared branch and "
                    "not a decision taken after a score"
                ),
                "qualification": _clause_summary(verdicts),
            },
        }
    return {
        "rungs": L.LINEAGE_RUNGS,
        "pairs": L.ADJACENT_PAIRS,
        "record": {
            "rungs": list(L.LINEAGE_RUNGS),
            "fallback": None,
            "qualification": _clause_summary(verdicts),
        },
    }


def _clause_summary(verdicts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "verdict": payload["verdict"],
            "strict_load": payload["strict_load"]["verdict"],
            "nll_self_check": payload["nll_self_check"]["verdict"],
            "directional_reversal": payload["directional_reversal"]["verdict"],
            "directional_reversal_cost_nats_per_scored_token": payload[
                "directional_reversal"
            ].get("cost_nats_per_scored_token"),
        }
        for name, payload in verdicts.items()
    }


def _failed_clause(directory: Path, rung: str) -> str | None:
    """The clause a refused rung's own artefact names, when it wrote one."""

    path = Path(directory) / QUALIFICATION.artefact_name(rung)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("failed_clause")


# ------------------------------------------------------- the pooling refusals


def require_uniform_symbol_unit(
    models: Mapping[str, dict[str, Any]], *, rungs: Sequence[str], label: str
) -> dict[str, Any]:
    """One scoring symbol unit across the rungs, and it must be this family's.

    :func:`src.capability.models.scale_comparison.require_uniform_stratum`'s shape on the
    axis that separates *this* ladder from EXP-R2-224's. Both are N-to-C summed
    log-likelihoods, so the stratum alone does not tell them apart; what does is
    that one sums over merged multi-residue pieces and the other over single
    residues. A payload that records no unit is refused rather than assumed to
    share the others': the field is written by the scorer that produced it, and
    its absence means the payload was produced by a scorer that is not this one.
    """

    seen: dict[str, str] = {}
    rates: dict[str, float] = {}
    for name in rungs:
        settings = models[name].get("settings")
        accounting = (
            settings.get("symbol_unit_accounting") if isinstance(settings, Mapping) else None
        )
        if not isinstance(accounting, Mapping) or not accounting.get("symbol_unit"):
            raise ValueError(
                f"{label}: {name} records no symbol_unit_accounting, so it was not "
                "produced by this lineage's scorer and cannot be paired into this "
                "ladder"
            )
        seen[name] = str(accounting["symbol_unit"])
        rates[name] = float(accounting["residues_per_scored_token"])
    if set(seen.values()) != {TOKEN_UNIT}:
        raise ValueError(
            f"{label}: the rungs were scored under mixed or foreign symbol units: "
            f"{seen}. A summed log-likelihood over merged multi-residue pieces and "
            "one over single residues are different scoring functionals and are "
            "never pooled to complete a ladder"
        )
    return {
        "symbol_unit": TOKEN_UNIT,
        "residues_per_scored_token": rates,
        "note": SYMBOL_UNIT_NOTE,
    }


# ------------------------------------------------------------- the endpoints


def _endpoint(
    per_rung: dict[str, dict[str, float]],
    units: dict[str, str],
    *,
    rungs: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    return C.endpoint(
        per_rung, units, rungs=rungs, pairs=pairs, resamples=resamples, seed=seed
    )


def zero_hit_design_cohort(path: Path) -> dict[str, Any]:
    """The certified zero-hit design census, read from the stage-29 cohort that
    carries the flag."""

    referent = D.load_referent(path)
    names = sorted(wildtype.name for wildtype in referent.side("design"))
    if not names:
        raise ValueError(f"{path} carries no certified zero-hit design")
    return {
        "source": path.name,
        "cohort_sha256": sha256_file(path),
        "zero_hit_designs": names,
    }


def fragment_margins(
    fragment_order: dict[str, Any] | None, *, rungs: Sequence[str]
) -> dict[str, Any]:
    """The 3-7-mer margins this ladder carries, and the rungs it does not.

    EXP-R2-226 lists these in its ``reported_not_gated`` block "where the
    artefact carries them, with the rungs it does not carry named". The
    stage-29 fragment pass is a separate run over the corpus k-mer background
    and this campaign did not commission one, so the honest report is which
    rungs the existing artefact holds -- and naming the ones it does not is the
    point, because an endpoint nobody ran must not read as an endpoint that
    returned nothing.
    """

    arms = (fragment_order or {}).get("arms") or {}
    present = [name for name in rungs if name in arms]
    missing = [name for name in rungs if name not in arms]
    return {
        "reported_not_gated": True,
        "rungs_present": present,
        "rungs_missing": missing,
        "source_arms": sorted(arms),
        "not_run_reason": (
            None
            if not missing
            else "the stage-29 fragment_order pass on disk carries no rung of this "
            "lineage, so these margins were not measured for this ladder. They are "
            "reported and never gated, so their absence stops nothing; it is named "
            "here rather than silently omitted"
        ),
        "cohort_sha256": (fragment_order or {}).get("cohort_sha256"),
    }


def adaptation_stage_transitions(
    dms: dict[str, Any] | None,
    megascale: dict[str, Any] | None,
    *,
    pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """The two pre-registered compounds, judged independently. No composite.

    Neither compound ever asks the earlier rung to fail. A rule that required the
    earlier rung's own contrast to be non-significant would make the gate depend
    on accepting a null, so a well-powered pair in which both rungs are competent
    could not register a transition however large the improvement (L44). The
    earlier rung's contrasts are reported beside each gate instead.
    """

    dms_gates: dict[str, Any] = {}
    mega_gates: dict[str, Any] = {}
    for earlier, later in pairs:
        pair = f"{earlier}__{later}"
        if dms is not None:
            conditions = {
                "later_model_minus_lookup": lower_bound_positive(
                    dms["model_minus_lookup"]["per_rung"][later]
                ),
                "raw_spearman_delta": lower_bound_positive(
                    dms["raw_spearman"]["adjacent_delta_rho"][pair]
                ),
            }
            dms_gates[pair] = {
                "verdict": compound_verdict(conditions),
                "conditions": conditions,
                "blosum62_is_not_a_dms_gate": True,
                "transition_means": TRANSITION_MEANS,
                "reported_not_gated": {
                    "earlier_model_minus_lookup": lower_bound_positive(
                        dms["model_minus_lookup"]["per_rung"][earlier]
                    ),
                    "later_model_minus_blosum62": lower_bound_positive(
                        dms["model_minus_blosum62"]["per_rung"][later]
                    ),
                    "earlier_model_minus_blosum62": lower_bound_positive(
                        dms["model_minus_blosum62"]["per_rung"][earlier]
                    ),
                },
            }
        if megascale is not None:
            conditions = {
                "design_later_model_minus_hydropathy": lower_bound_positive(
                    megascale["designs"]["model_minus_hydropathy"]["per_rung"][later]
                ),
                "design_later_model_minus_blosum62": lower_bound_positive(
                    megascale["designs"]["model_minus_blosum62"]["per_rung"][later]
                ),
                "natural_later_model_minus_hydropathy": lower_bound_positive(
                    megascale["control"]["model_minus_hydropathy"]["per_rung"][later]
                ),
                "natural_later_model_minus_blosum62": lower_bound_positive(
                    megascale["control"]["model_minus_blosum62"]["per_rung"][later]
                ),
                "design_raw_spearman_delta": lower_bound_positive(
                    megascale["designs"]["raw_spearman"]["adjacent_delta_rho"][pair]
                ),
            }
            mega_gates[pair] = {
                "verdict": compound_verdict(conditions),
                "conditions": conditions,
                "transition_means": TRANSITION_MEANS,
                "reported_not_gated": {
                    "natural_raw_spearman_delta": lower_bound_positive(
                        megascale["control"]["raw_spearman"]["adjacent_delta_rho"][pair]
                    ),
                    "earlier_design_model_minus_hydropathy": lower_bound_positive(
                        megascale["designs"]["model_minus_hydropathy"]["per_rung"][earlier]
                    ),
                    "earlier_design_model_minus_blosum62": lower_bound_positive(
                        megascale["designs"]["model_minus_blosum62"]["per_rung"][earlier]
                    ),
                },
            }
    return {
        "label": TRANSITION_LABEL,
        "label_note": (
            "deliberately not descriptive_gate_transition, which is reserved for "
            "adjacent rungs of a same-family SCALE ladder. This axis is a training "
            "stage and a row here may not be counted alongside a ProGen2 rung "
            "transition"
        ),
        "dms": dms_gates,
        "megascale": mega_gates,
    }


def compare_lineage(
    *,
    ladder: dict[str, Any],
    dms_models: dict[str, dict[str, Any]] | None,
    lookup: dict[str, Any] | None,
    megascale_models: dict[str, dict[str, Any]] | None,
    baselines: dict[str, Any] | None,
    design_cohort: dict[str, Any] | None,
    fragment_order: dict[str, Any] | None,
    qualification_reports: dict[str, dict[str, Any]],
    resamples: int,
    seed: int,
    require_fixed_census: bool = True,
) -> dict[str, Any]:
    rungs = tuple(ladder["rungs"])
    pairs = tuple(ladder["pairs"])
    context_qualification = C.qualify_per_rung_stage41(
        qualification_reports, rungs=rungs, row_arm=STAGE41_ROW_ARM
    )

    dms: dict[str, Any] | None = None
    dms_units_record: dict[str, Any] | None = None
    if dms_models is not None:
        if lookup is None:
            raise ValueError("the DMS endpoint needs LOOKUP")
        C.require_rungs(list(dms_models), rungs)
        dms_dtype = C.require_uniform_dtype(dms_models, rungs=rungs, label="DMS")
        dms_stratum = C.require_uniform_stratum(dms_models, rungs=rungs, label="DMS")
        dms_units_record = require_uniform_symbol_unit(
            dms_models, rungs=rungs, label="DMS"
        )
        alignment = C.align_dms(
            dms_models,
            lookup,
            rungs=rungs,
            context=LINEAGE_CONTEXT,
            exclusion_reason=DMS_CONTEXT_EXCLUSION_REASON,
            census=DMS_CENSUS if require_fixed_census else None,
        )
        units = alignment["units"]
        dms = {
            "declared_cohort": {
                "n_assays": len(alignment["declared_assays"]),
                "n_clusters": alignment["declared_clusters"],
                "source": "ProteinGym substitution assays, wild-type families at 50% identity",
            },
            "context": LINEAGE_CONTEXT,
            "context_excluded_assays": alignment["context_excluded_assays"],
            "n_assays": len(alignment["analysis_assays"]),
            "n_clusters": len(set(units.values())),
            "unit": "wild-type family at 50% identity",
            "precision": dms_dtype,
            "scoring_stratum": dms_stratum,
            "symbol_unit": dms_units_record,
            "raw_spearman": _endpoint(
                alignment["raw"], units, rungs=rungs, pairs=pairs,
                resamples=resamples, seed=seed,
            ),
            "model_minus_lookup": _endpoint(
                alignment["contrasts"]["model_minus_lookup"], units, rungs=rungs,
                pairs=pairs, resamples=resamples, seed=seed + 20,
            ),
            "model_minus_blosum62": _endpoint(
                alignment["contrasts"]["model_minus_blosum62"], units, rungs=rungs,
                pairs=pairs, resamples=resamples, seed=seed + 40,
            ),
        }

    megascale: dict[str, Any] | None = None
    if megascale_models is not None:
        if baselines is None or design_cohort is None:
            raise ValueError("the MegaScale endpoint needs baselines and the cohort")
        C.require_rungs(list(megascale_models), rungs)
        mega_dtype = C.require_uniform_dtype(
            megascale_models, rungs=rungs, label="MegaScale"
        )
        mega_units_record = require_uniform_symbol_unit(
            megascale_models, rungs=rungs, label="MegaScale"
        )
        digests = {name: payload["cohort_sha256"] for name, payload in megascale_models.items()}
        digests["baselines"] = baselines["cohort_sha256"]
        if len(set(digests.values())) != 1:
            raise ValueError(f"MegaScale cohort_sha256 disagree: {digests}")
        if design_cohort["cohort_sha256"] != baselines["cohort_sha256"]:
            raise ValueError(
                "the zero-hit design census was read from a different cohort: "
                f"{design_cohort['cohort_sha256']} against {baselines['cohort_sha256']}"
            )
        design_names = frozenset(design_cohort["zero_hit_designs"])
        megascale = {
            "cohort_sha256": baselines["cohort_sha256"],
            "hydropathy_baseline": HYDROPATHY_BASELINE,
            "precision": mega_dtype,
            "symbol_unit": mega_units_record,
            "design_census": {
                "source": design_cohort["source"],
                "n_zero_hit_designs": len(design_names),
            },
        }
        for side, label, unit_name, offset in (
            ("design", "designs", "design series", 60),
            ("natural", "control", "WT_cluster", 80),
        ):
            admissible = design_names if side == "design" else None
            census = MEGASCALE_CENSUS if require_fixed_census else None
            units, raw, hydro = C.align_megascale(
                megascale_models, baselines, rungs=rungs, side=side,
                baseline_name=HYDROPATHY_BASELINE, admissible=admissible, census=census,
            )
            _, _, blosum = C.align_megascale(
                megascale_models, baselines, rungs=rungs, side=side,
                baseline_name="blosum62", admissible=admissible, census=census,
            )
            megascale[label] = {
                "unit": unit_name,
                "n_wildtypes": len(units),
                "raw_spearman": _endpoint(
                    raw, units, rungs=rungs, pairs=pairs,
                    resamples=resamples, seed=seed + offset,
                ),
                "model_minus_hydropathy": _endpoint(
                    hydro, units, rungs=rungs, pairs=pairs,
                    resamples=resamples, seed=seed + offset + 5,
                ),
                "model_minus_blosum62": _endpoint(
                    blosum, units, rungs=rungs, pairs=pairs,
                    resamples=resamples, seed=seed + offset + 10,
                ),
            }

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "adaptation_stage_capability_report",
        "campaign": "EXP-R2-226",
        "created_utc": _timestamp(),
        "rungs": list(rungs),
        "adjacent_pairs": [list(pair) for pair in pairs],
        "ladder": ladder["record"],
        "rendering": {
            "family": L.RENDERING_FAMILY,
            "mode": L.PROTEIN_MODE,
            "protein_context": None,
            "note": NOT_A_STATEMENT_ABOUT_THE_INSTRUCTION_FORMAT,
        },
        "not_panel_admission": True,
        "descriptive_not_causal": True,
        "descriptive_not_causal_note": DESCRIPTIVE_NOT_CAUSAL,
        "no_biological_knowledge_claim": True,
        "no_biological_knowledge_claim_note": NO_BIOLOGICAL_KNOWLEDGE_CLAIM,
        "not_contamination_controlled": True,
        "not_contamination_controlled_note": NOT_CONTAMINATION_CONTROLLED,
        "base_rung_is_a_declared_floor": BASE_RUNG_IS_A_FLOOR,
        "never_pooled": NEVER_POOLED,
        "no_cross_task_total": True,
        "precision_note": PRECISION_NOTE,
        "bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "default_seed": DEFAULT_BOOTSTRAP_SEED,
        },
        "context_information_qualification": context_qualification,
        "dms": dms,
        "dms_not_run_reason": None if dms is not None else "no stage-20 payload was supplied",
        "megascale": megascale,
        "megascale_not_run_reason": (
            None if megascale is not None else "no stage-29 payload was supplied"
        ),
        TRANSITION_LABEL + "s": adaptation_stage_transitions(dms, megascale, pairs=pairs),
        "fragment_order": fragment_margins(fragment_order, rungs=rungs),
    }


def _load_rung_models(directory: Path, rungs: Sequence[str]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for name in rungs:
        payload = _read(directory / f"model_{name}.json")
        if payload.get("arm") != name:
            raise ValueError(
                f"model_{name}.json declares arm {payload.get('arm')!r}, expected {name}"
            )
        models[name] = payload
    return models


def _stage41_reports(directory: Path, rungs: Sequence[str]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for name in rungs:
        folder = f"s21_{L.rung(name).directory_name}"
        reports[name] = _read(
            directory / folder / L.PROTEIN_MODE / "context_information_bootstrap.json"
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qualification-dir",
        type=Path,
        required=True,
        help="directory holding this campaign's adaptation_stage_qualification_"
        "<rung>.json artefacts; it is what decides the ladder's length",
    )
    parser.add_argument(
        "--context-information-dir",
        type=Path,
        required=True,
        help="directory holding s21_<checkpoint>/protein/context_information_"
        "bootstrap.json for every rung. The EXP-R2-221 clause is REUSED, not "
        "redrawn: these are the stage-21 readings at corrected seed 20260728 "
        "re-analysed under the displacement-corrected rule",
    )
    parser.add_argument(
        "--retrieval-bound-dir",
        type=Path,
        default=None,
        help="directory holding model_<rung>.json and lookup.json from stage 20. "
        "Omitted means the DMS endpoint is reported NOT RUN, never as a null",
    )
    parser.add_argument(
        "--designed-referent-dir",
        type=Path,
        default=None,
        help="directory holding model_<rung>.json, baselines.json and cohort.json "
        "from stage 29. Omitted means the MegaScale endpoint is reported NOT RUN",
    )
    parser.add_argument(
        "--fragment-order",
        type=Path,
        default=None,
        help="an existing stage-29 fragment_order.json. Its 3-7-mer margins are "
        "reported and never gated; the rungs it does not carry are named, so an "
        "endpoint nobody ran does not read as one that returned nothing",
    )
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    require_frozen_bootstrap(args.bootstrap, args.seed)

    ladder = resolve_ladder(args.qualification_dir)
    rungs = ladder["rungs"]
    reports = _stage41_reports(args.context_information_dir, rungs)

    dms_models = lookup = None
    if args.retrieval_bound_dir is not None:
        dms_models = _load_rung_models(args.retrieval_bound_dir, rungs)
        lookup = _read(args.retrieval_bound_dir / "lookup.json")
    megascale_models = baselines = design_cohort = None
    if args.designed_referent_dir is not None:
        megascale_models = _load_rung_models(args.designed_referent_dir, rungs)
        baselines = _read(args.designed_referent_dir / "baselines.json")
        design_cohort = zero_hit_design_cohort(args.designed_referent_dir / "cohort.json")
    fragment_order = None
    if args.fragment_order is not None:
        fragment_order = _read(args.fragment_order)

    payload = compare_lineage(
        ladder=ladder,
        dms_models=dms_models,
        lookup=lookup,
        megascale_models=megascale_models,
        baselines=baselines,
        design_cohort=design_cohort,
        fragment_order=fragment_order,
        qualification_reports=reports,
        resamples=args.bootstrap,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "adaptation_stage_capability.json"
    write_json(destination, payload)
    print(f"wrote {destination}")
    print(f"ladder: {' -> '.join(payload['rungs'])}")
    gates = payload[TRANSITION_LABEL + "s"]
    for endpoint, blocks in (("DMS", gates["dms"]), ("MegaScale", gates["megascale"])):
        for pair, block in blocks.items():
            print(f"{endpoint} {TRANSITION_LABEL} {pair}: {block['verdict']}")


if __name__ == "__main__":
    main()
