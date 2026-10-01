#!/usr/bin/env python3
"""One declaration of which arms the campaign stage may run.

:data:`STAGE_CONTRACTS` declares one stage, ``cohort_power``. Its entry point
is ``stages/cohort_power.py``. The stage is panel-wide, requires the ``budget``
capability, and defines the qualifying protein residue band. One invocation
scores every arm it is given and writes one combined report, so the worker
splits the stage by vocabulary regime (:func:`cohort_power_items`) instead of
dispatching it per arm.

:func:`arm_can_run` answers, for one (stage, arm) pair, whether that entry
point can produce a commensurate number for the arm, and when it cannot, which
declaration refuses it. It reads ``ArmSpec`` fields and the fields this stage
contract sets. It does not restate those declarations.

The controller and the worker are bash and the declaration is Python that
imports torch. ``--emit`` renders the resolved contract into
``h200/panel_contract.sh``, which both shell scripts source, and ``--verify``
re-derives it and refuses if the rendered file disagrees with the live panel.
The worker runs ``--verify`` in its preflight, before any GPU is scheduled.

``lens_family.py``, ``pathway_budget.py``, ``estimand_power.py``,
``induction_path_patching.py``, ``path_patching`` and ``relational_channel.py``
are not in this tree.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.arms import (  # noqa: E402
    MODEL_ROOT,
    PANEL,
    STAGED_ARMS,
    TEXT_MODEL_BASE,
    TEXT_MODEL_ROOT,
)

#: Where ``--emit`` writes and ``--verify`` reads. Beside this file so that the
#: controller's code freeze picks both up from ``scripts/capability`` without a
#: second path to keep in step.
SHELL_CONTRACT = REPO_ROOT / "h200" / "panel_contract.sh"

#: The shell contract carries the campaign panel, each stage's entry and scope,
#: the admitted ``ArmSpec.input_format`` set, named ``excluded_arms``, and the
#: protein-cohort bands. ``matches_qualifying_stage: null`` means the stage
#: declares no protein cohort. ``staged_arm_cohort_power`` and
#: ``truncation_curve_logit_limit`` answer the cohort-power ``--skip-truncation``
#: rule per staged checkpoint, because a staged rung reaches a card through a
#: campaign manifest rather than through those buckets.
SCHEMA_VERSION = "r2_transfer_panel_contract_v6"


# --------------------------------------------------------------- campaign panel

#: The arms a campaign may schedule: every :data:`~src.capability.core.arms.PANEL`
#: member whose checkpoint is staged on GPFS and byte-verified (EXP-R2-058),
#: which since 2026-08-21 is all fifteen of them.
#:
#: Staging cannot be derived from ``PANEL``, so this is declared -- but
#: :func:`_check_campaign_panel` requires every excluded panel member to carry a
#: reason in :data:`PANEL_MEMBERS_NOT_STAGED`, which makes adding an arm to
#: ``PANEL`` without deciding its campaign status an import-time failure rather
#: than a silent omission from every campaign.
#:
#: What this list is *not* is a per-stage panel. A stage that cannot read its
#: statistic on an arm refuses that arm in :data:`STAGE_CONTRACTS`, with the
#: reason, and every consumer resolves through :func:`stage_arms`. Withholding
#: campaign membership to express one stage's refusal removes the arm from every
#: other stage as well, silently and without a reason attached to any of them.
CAMPAIGN_PANEL: tuple[str, ...] = (
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "dialogpt-small",
    "qwen2.5-0.5b",
    "llama-3.2-3b",
    "protgpt2",
    "zymctrl",
    "progen2-base",
    "progen2-medium",
    "progen2-small",
    # The byte-level text control for paa_census (EXP-R2-129). Admitted 2026-08-06 on
    # the staging fact its exclusion turned on: the checkpoint is on GPFS at
    # models/bygpt5-medium-en, 1,156,247,841 bytes, and was load-tested in the pod
    # at 12 layers, d_model 1536, vocab 384 during EXP-R2-058. Verified present
    # again before admission.
    #
    # What membership widens, stated rather than discovered later. Three stages
    # accept it and the rest refuse on their own architecture declarations.
    # `paa_census` is what it is here for. `cohort_power` is not a widening but
    # this arm's PREREQUISITE -- evidence-discipline rule 2 forbids scoring an arm
    # whose cohort context-information has not been qualified, which is how
    # dialogpt-small came to be reported unmeasurable rather than failing.
    # `probe_and_erasure` also accepts it and is no part of this track; it is
    # scheduled by naming STAGES, so nothing runs it by accident, and it is
    # recorded here so that a later full-panel campaign finds the fact declared
    # instead of inferring it from an artefact.
    "bygpt5-medium-en",
    # The other two ByGPT5 rungs, admitted 2026-08-21. Their exclusion until
    # today was a `paa_census` refusal applied one level too high, plus a
    # circular prerequisite, and neither survives being written out.
    #
    # The refusal itself is real and is unchanged: `hit@20` has a grid-dependent
    # chance level, so on a 4x6 = 24-head and a 6x12 = 72-head grid it cannot
    # separate a census that retrieves the causally important heads from one that
    # returns the grid. It is declared where it applies --
    # STAGE_CONTRACTS["paa_census"].excluded_arms still names both arms by hand,
    # because no ArmSpec field carries a head count -- so admitting them here
    # does not admit them there and `panel_contract.py --json` still prints the
    # refusal against their names. What was wrong was the level. CAMPAIGN_PANEL
    # is the set of checkpoints a campaign may schedule; every stage intersects
    # it with its own predicate; refusing an arm from the whole campaign because
    # one stage cannot read one statistic on it is the restatement of a stage's
    # panel that this file exists to end.
    #
    # The second half of the old reason was circular. It said the two may not be
    # scored anywhere because their cohort has never been through `cohort_power`
    # -- and `cohort_power` is a campaign stage, so non-membership was the only
    # thing keeping that prerequisite undischarged. It is these arms'
    # PREREQUISITE for exactly the reason it is bygpt5-medium-en's, and
    # membership is what discharges it.
    #
    # The cost of leaving them out was being paid in readings that already
    # exist. `collision_null_census` has scored all three rungs since EXP-R2-155
    # -- their vocabulary collision rate sits among the residue-tokenised protein
    # arms while their modality is text, which is why that stage declares against
    # the pattern set rather than the circuit set -- but only bygpt5-medium-en
    # may carry a verdict, and the audit records the other two as unqualified
    # supporting rungs (§4 bounds) and as unscored on the context-information
    # estimand anywhere, where an absence is not a pass (§5.06(h)).
    #
    # Verified before admission rather than assumed, at float32 on this host's
    # CPU: both load through `arms.load_arm`; their live block count and width
    # match the ArmSpec exactly, at 4 x 1472 / 73,495,680 parameters and
    # 6 x 1536 / 139,218,816; both return logits of width 384 against the
    # 384-symbol vocabulary `budget.arm_power` reads out of `config.vocab_size`;
    # and both carry the `budget` capability every campaign member owes. Their
    # checkpoints are staged on the cluster and run there: EXP-R2-171 took all
    # fifteen panel arms through `collision_null_census` in the pod.
    #
    # What membership widens, stated rather than discovered later: the same three
    # stages as bygpt5-medium-en and no others. `cohort_power` (the prerequisite
    # above), `collision_null_census` (already run, now qualifiable) and
    # `probe_and_erasure`, which admits every arm and writes per-concept refusals
    # into its own output. Every other stage refuses `t5_decoder` on its own
    # module's architecture declaration.
    "bygpt5-small-en",
    "bygpt5-base-en",
)

#: Panel members deliberately outside :data:`CAMPAIGN_PANEL`, with the reason.
#:
#: **Empty, and that is a state rather than an unfilled field.** Every
#: :data:`~src.capability.core.arms.PANEL` member has been a campaign arm since
#: 2026-08-21; the two entries here until then were the narrow ByGPT5 rungs, and
#: the admission comment above says why their reason did not hold at this level.
#:
#: The table stays because :func:`_check_campaign_panel` reads it: an arm added
#: to ``PANEL`` must be admitted above or refused here by name, and an import
#: fails if it is neither. An empty table is therefore the strongest state this
#: declaration has -- nothing is outside the campaign -- and not a gap in it.
PANEL_MEMBERS_NOT_STAGED: dict[str, str] = {}

#: Checkpoints that are staged and load cleanly but are NOT declared in
#: :data:`~src.capability.core.arms.PANEL`, with the measured reason. Recorded here rather
#: than left as an absence, because "we have not got round to it" and "admitting it
#: would corrupt a statistic" are different facts and only one of them is a
#: decision. The two ProGen2 rungs were load-checked on the pod (EXP-R2-068); the
#: EXP-R2-225 second-stage checkpoints carry the reason each was staged without
#: being admitted, and every one of those reasons is a property this repository
#: has read off the checkpoint rather than an intention about it.
#:
#: EXP-R2-225's joint wave is absent from this table because it is absent from
#: :data:`~src.capability.core.arms.STAGED_ARMS`: a joint checkpoint is reached by path
#: rather than declared as an arm, and ``src.capability.core.arms`` records why at the
#: point where its declaration would otherwise sit.
#:
#: This table's keys must be exactly
#: :data:`~src.capability.core.arms.STAGED_ARMS`, so a checkpoint cannot become
#: reachable to a loader without a reason for its non-admission being written
#: here. The test that enforced it, ``tests/test_replaceable_arms.py``, did not
#: survive the test restructure; the requirement holds by inspection, at eight
#: keys on each side, and is unguarded until that check is restored.
STAGED_BUT_NOT_ADMITTED: dict[str, str] = {
    "progen2-large": (
        "2779.4M parameters, 32 blocks of width 2560, loads and runs -- but its "
        "config declares vocab_size 51200 against a 31-token tokenizer, so 51169 "
        "of its logit rows are unreachable. Every statistic this package derives "
        "from config.vocab_size would be computed over a mostly dead alphabet and "
        "would not be comparable with the other ProGen2 arms: the held-out unigram "
        "support, the plug-in entropy, and the rank-(V-1) aperture of L8. It is a "
        "natural experiment on whether that aperture tracks the output MATRIX or "
        "the reachable SYMBOLS -- same lineage, same effective alphabet, a 1600x "
        "wider head -- and that is worth its own gated design, not a quiet "
        "admission into a panel whose other arms read vocab_size as the alphabet"
    ),
    "progen2-xlarge": (
        "6443.6M parameters, 32 blocks of width 4096, loads and runs a forward "
        "pass returning logits of width 32 -- but its config carries no "
        "vocab_size attribute at all, only vocab_size_emb and vocab_size_lm_head. "
        "budget.arm_power reads config.vocab_size directly, so admitting it would "
        "raise AttributeError inside cohort_power rather than produce a wrong "
        "number. Admissible once the panel reads a declared alphabet size instead "
        "of trusting a config key that two checkpoints in this lineage spell "
        "differently"
    ),
    # ---- EXP-R2-225 second stage. Staged for a descriptive read;
    # none is qualified and none is admitted.
    "qwen2.5-7b": (
        "28 blocks of width 3584 over a vocab_size of 152064, the same "
        "architecture, tokenizer and pretraining mixture as the panel member "
        "qwen2.5-0.5b. Nothing about it would corrupt a panel statistic; it is "
        "out because panel membership is a campaign obligation on every stage, "
        "and a descriptive second-stage scale reading needs a staged rung rather "
        "than a fifteenth arm every campaign must then schedule"
    ),
    "qwen2.5-32b": (
        "as qwen2.5-7b, at 64 blocks of width 5120 over the same 152064 "
        "vocab_size. Additionally unmeasured: no device-memory measurement exists "
        "for it, so it is a single-card H200 candidate and not a checkpoint this "
        "repository has run"
    ),
    "proteinglm-7b-clm": (
        "36 blocks of width 4096 over a padded vocab_size of 128. AutoModelForCausalLM "
        "from the original modeling file remains unloadable: modeling_proteinglm.py "
        "line 15 reads 'import torch, deepspeed' and Transformers' AST import check "
        "fires on that name. Budget serving loads ProteinGLMForCasualLM from a "
        "derived inference-only copy and does not vendor the checkpoint. Native "
        "rendering is gmask_sop_eos: <gmask><sop><eos> at ids [29, 32, 34] plus AA20, "
        "no trailing EOS, scored over residues 2..L as continuation NLL. The 7B probe "
        "read 1.1277 nats/residue native against 2.8974 shuffled and 16.9930 "
        "unprefixed; that is a future reproduction target, not a tiny-test result. "
        "capabilities={budget} is an implementation door, not 7B numerical "
        "qualification. Hidden states remain [seq, batch, hidden] beside logits "
        "[batch, seq, vocab]; output_attentions is dead; attn_implementation is inert"
    ),
    "rita-xl": (
        "24 blocks of width 2048 over a vocab_size of 26, native rendering raw. "
        "Out of the panel because its rita architecture is declared in none of "
        "the tables the interpretability families resolve through, so it could "
        "carry only budget, and because no padding id can be established for it "
        "at all: its tokenizer declares no pad and no eos token "
        "(special_tokens_map.json is {} while tokenizer.json defines <PAD> at 1 "
        "and <EOS> at 2), its config declares no pad_token_id for load_arm_spec's "
        "config-declared step to adopt, and the eos_token_id 50256 it does "
        "declare is not an id its 26-symbol vocabulary contains. Every batch is "
        "refused"
    ),
    # ---- budget-only candidates. Staged for the shared measurement premises'
    # loglikelihood path; not campaign panel members and not experiment admission.
    "qwen3-8b-base": (
        "36 blocks of width 4096 over a vocab_size of 151936, Qwen3 dense "
        "pretrained Base -- not the Instruct sibling Qwen3-8B. Out of the panel "
        "because candidate serving is a third opt-in door, not a campaign "
        "obligation: only budget is granted, QK-Norm/GQA/rotary intervention "
        "families are not, and an implemented interface is not an 8-block "
        "identification interval or an ADMITTED digest"
    ),
    "protgpt3-1.3b": (
        "17 blocks of width 1024 Mixtral-style MoE over a vocab_size of 31. Out "
        "of the panel because candidate serving is not panel admission: only "
        "budget is granted, the weight licence is unknown, and a residue-level "
        "tokenizer with a 31-wide head is not a ProteinGym scoring protocol"
    ),
}


def _check_campaign_panel() -> None:
    unknown = [name for name in CAMPAIGN_PANEL if name not in PANEL]
    if unknown:
        raise AssertionError(
            f"CAMPAIGN_PANEL names arms that are not in src.capability.core.arms.PANEL: {unknown}"
        )
    if len(set(CAMPAIGN_PANEL)) != len(CAMPAIGN_PANEL):
        raise AssertionError("CAMPAIGN_PANEL repeats an arm")
    undecided = sorted(
        name
        for name in PANEL
        if name not in CAMPAIGN_PANEL and name not in PANEL_MEMBERS_NOT_STAGED
    )
    if undecided:
        raise AssertionError(
            f"panel members {undecided} are neither in CAMPAIGN_PANEL nor given a "
            "reason in PANEL_MEMBERS_NOT_STAGED; a new arm must be admitted or "
            "excluded explicitly, never by omission"
        )


_check_campaign_panel()


# ------------------------------------------------------------- stage contracts


@dataclass(frozen=True)
class ProteinBand:
    """One residue band a stage draws a cohort on, and the argument that sets it.

    ``argument_prefix`` names the argparse option pair the stage carries, so a
    reader (and ``../d2/tests/test_farband_estimand.py``) can read the band back out of
    the entry point instead of trusting this table.

    Deliberately the same record, spelled the same way, as
    ``../d2/scripts/capability/mechanisms/gap_contract.py::ProteinBand``. The TG campaign found
    three live undeclared bands inside the mechanism built to stop undeclared
    bands, and fixed it with this shape; the transfer campaign's own contract had
    room for exactly one band per stage, which is how ``circuit_primitives`` came
    to draw two and declare neither.
    """

    argument_prefix: str
    residues: tuple[int, int]
    reason: str = ""

    def __post_init__(self) -> None:
        low, high = self.residues
        if low < 1 or high < low:
            raise AssertionError(f"invalid protein band {self.residues} on {self.argument_prefix}")

    @property
    def matches_qualifying_stage(self) -> bool:
        return tuple(self.residues) == QUALIFYING_PROTEIN_BAND


@dataclass(frozen=True)
class StageContract:
    """What one campaign stage requires of an arm, and how it is dispatched.

    ``scope`` is the stage's *panel contract*, made explicit because it was
    previously only inferable from how the worker happened to call the script,
    and getting it wrong cost a run of EXP-R2-060:

    ``per_arm``
        one process per arm; the arm list is a set of independent measurements.
    ``panel_wide``
        one process for the whole arm list, which writes a combined artefact.
        Splitting it produces incomplete panels that overwrite each other.
    ``control_anchored``
        one process whose arm list must contain exactly one text arm, because
        the verdict it produces is anchored on that control (evidence discipline
        rule 1). ``estimand_power.py recommend`` enforces this literally.
    ``armless``
        the stage takes no arm argument at all.
    """

    name: str
    entry_point: str
    scope: str
    capabilities: frozenset[str] = frozenset()
    architectures: frozenset[str] | None = None
    architecture_source: str = ""
    modalities: frozenset[str] | None = None
    tokenisations: frozenset[str] | None = None
    tokenisation_reason: str = ""
    #: The ``ArmSpec.input_format`` values this stage's entry point can render
    #: into its own cohort. Separate from :attr:`tokenisations` because the two
    #: refuse different things: a tokenisation refusal is about how residues map
    #: to tokens, an input-format refusal is about what the *rendering* wraps
    #: them in. ``paa_census`` needs the second and not the first -- ZymCTRL is
    #: residue-tokenised like the ProGen2 arms it runs beside, and what excludes
    #: it is the conditioning prefix its rendering carries.
    input_formats: frozenset[str] | None = None
    input_format_reason: str = ""
    declared_arms: tuple[str, ...] | None = None
    declared_arms_source: str = ""
    #: Arms this stage refuses for a reason none of the declared properties above
    #: can express, keyed to that reason.
    #:
    #: Distinct from :attr:`declared_arms`, which is an allow-list that has to
    #: restate a whole panel and goes stale the moment one is added. This is a
    #: deny-list of named exceptions, and it exists because ``paa_census`` has one
    #: that is genuine and unexpressible: its retrieval statistic ``hit@k`` is
    #: comparable only between arms with the SAME number of heads, and an
    #: ``ArmSpec`` declares depth and width but not a head count, so no property in
    #: this file can decide it. A reason is mandatory -- see
    #: :func:`_check_stage_contracts` -- because an unexplained name in a deny-list
    #: is indistinguishable from a typo.
    excluded_arms: dict[str, str] = field(default_factory=dict)
    #: EVERY protein residue band this stage draws a cohort on, as the stage's own
    #: argparse defaults set it. See :data:`QUALIFYING_PROTEIN_BAND`.
    #:
    #: A tuple rather than a single band because stages draw more than one:
    #: ``circuit_primitives`` draws its analysis, attribution and patching cohort
    #: at 600-1000 residues and its two natural-repeat cohorts at 200-800, and the
    #: field it had room for could only ever have declared one of them. It
    #: declared neither, and ``matches_qualifying_stage`` therefore read ``null``
    #: -- identical to a stage that draws no protein cohort at all.
    #:
    #: Empty means the stage declares that it draws no protein cohort with a single
    #: residue band; :attr:`protein_band_reason` then says which of the two reasons
    #: applies, and :func:`_check_stage_contracts` requires it.
    protein_bands: tuple[ProteinBand, ...] = ()
    #: How this stage's bands relate to :data:`QUALIFYING_PROTEIN_BAND`, or why
    #: there are none. Required either way.
    protein_band_reason: str = ""
    notes: str = ""

    @property
    def protein_band(self) -> tuple[int, int] | None:
        """The single band this stage draws on, or ``None`` if it draws 0 or 2+.

        A stage on several bands has no one band, and answering with the first
        would be the same defect one level down. Readers that need the complete
        answer take :attr:`protein_bands`.
        """

        if len(self.protein_bands) == 1:
            return self.protein_bands[0].residues
        return None


SCOPES = ("per_arm", "panel_wide", "control_anchored", "armless")

#: The band ``cohort_power.py`` qualifies an arm on. Every stage that draws a
#: protein cohort at a *different* band is measuring a different population from
#: the one the qualification verdict covers, so the difference is declared per
#: stage below and written into each artefact rather than left to be discovered
#: by comparing four argparse defaults.
QUALIFYING_PROTEIN_BAND = (64, 246)

#: The PAA instance-pool width ``paa_census`` is scheduled at, and the width its
#: eligible arm list below is declared *against*.
#:
#: This is not a scale knob and it does not belong in ``ARGS_PAA_CENSUS``. It is
#: the parameter that decides whether an arm can enter the stage at all:
#: ``prediction_addressed.tokenised_rows`` drops every cohort record that does
#: not reach exactly ``width`` tokens, so an arm whose tokeniser cannot put a
#: full-width row inside the census band raises ``no cohort record reached
#: <width> tokens`` -- after the checkpoint is on the GPU.
#:
#: 192 rather than ``paa_census.py``'s own default of 512 because the matched
#: pair (§2's only modality-identifying comparison) exists at exactly one of the
#: two. Measured, not argued: in the *unchanged* 520-800 census band ProtGPT2
#: admits 400/400 rows at width 128, 320-355/400 at 192, 65/400 at 256 and **0 at
#: width >= 320**, while gpt2-large admits 400/400 at every width (EXP-R2-082).
#: The alternative route -- keeping width 512 and raising the band floor to
#: ~1550 -- shares not one record with the band L22's protein arms were measured
#: on and moves context information by 1.75-2.31 nats against L13's catalogued
#: 1.01, so the width is what moves and the band is what is preserved. The
#: attainability control was re-established at this width and passed against
#: gpt2-large's own induction band: exhaustive matched rho +0.4515 [+0.401,
#: +0.498] inside +0.428 to +0.535 (EXP-R2-087).
#:
#: Declared here rather than written into ``h200_worker.sh`` because the worker
#: passing one width and this file admitting arms at another is precisely the
#: kind of divergence that would make the refusal reasons below false.
PAA_CENSUS_WIDTH = 192

STAGE_CONTRACTS: dict[str, StageContract] = {"cohort_power": StageContract(
        name="cohort_power",
        entry_point="stages/cohort_power.py",
        scope="panel_wide",
        capabilities=frozenset({"budget"}),
        protein_bands=(ProteinBand("res", (64, 246)),),
        protein_band_reason="this stage defines the qualifying band",
        notes=(
            "scores every arm passed to one invocation in one process and writes "
            "one combined report, so it cannot be dispatched per arm; the worker "
            "splits it by vocabulary regime instead (see COHORT_POWER_ITEMS)"
        ),
    )}

#: Campaign stage order. The worker's tier structure depends on it: cohort_power
#: qualifies the cohort every later stage draws on, and estimand_power's
#: recommendation reads pathway_budget's regime.
STAGE_ORDER: tuple[str, ...] = tuple(STAGE_CONTRACTS)


def _check_stage_contracts() -> None:
    for stage, contract in STAGE_CONTRACTS.items():
        if contract.name != stage:
            raise AssertionError(f"stage {stage!r} declares name {contract.name!r}")
        if contract.scope not in SCOPES:
            raise AssertionError(f"stage {stage!r} declares unknown scope {contract.scope!r}")
        if contract.architectures is not None and not contract.architecture_source:
            raise AssertionError(
                f"stage {stage!r} restricts architectures without naming the module "
                "declaration it mirrors"
            )
        if contract.input_formats is not None:
            if not contract.input_format_reason:
                raise AssertionError(
                    f"stage {stage!r} restricts input formats without saying why; a "
                    "refusal an operator cannot read is indistinguishable from an arm "
                    "nobody thought about"
                )
            # A typo in an allow-list refuses silently and refuses everything, which
            # is the one failure mode an allow-list has that a deny-list does not.
            declared = {spec.input_format for spec in PANEL.values()}
            unknown_formats = sorted(contract.input_formats - declared)
            if unknown_formats:
                raise AssertionError(
                    f"stage {stage!r} admits input formats {unknown_formats} that no "
                    f"member of src.capability.core.arms.PANEL declares; declared formats are "
                    f"{sorted(declared)}"
                )
        if contract.declared_arms is not None:
            unknown = [a for a in contract.declared_arms if a not in PANEL]
            if unknown:
                raise AssertionError(f"stage {stage!r} declares unknown arms {unknown}")
        for arm, reason in contract.excluded_arms.items():
            if arm not in PANEL:
                raise AssertionError(
                    f"stage {stage!r} excludes {arm!r}, which is not in "
                    "src.capability.core.arms.PANEL; a deny-list entry that names nothing "
                    "refuses nothing and reads exactly like one that works"
                )
            if not reason:
                raise AssertionError(
                    f"stage {stage!r} excludes {arm!r} without saying why. This "
                    "deny-list exists for refusals no declared property can express, "
                    "so the reason is the whole declaration"
                )
        prefixes = [band.argument_prefix for band in contract.protein_bands]
        if len(set(prefixes)) != len(prefixes):
            raise AssertionError(
                f"stage {stage!r} declares two protein bands on one argument pair "
                f"{prefixes}"
            )
        # An absent band has to be a decision. It was an omission for
        # circuit_primitives, which draws protein cohorts on two bands and declared
        # neither, and a null `matches_qualifying_stage` read exactly the same there
        # as it does for a stage that draws no protein cohort at all.
        if not contract.protein_band_reason:
            raise AssertionError(
                f"stage {stage!r} does not say how its protein cohort relates to the "
                f"qualifying band {QUALIFYING_PROTEIN_BAND}. A stage that draws no "
                "protein cohort, and one whose bands nobody has looked up, must not "
                "be spelled the same way"
            )
_check_stage_contracts()


# ------------------------------------------------------------------ predicate


@dataclass(frozen=True)
class Eligibility:
    """Whether one arm may enter one stage, and the declaration that decides it."""

    stage: str
    arm: str
    can_run: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.can_run


def require_known(stage: str, arm: str) -> StageContract:
    if stage not in STAGE_CONTRACTS:
        raise KeyError(f"unknown stage {stage!r}; stages are {list(STAGE_CONTRACTS)}")
    if arm not in PANEL:
        raise KeyError(f"unknown arm {arm!r}; panel is {sorted(PANEL)}")
    return STAGE_CONTRACTS[stage]


def arm_can_run(stage: str, arm: str) -> Eligibility:
    """Can ``stage``'s entry point produce a commensurate number for ``arm``?

    A ``False`` is always accompanied by the declaration that refuses, so an
    operator reading a skip line can tell a capability decision (the panel's) from
    a module limitation (the measuring code's) from a staging fact.

    Unknown stage or unknown arm raises rather than returning ``False``: "this
    arm cannot run" and "nobody has heard of this arm" are different facts and
    collapsing them is how a typo becomes a silently narrower panel.
    """

    contract = require_known(stage, arm)
    spec = PANEL[arm]

    if contract.scope == "armless":
        return Eligibility(
            stage,
            arm,
            False,
            f"{contract.entry_point} takes no arm argument; it is not dispatched per arm",
        )

    if contract.declared_arms is not None and arm not in contract.declared_arms:
        return Eligibility(
            stage,
            arm,
            False,
            f"not in {contract.declared_arms_source}, which is this stage's own "
            f"declaration of the arms it measures ({list(contract.declared_arms)})",
        )

    # Before the property gates, because a named exception is a decision about
    # this arm and the property gates would otherwise report a reason that is true
    # of the arm and not the reason it is refused.
    if arm in contract.excluded_arms:
        return Eligibility(stage, arm, False, contract.excluded_arms[arm])

    missing = sorted(contract.capabilities - spec.capabilities)
    if missing:
        return Eligibility(
            stage,
            arm,
            False,
            f"ArmSpec.capabilities does not grant {missing}; declared capabilities "
            f"are {sorted(spec.capabilities)}",
        )

    if contract.architectures is not None and spec.architecture not in contract.architectures:
        return Eligibility(
            stage,
            arm,
            False,
            f"architecture {spec.architecture!r} is not in "
            f"{contract.architecture_source} = {sorted(contract.architectures)}, so the "
            "measuring module has no code path for this arm's module layout",
        )

    if contract.modalities is not None and spec.modality not in contract.modalities:
        return Eligibility(
            stage,
            arm,
            False,
            f"modality {spec.modality!r}; this stage measures "
            f"{sorted(contract.modalities)} arms only",
        )

    if contract.tokenisations is not None and spec.tokenisation not in contract.tokenisations:
        return Eligibility(
            stage,
            arm,
            False,
            f"tokenisation {spec.tokenisation!r} is not in "
            f"{sorted(contract.tokenisations)}: {contract.tokenisation_reason}",
        )

    if contract.input_formats is not None and spec.input_format not in contract.input_formats:
        return Eligibility(
            stage,
            arm,
            False,
            f"input_format {spec.input_format!r} is not in "
            f"{sorted(contract.input_formats)}: {contract.input_format_reason}",
        )

    return Eligibility(stage, arm, True)


def stage_arms(
    stage: str, requested: list[str] | tuple[str, ...] | None = None
) -> tuple[list[str], list[Eligibility]]:
    """``(eligible, refused)`` for one stage over ``requested`` (default: the campaign panel).

    Order follows ``requested``, so a caller that cares about invocation order
    keeps it. ``refused`` carries one :class:`Eligibility` per skipped arm, each
    with its reason, for the caller to log -- a skip that is not logged is
    indistinguishable from an arm that was never asked for.
    """

    names = list(CAMPAIGN_PANEL if requested is None else requested)
    eligible: list[str] = []
    refused: list[Eligibility] = []
    for arm in names:
        verdict = arm_can_run(stage, arm)
        if verdict.can_run:
            eligible.append(arm)
        else:
            refused.append(verdict)
    return eligible, refused


def stage_contract_record(stage: str, arms: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """The block a stage writes into its own artefact to declare its panel and band.

    Two facts that were previously only recoverable by comparing argparse
    defaults across four files:

    ``arm_selection``
        which arms ran, which panel members did not, and why. A stage that
        measures a subset of the panel and does not say so is the L18 shape.
        ``eligible_for_this_stage`` is resolved over :data:`CAMPAIGN_PANEL`, so an
        arm measured from outside that panel -- a deliberate per-arm run of a
        checkpoint no campaign schedules -- would otherwise read as an arm the
        stage refuses. ``measured_outside_campaign_panel`` answers that directly,
        with the stage's own verdict on the arm beside the reason it is not in the
        campaign. It is empty on every artefact while
        :data:`PANEL_MEMBERS_NOT_STAGED` is empty, which is the current state;
        the field is what keeps the next non-campaign arm from reading as a
        contradiction rather than as a declared direct invocation.
    ``cohort_band``
        the protein residue band this stage draws on, beside the band
        ``cohort_power.py`` *qualified* the arms on, and a flag for whether
        they agree. They do not agree for ``lens_family``, and EXP-R2-060
        measured protein cohort-block sensitivity at 0.16-0.60 nats, so the
        difference is worth carrying with the number.
    """

    contract = STAGE_CONTRACTS[stage]
    eligible, refused = stage_arms(stage)
    measured = list(arms)
    # Two different reasons an arm is absent, kept apart because they mean
    # different things to a reader. "Refused" is a property of the arm and the
    # module -- it could not have been measured. "Eligible but not asked for" is a
    # property of *this invocation* -- the operator, a default or a worker scoped
    # the run, and that is the narrowing L18 records.
    return {
        "stage": stage,
        "scope": contract.scope,
        "arm_selection": {
            "measured": measured,
            "campaign_panel": list(CAMPAIGN_PANEL),
            "eligible_for_this_stage": eligible,
            "measured_outside_campaign_panel": {
                name: {
                    "eligible_for_this_stage": arm_can_run(stage, name).can_run,
                    "eligibility_reason": arm_can_run(stage, name).reason,
                    "not_in_campaign_panel_because": PANEL_MEMBERS_NOT_STAGED.get(name),
                }
                for name in measured
                if name not in CAMPAIGN_PANEL
            },
            "eligible_but_not_measured": [
                name for name in eligible if name not in measured
            ],
            "not_measured": {
                **{
                    name: "eligible for this stage but not in this invocation's arm list"
                    for name in eligible
                    if name not in measured
                },
                **{v.arm: v.reason for v in refused if v.arm not in measured},
            },
        },
        "cohort_band": cohort_band_record(contract),
    }


def cohort_band_record(contract: StageContract) -> dict[str, Any]:
    """Every protein band a stage draws on, beside the band it was qualified on.

    ``protein_residue_bands`` is the complete answer and is the field to read.
    ``protein_residues`` is the single band for the common case of a stage that
    draws exactly one, kept because that is what a reader expects and what the
    frozen artefacts of the single-band stages carry; it is ``null`` for a stage
    that draws several, whose bands are then all in the list beside it.

    ``matches_qualifying_stage`` is ``null`` if and only if the stage declares no
    protein band at all, which is now a declared fact carrying a ``reason`` rather
    than the absence of one. Otherwise it is true only when every band this stage
    draws is the qualifying band -- a stage that draws a second cohort somewhere
    else does not match it however its first band reads.
    """

    bands = contract.protein_bands
    return {
        "protein_residues": (
            None if contract.protein_band is None else list(contract.protein_band)
        ),
        "protein_residue_bands": [
            {
                "argument_prefix": band.argument_prefix,
                "protein_residues": list(band.residues),
                "matches_qualifying_stage": band.matches_qualifying_stage,
                "reason": band.reason or None,
            }
            for band in bands
        ],
        "qualifying_stage_protein_residues": list(QUALIFYING_PROTEIN_BAND),
        "matches_qualifying_stage": (
            None if not bands else all(band.matches_qualifying_stage for band in bands)
        ),
        "reason": contract.protein_band_reason or None,
    }


# --------------------------------------------------- cohort_power item dispatch


@dataclass(frozen=True)
class CohortPowerItem:
    """One ``cohort_power.py`` invocation, and why it is separate from the rest.

    ``01`` writes its combined report only after its whole per-arm loop finishes,
    so one arm raising loses every arm already computed in that invocation. The
    split is therefore by the properties that decide whether an arm *can* be in
    the same process, and each reason is measured rather than assumed.
    """

    item: str
    arms: tuple[str, ...]
    extra_args: tuple[str, ...]
    cohort_name: str | None
    reason: str


COHORT_POWER_ITEM_RULES: tuple[tuple[str, str], ...] = (
    (
        "text",
        "text arms share one OpenWebText cohort. --skip-truncation because the "
        "invocation holds vocabularies far above the 1024-piece limit "
        "budget.truncation_curve can compute on this transformers build, and the "
        "flag is per invocation rather than per arm. That is a cost the three "
        "byte-level rungs pay for sharing the process: their 384-symbol "
        "vocabularies are inside the limit and their curve is computable, and it "
        "is declared here rather than left to be inferred from a missing field",
    ),
    (
        "protein_large_vocab",
        "protein arms with vocabulary > 1024: the truncation curve is not computable "
        "on this transformers build, so --skip-truncation",
    ),
    (
        "protein_small_vocab",
        "EC-conditioned residue-level arms: --with-ec, and the truncation curve is "
        "computable so it must NOT be skipped",
    ),
    (
        "protein_default_dtype",
        "residue-level protein arms taking the script's declared default dtype; no "
        "precision override is inferred from a different checkpoint. Named for the "
        "rule rather than for a member: the old name described its only occupant, "
        "and admitting ProGen2-small made that name cover one of two arms. Both "
        "occupants' checkpoints declare ``torch_dtype`` float32 -- progen2-small's "
        "and progen2-base's configs, as progen2-medium's does; progen2-large and "
        "progen2-xlarge declare float16 -- so what this item chooses is bfloat16 for "
        "two of the three fp32-declared ProGen2 rungs, and the choice costs this "
        "much. Measured on the staged checkpoints over 200 Swiss-Prot records of "
        "64-246 residues at draw seed 20260728 (32,666 scored residues, this default "
        "against float32, agreement taken over the same positions): progen2-small "
        "1.965933 nats/residue at bfloat16 against 1.957239 at float32, +8.694e-03, "
        "KL 1.125e-02, argmax agreement 0.7975; progen2-base 1.579147 against "
        "1.574456, +4.691e-03, KL 3.456e-03, argmax agreement 0.9109. Each delta "
        "exceeds the upper end of the 1.4e-04 to 2.9e-03 nat host divergence the "
        "L20-vs-H200 cross-check prices, the smaller by 1.6x and the larger by "
        "3.0x, so neither is inside the spread that cross-check treats as "
        "host-bound. This item is where that cost is taken: the medium rung is kept "
        "out of it by the item below, and moving either arm across that boundary is "
        "a scoring decision this campaign has not taken",
    ),
    (
        "protein_progen2_medium",
        "ProGen2-medium is isolated at --dtype float32: its "
        "nll_reduction_shortest_to_longest_nats moved 0.6266 -> 0.7293 (+16%) under "
        "bfloat16 in the L20-vs-H200 cross-check and collapsed to 2.6e-7 in float32; "
        "--dtype governs model loading so it cannot be set per arm within one process. "
        "THIS SPLIT IS WHAT MAKES THE progen2-base/progen2-medium CONTRAST UNPAIRED "
        "(L38), and the two ways of removing it are both worse than the split. "
        "Returning ProGen2-medium to the default dtype undoes a measured repair. "
        "Promoting ProGen2-base to float32 would infer a precision override from a "
        "different checkpoint, which protein_default_dtype above refuses on principle "
        "and for which no measurement exists: the bfloat16 instability was measured on "
        "ProGen2-medium alone, and establishing it for ProGen2-base needs a two-host "
        "cross-check nobody has run. What the split does NOT require is the unpaired "
        "reading. The two items draw byte-identical cohorts and byte-identical held-out "
        "references -- Cohort.digest hashes records and never the name, and on all "
        "eight blocks of EXP-R2-216 the two --cohort-name values carry the same digest "
        "pair -- so the near-duplicate groups a bootstrap resamples are the same units, "
        "whatever precision each arm's NLL was computed at. The contrast is unpaired "
        "only because context_information_bootstrap.py keys pairing on the sidecar, "
        "one cohort_power.py invocation, rather than on the cohort digest the "
        "sidecar already carries. That is where L38 is repairable; it is not repairable "
        "here without changing what an arm is scored at",
    ),
)


def cohort_power_items(requested: list[str] | tuple[str, ...] | None = None) -> list[CohortPowerItem]:
    """The ``cohort_power.py`` invocations covering ``requested``.

    Derived from ``PANEL`` -- modality, tokenisation and the declared EC input
    format -- rather than from arm names, which is what the worker used to do in
    bash. Every protein item gets a distinct ``--cohort-name`` because two
    protein items can otherwise produce byte-identical cohorts under the shared
    default name and collide on the same output filename.
    """

    eligible, _ = stage_arms("cohort_power", requested)
    buckets: dict[str, list[str]] = {name: [] for name, _ in COHORT_POWER_ITEM_RULES}
    for arm in eligible:
        spec = PANEL[arm]
        if spec.modality == "text":
            buckets["text"].append(arm)
        elif spec.input_format == "ec_conditioned":
            buckets["protein_small_vocab"].append(arm)
        elif vocabulary_regime(arm) == "large":
            buckets["protein_large_vocab"].append(arm)
        elif arm == "progen2-medium":
            buckets["protein_progen2_medium"].append(arm)
        else:
            buckets["protein_default_dtype"].append(arm)

    extra = {
        "text": ("--skip-truncation",),
        "protein_large_vocab": ("--skip-truncation",),
        "protein_small_vocab": ("--with-ec",),
        "protein_default_dtype": (),
        "protein_progen2_medium": ("--dtype", "float32"),
    }
    cohort_names = {
        "text": None,
        "protein_large_vocab": "swissprot_large_vocab",
        "protein_small_vocab": "swissprot_small_vocab",
        "protein_default_dtype": "swissprot_default_dtype",
        "protein_progen2_medium": "swissprot_progen2_medium_f32",
    }
    items: list[CohortPowerItem] = []
    for item, reason in COHORT_POWER_ITEM_RULES:
        if not buckets[item]:
            continue
        items.append(
            CohortPowerItem(
                item=item,
                arms=tuple(buckets[item]),
                extra_args=extra[item],
                cohort_name=cohort_names[item],
                reason=reason,
            )
        )
    return items


#: Vocabulary regime per panel arm, keyed off the declared tokenisation family
#: rather than read from a config file, because the campaign panel's regimes are a
#: declared design property (the 1600-fold aperture spread of L8) and a scheduler
#: must not need a checkpoint on disk to plan a run.
_LARGE_VOCAB_TOKENISATIONS = frozenset({"bpe", "multi_residue_bpe"})

#: The output width above which ``budget.truncation_curve`` refuses to compute a
#: curve at all, on a build of ``transformers`` whose forward takes no
#: ``logits_to_keep``. It is the model's own guard, restated here as the number
#: this scheduler buckets on rather than as a second threshold: the untrimmed
#: path is a different unembedding kernel and would not reproduce a curve
#: measured on the trimmed one.
TRUNCATION_CURVE_LOGIT_LIMIT = 1024

#: Released output-head width per staged checkpoint, which is what the guard
#: above reads and what a panel arm's tokenisation family is only a *proxy* for.
#:
#: Declared for the staged checkpoints specifically because the proxy fails on
#: them: ``progen2-large`` is residue-tokenised -- ``small`` by every panel rule
#: here -- over a 51200-column head, and that is precisely the arm the rule was
#: applied to by hand during the EXP-R2-224 stage-01 dispatch after
#: ``budget.truncation_curve`` refused its three cells. Every entry is the width
#: the checkpoint's own ``config.json`` declares (``vocab_size``, and
#: ``vocab_size_lm_head`` on ``progen2-xlarge``, which carries no ``vocab_size``
#: key at all), so ``tests/test_second_stage_capability.py`` checks each against
#: the file rather than against this table.
#:
#: ``progen2-large``'s 51200 is also the frozen ``REQUIRED_LIVE_WIDTH`` of
#: ``scale_interface_qualification.py``. The two are deliberately not shared:
#: that one is an EXP-R2-224 experimental requirement, frozen before the
#: campaign and checked against the *live* head as the artefact of an all-or-stop
#: gate, and this one is a scheduling declaration read before any checkpoint is
#: opened. Binding a scheduler to a frozen experiment's constant would make
#: planning depend on that experiment's freeze.
STAGED_OUTPUT_LOGIT_WIDTH: dict[str, int] = {
    "progen2-large": 51200,
    "progen2-xlarge": 32,
    "qwen2.5-7b": 152064,
    "qwen2.5-32b": 152064,
    "proteinglm-7b-clm": 128,
    "rita-xl": 26,
    "qwen3-8b-base": 151936,
    "protgpt3-1.3b": 31,
}


def _check_staged_output_widths() -> None:
    """Every staged checkpoint declares a width, so none falls outside the rule.

    The defect this closes is not a wrong width, it is an *absent* one: an arm
    with no declared width would silently take the small-vocabulary branch and be
    scheduled without ``--skip-truncation``, which is exactly how the three
    ``progen2-large`` cells were refused with exit 1 having touched no card.
    """

    missing = sorted(set(STAGED_ARMS) - set(STAGED_OUTPUT_LOGIT_WIDTH))
    if missing:
        raise AssertionError(
            f"staged checkpoints {missing} declare no output-head width, so the "
            "truncation-curve rule cannot be applied to them and they would be "
            "scheduled as if their heads were narrow"
        )
    unknown = sorted(set(STAGED_OUTPUT_LOGIT_WIDTH) - set(STAGED_ARMS))
    if unknown:
        raise AssertionError(
            f"{unknown} declare an output-head width but are not staged arms"
        )


_check_staged_output_widths()


def vocabulary_regime(arm: str) -> str:
    """``"large"`` when this arm's declared output vocabulary exceeds the limit.

    One rule, two declared sources, and the second exists because the first is a
    proxy. A panel arm's regime follows its declared tokenisation family, which is
    a design property of the campaign panel and needs no checkpoint on disk. A
    staged checkpoint declares its released head width directly, because
    tokenisation does not predict it there.
    """

    if arm in STAGED_OUTPUT_LOGIT_WIDTH:
        return (
            "large"
            if STAGED_OUTPUT_LOGIT_WIDTH[arm] > TRUNCATION_CURVE_LOGIT_LIMIT
            else "small"
        )
    return "large" if PANEL[arm].tokenisation in _LARGE_VOCAB_TOKENISATIONS else "small"


def skips_truncation_curve(arm: str) -> bool:
    """Does this arm need ``--skip-truncation`` on ``cohort_power.py``?

    The rule ``COHORT_POWER_ITEM_RULES`` already states for panel members --
    a vocabulary above the limit makes the curve uncomputable -- asked of any
    arm this repository declares, panel member or staged checkpoint. It is the
    same question either way, and answering it in one place is what turns the
    EXP-R2-224 dispatch's hand-applied flag into a scheduling rule.
    """

    return vocabulary_regime(arm) == "large"


def staged_cohort_power_args(arm: str) -> tuple[str, ...]:
    """The ``cohort_power.py`` flags a staged checkpoint takes from this rule.

    Panel members get theirs through :func:`cohort_power_items`, which buckets
    whole invocations. A staged rung is dispatched one arm at a time by a
    campaign manifest rather than by that bucketing, so the flags it owes are
    answered per arm here and land in :func:`contract_payload`.
    """

    if arm not in STAGED_OUTPUT_LOGIT_WIDTH:
        raise KeyError(
            f"{arm!r} is not a staged checkpoint; panel members take their "
            "cohort_power flags from cohort_power_items()"
        )
    return ("--skip-truncation",) if skips_truncation_curve(arm) else ()


# ------------------------------------------------------- per-arm data locations

#: Environment variable that relocates each arm's checkpoint, resolved from how
#: ``arms.PANEL`` *builds* the path rather than from the arm's name.
#:
#: The worker used to answer this with ``case "$1" in gpt2-large) ...`` and a
#: modality fallback, and got it wrong for six of seven text arms until
#: 2026-07-29: an arm addressed beneath ``TEXT_MODEL_BASE_DIR`` had its
#: preflight check ``MODEL_ROOT`` instead, so a genuinely missing
#: checkpoint reached ``load_arm`` rather than being reported as a skip.
#:
#: The comparison below is invariant to which host it runs on, because every
#: ``ArmSpec.path`` is *constructed* from one of these three constants: gpt2-large
#: is declared as ``TEXT_MODEL_ROOT`` itself, the protein arms as
#: ``MODEL_ROOT / name`` and the remaining text arms as ``TEXT_MODEL_BASE / name``.
#: Re-pointing any of the three environment variables moves the constant and the
#: arm's path together, so the mapping this produces on the controller host is the
#: mapping the pod re-derives under ``--verify``.
MODEL_PATH_VARIABLES = frozenset(
    {"TEXT_MODEL_DIR", "MODEL_ROOT", "TEXT_MODEL_BASE_DIR"}
)


def model_variable(arm: str) -> str:
    """The declared variable, read from the panel rather than inferred from paths.

    This used to compare the resolved ``ArmSpec.path`` against the three
    constants, and claimed to be host-invariant on the grounds that re-pointing a
    variable moves the constant and the arm's path together. That reasoning holds
    only while the three constants resolve to *distinct* directories. They do not
    on the H200 pod: every checkpoint lives in one GPFS directory, so
    ``h200_env.sh`` sets ``TEXT_MODEL_BASE_DIR="${MODEL_ROOT}"``,
    the ``path.parent == MODEL_ROOT`` branch matched first, and six text arms
    classified as protein-root arms. The rendered contract therefore disagreed
    with the live panel *inside the pod and nowhere else*, which is why the
    worker's own re-derivation refused the campaign before any GPU was scheduled.

    The variable is now declared beside the path it builds, so an alias cannot
    change the answer.
    """

    variable = PANEL[arm].path_variable
    if variable not in MODEL_PATH_VARIABLES:
        raise AssertionError(
            f"{arm}: declares path_variable {variable!r}, which is not one of "
            f"{sorted(MODEL_PATH_VARIABLES)}, so no environment variable relocates "
            "it and the worker cannot preflight it"
        )
    return variable


def _check_declared_paths_match_their_variables() -> None:
    """The declaration and the construction must agree where they can be compared.

    A declared variable that no longer matches how the path is built would be a
    silent lie, so it is checked -- but only when the constants are distinct,
    because when two of them alias there is nothing to check and the declaration
    is the only thing that carries the answer. That is precisely the situation
    this field exists for.
    """

    roots = {
        "TEXT_MODEL_DIR": TEXT_MODEL_ROOT,
        "MODEL_ROOT": MODEL_ROOT,
        "TEXT_MODEL_BASE_DIR": TEXT_MODEL_BASE,
    }
    if len({str(value) for value in roots.values()}) != len(roots):
        return
    for arm, spec in PANEL.items():
        variable = spec.path_variable
        expected = roots[variable]
        built = spec.path if variable == "TEXT_MODEL_DIR" else spec.path.parent
        if built != expected:
            raise AssertionError(
                f"{arm}: declares path_variable {variable!r} but its path {spec.path} "
                f"is not built from {expected}"
            )


_check_declared_paths_match_their_variables()


#: The arm's checkpoint path *relative to* :func:`model_variable`'s answer: ``"."``
#: when the arm is declared as that variable itself, and the checkpoint directory's
#: own name otherwise.
#:
#: :func:`model_variable` alone is the wrong granularity for a preflight. Six of the
#: seven text arms resolve ``TEXT_MODEL_BASE_DIR``, which is the models
#: *root*: it exists as soon as any text checkpoint is staged, so an arm whose own
#: checkpoint was absent passed the worker's data check and raised inside
#: ``load_arm`` instead -- and ``cohort_power`` scores all seven text arms in one
#: process, so that lost the six arms that were fine along with the one that was
#: not. Checking ``${!variable}/<relative>`` turns it back into a logged skip.
#:
#: Derived from ``ArmSpec.path``, never a restated leaf name, and classified through
#: :func:`model_variable` so there is exactly one place that decides which constant
#: an arm's path is built from. Re-pointing any of the three environment variables
#: moves the constant and the arm's path together, so this is as host-independent as
#: the variable mapping it accompanies.
def model_relative_path(arm: str) -> str:
    if model_variable(arm) == "TEXT_MODEL_DIR":
        return "."
    return PANEL[arm].path.name


#: Corpus variables a cohort covering this arm needs, from the arm's declared
#: evaluation cohort rather than from its name. ``zymctrl_ec`` needs the
#: EC-labelled FASTA *and* nothing else: its records carry the conditioning tag
#: that ``Cohort.input_strings`` rebuilds the native prompt from.
_CORPUS_VARIABLES: dict[str, tuple[str, ...]] = {
    "openwebtext": ("OPENWEBTEXT_DIR",),
    "swissprot": ("SWISSPROT_FASTA",),
    "zymctrl_ec": ("ZYMCTRL_FASTA",),
}


def corpus_variables(arm: str) -> tuple[str, ...]:
    source = PANEL[arm].evaluation_cohort_source
    if source not in _CORPUS_VARIABLES:
        raise AssertionError(
            f"{arm}: evaluation cohort source {source!r} has no declared corpus "
            f"variable; known sources are {sorted(_CORPUS_VARIABLES)}"
        )
    return _CORPUS_VARIABLES[source]


def _check_data_locations() -> None:
    for arm in CAMPAIGN_PANEL:
        model_variable(arm)
        model_relative_path(arm)
        corpus_variables(arm)


_check_data_locations()


# -------------------------------------------------------------- serialisation


def contract_payload() -> dict[str, Any]:
    """The whole resolved contract, as the record a run manifest can embed."""

    stages: dict[str, Any] = {}
    for stage, contract in STAGE_CONTRACTS.items():
        eligible, refused = stage_arms(stage)
        stages[stage] = {
            "entry_point": contract.entry_point,
            "scope": contract.scope,
            "required_capabilities": sorted(contract.capabilities),
            "required_architectures": (
                None if contract.architectures is None else sorted(contract.architectures)
            ),
            "architecture_source": contract.architecture_source or None,
            "required_modalities": (
                None if contract.modalities is None else sorted(contract.modalities)
            ),
            "required_tokenisations": (
                None if contract.tokenisations is None else sorted(contract.tokenisations)
            ),
            "required_input_formats": (
                None if contract.input_formats is None else sorted(contract.input_formats)
            ),
            "declared_arms": (
                None if contract.declared_arms is None else list(contract.declared_arms)
            ),
            "excluded_arms": dict(contract.excluded_arms),
            "cohort_band": cohort_band_record(contract),
            "eligible_arms": eligible,
            "refused_arms": {v.arm: v.reason for v in refused},
            "notes": contract.notes or None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_panel": list(CAMPAIGN_PANEL),
        "arms": {
            arm: {
                "modality": PANEL[arm].modality,
                "model_variable": model_variable(arm),
                "model_relative_path": model_relative_path(arm),
                "corpus_variables": list(corpus_variables(arm)),
            }
            for arm in CAMPAIGN_PANEL
        },
        "panel_members_not_staged": dict(PANEL_MEMBERS_NOT_STAGED),
        "qualifying_protein_residue_band": list(QUALIFYING_PROTEIN_BAND),
        "paa_census_pool_width": PAA_CENSUS_WIDTH,
        "stage_order": list(STAGE_ORDER),
        "stages": stages,
        "cohort_power_items": [
            {
                "item": item.item,
                "arms": list(item.arms),
                "extra_args": list(item.extra_args),
                "cohort_name": item.cohort_name,
                "reason": item.reason,
            }
            for item in cohort_power_items()
        ],
        # The same truncation-curve rule, answered per staged checkpoint. Not
        # rendered into panel_contract.sh: the worker schedules panel arms, and a
        # staged rung reaches a card through a campaign manifest, which is what
        # this block is for.
        "staged_arm_cohort_power": {
            arm: {
                "output_logit_width": STAGED_OUTPUT_LOGIT_WIDTH[arm],
                "vocabulary_regime": vocabulary_regime(arm),
                "extra_args": list(staged_cohort_power_args(arm)),
            }
            for arm in sorted(STAGED_OUTPUT_LOGIT_WIDTH)
        },
        "truncation_curve_logit_limit": TRUNCATION_CURVE_LOGIT_LIMIT,
    }


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def render_shell() -> str:
    """The bash fragment the controller and the worker source.

    Associative arrays rather than one variable per (stage, arm): arm names carry
    ``.`` and ``-`` (``qwen2.5-0.5b``), which cannot appear in a bash variable
    name, and encoding around that is exactly the kind of second representation
    this file exists to remove.
    """

    payload = contract_payload()
    lines = [
        "# GENERATED by scripts/capability/panel_contract.py --emit. Do not edit.",
        "# Sourced by run_transfer_h200.sh (controller) and h200_worker.sh (worker),",
        "# which is why the campaign panel and every stage's arm list exist in exactly",
        "# one place. h200_worker.sh re-derives this file from src/capability/core/arms.py in",
        "# its preflight (panel_contract.py --verify) and refuses to schedule a GPU if",
        "# the two disagree, so a stale copy cannot reach a measurement.",
        f"TRANSFER_CONTRACT_SCHEMA={_quote(SCHEMA_VERSION)}",
        f"TRANSFER_CAMPAIGN_PANEL={_quote(' '.join(payload['campaign_panel']))}",
        f"TRANSFER_STAGE_ORDER={_quote(' '.join(payload['stage_order']))}",
        # Not a scale knob: the width paa_census's eligible arm list is declared
        # against. See PAA_CENSUS_WIDTH in panel_contract.py.
        f"TRANSFER_PAA_CENSUS_WIDTH={_quote(str(payload['paa_census_pool_width']))}",
        "declare -A TRANSFER_STAGE_SCOPE=()",
        "declare -A TRANSFER_STAGE_ENTRY=()",
        "declare -A TRANSFER_STAGE_ARMS=()",
        "declare -A TRANSFER_STAGE_REFUSAL=()",
        "declare -A TRANSFER_ARM_MODALITY=()",
        "declare -A TRANSFER_ARM_MODEL_VAR=()",
        "declare -A TRANSFER_ARM_MODEL_REL=()",
        "declare -A TRANSFER_ARM_CORPUS_VARS=()",
        "declare -A TRANSFER_COHORT_ITEM_ARMS=()",
        "declare -A TRANSFER_COHORT_ITEM_ARGS=()",
        "declare -A TRANSFER_COHORT_ITEM_COHORT_NAME=()",
    ]
    for arm in payload["campaign_panel"]:
        record = payload["arms"][arm]
        key = _quote(arm)
        lines.append(f"TRANSFER_ARM_MODALITY[{key}]={_quote(record['modality'])}")
        lines.append(f"TRANSFER_ARM_MODEL_VAR[{key}]={_quote(record['model_variable'])}")
        lines.append(f"TRANSFER_ARM_MODEL_REL[{key}]={_quote(record['model_relative_path'])}")
        lines.append(
            f"TRANSFER_ARM_CORPUS_VARS[{key}]={_quote(' '.join(record['corpus_variables']))}"
        )
    for stage in payload["stage_order"]:
        record = payload["stages"][stage]
        lines.append(f"TRANSFER_STAGE_SCOPE[{_quote(stage)}]={_quote(record['scope'])}")
        lines.append(f"TRANSFER_STAGE_ENTRY[{_quote(stage)}]={_quote(record['entry_point'])}")
        lines.append(
            f"TRANSFER_STAGE_ARMS[{_quote(stage)}]={_quote(' '.join(record['eligible_arms']))}"
        )
        for arm, reason in sorted(record["refused_arms"].items()):
            key = f"{stage}/{arm}"
            lines.append(f"TRANSFER_STAGE_REFUSAL[{_quote(key)}]={_quote(reason)}")
    cohort_items = [item["item"] for item in payload["cohort_power_items"]]
    lines.append(f"TRANSFER_COHORT_ITEMS={_quote(' '.join(cohort_items))}")
    for item in payload["cohort_power_items"]:
        key = _quote(item["item"])
        lines.append(f"TRANSFER_COHORT_ITEM_ARMS[{key}]={_quote(' '.join(item['arms']))}")
        lines.append(f"TRANSFER_COHORT_ITEM_ARGS[{key}]={_quote(' '.join(item['extra_args']))}")
        lines.append(
            f"TRANSFER_COHORT_ITEM_COHORT_NAME[{key}]="
            f"{_quote(item['cohort_name'] or '')}"
        )
    return "\n".join(lines) + "\n"


def emit(path: Path = SHELL_CONTRACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_shell(), encoding="utf-8")
    return path


def verify(path: Path = SHELL_CONTRACT) -> list[str]:
    """Differences between the rendered file and the live panel; empty means clean."""

    if not path.exists():
        return [f"{path} does not exist; run panel_contract.py --emit"]
    on_disk = path.read_text(encoding="utf-8")
    expected = render_shell()
    if on_disk == expected:
        return []
    disk_lines = on_disk.splitlines()
    want_lines = expected.splitlines()
    problems = [f"{path} disagrees with src/capability/core/arms.py"]
    for index in range(max(len(disk_lines), len(want_lines))):
        got = disk_lines[index] if index < len(disk_lines) else "<missing>"
        want = want_lines[index] if index < len(want_lines) else "<missing>"
        if got != want:
            problems.append(f"  line {index + 1}: on disk [{got}] expected [{want}]")
    return problems


def declared_arms_in_source(entry_point: str, symbol: str) -> tuple[str, ...]:
    """A module-level tuple literal read out of a sibling entry point without importing it.

    The numbered entry points cannot be imported by name, and importing one by
    path executes its module body. This reads the declaration statically, which
    is what lets :data:`STAGE_CONTRACTS` mirror ``homology_control.py``'s own
    arm list under test instead of restating it and hoping.
    """

    from scripts.capability.entrypoints import stage_path
    path = stage_path(REPO_ROOT, Path(entry_point).name)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    return tuple(ast.literal_eval(node.value))
    raise LookupError(f"{entry_point} has no module-level assignment to {symbol}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true", help="write panel_contract.sh")
    parser.add_argument(
        "--verify", action="store_true", help="fail if panel_contract.sh is stale"
    )
    parser.add_argument("--json", action="store_true", help="print the resolved contract")
    parser.add_argument("--path", type=Path, default=SHELL_CONTRACT)
    args = parser.parse_args()

    if args.json:
        print(json.dumps(contract_payload(), indent=2, sort_keys=True))
    if args.emit:
        print(f"wrote {emit(args.path)}")
    if args.verify:
        problems = verify(args.path)
        if problems:
            for line in problems:
                print(line, file=sys.stderr)
            raise SystemExit(2)
        print(f"{args.path} matches src/capability/core/arms.py")
    if not (args.emit or args.verify or args.json):
        parser.error("nothing to do: pass --emit, --verify or --json")


if __name__ == "__main__":
    main()
