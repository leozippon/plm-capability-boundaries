"""One control-set hierarchy and one band convention for the lanes beyond R1.

The variant-effect ladder in :mod:`.statistical_ladder` already contrasts the model
likelihood against explicit evolutionary statistics: pairwise couplings (a plmc Potts
energy difference) and GEMME. No other lane did, so everywhere else the model was
contrasted against weaker controls. This module is the declaration half of closing
that gap: it states the control-set hierarchy, the two directions, the matched
support rule, the multiplicity family and the finding states for the abundance,
stability, interaction and added-phenotype lanes, and it assembles and infers the
family a lane's own runner has fitted.

It fits nothing. Each lane keeps its own fitter, its own panel and its own archived
reproduction gate; :func:`fit_plan` names the designs that fitter must produce and
:func:`panel` reads the per-group held-out errors back. That boundary is the same one
``statistical_ladder`` draws, and it is why four heterogeneous panels are not
reimplemented here.

**Blocks are named by role, not by letter.** The same letter means different things in
different lanes: ``G`` is GEMME in the variant-effect ladder and the training-fold
nonlinear-additive nuisance in the abundance, stability and interaction lanes. So the
vocabulary is :data:`ROLES` -- lane nuisances, profile, couplings, GEMME -- and each
lane maps a role onto its own local block keys. The two new blocks take the lane-local
keys :data:`COUPLINGS_BLOCK` and :data:`GEMME_BLOCK`, which collide with no lane's
existing keys; :func:`_check_declarations` asserts that at import. Nothing frozen moves
and no letter is renamed.

**One hierarchy, identical in every lane** (:data:`RUNGS`): nuisances alone; plus
profile; plus profile and couplings; plus profile and GEMME; plus profile, couplings
and GEMME. Every lane already carries profile inside its primary control set, so its
nuisance-only rung is that set *without* profile and the first step is a genuine
increment rather than a relabelling. The second rung is exactly the lane's archived
primary control set, in its archived column order, which is what lets a lane reuse its
archived control and joint fits under its own reproduction pilot.

**Two directions**, from :data:`bidirectional.PRIMARY_DESIGNS` unchanged: what the
model adds to the controls, and what the controls add to the pure scalar readout.
Overlap is established only when both increments are small; the reverse direction is
what distinguishes shared information from model inferiority. These lanes have a
genuine groups-by-columns matrix -- assays or homology groups are the resampling unit
and every column carries a value for every group -- so :func:`bidirectional.paired_bands`
applies directly and no substitute estimand is invented.

**The likelihood channel is primary.** Where a lane's own fitter declares a projected
representation block it is reported as a secondary channel in its own family; the two
are never merged.

**Support is matched.** Within one stratum every control set is fitted on the identical
complete-case rows, where every block of that lane's hierarchy is available
(:func:`complete_case`). A nested increment computed on shifting rows means nothing.
The losses are enumerated by name, count and upstream reason: seventeen references
across the new cohorts have too few retained homologs to carry a coupling model, and a
reference whose alignment is too shallow is refused by GEMME as degenerate. The
nuisance-only rung needs neither new block, so it is also fitted on the full support
and reported beside the hierarchy as a labelled sensitivity.

**GEMME is non-identifiable for the interaction lane.** The official ``-m`` score of a
multi-mutant is the sum of its per-site scores, so a double-mutant cycle contrast of
GEMME scores is identically zero. That lane's hierarchy stops at profile and couplings
and the role is recorded as ``non_identifiable`` with that reason, never estimated as a
zero with a tight interval around it.

**Four finding states, kept apart** (:data:`STATES`, declared once in
:mod:`..core.statistics`): ``supported`` is a simultaneous interval strictly above
zero, ``bounded_negative`` one strictly below it, ``unresolved`` an interval containing
zero however narrow or a stratum under the bootstrap-unit floor, and
``non_identifiable`` a block that cannot inform the estimand at all. A wide interval is
not a negative result, and no equivalence margin is declared anywhere.

**The forward bound carries the overlap claim** (:func:`forward_bound`). Because no
margin is declared, the forward direction can never report a state for smallness, and
an unresolved forward increment is not the end of the statement: the simultaneous upper
limit is reported as a first-class quantity, in native units and as a fraction of the
control set's own performance on the identical support. *The model adds at most this
much beyond the explicit statistics, and at most this fraction of what those statistics
already achieve* is a bound on the increment, not a failure to reject, and it needs no
threshold.

**The variant-effect ladder is not re-run here** (:data:`VARIANT_EFFECT_BRIDGE`). Its
profile family already supplies rungs two through five with an empty nuisance set and
its ``C0`` family rungs two and five; adding the three rungs it lacks would enlarge its
declared family, widen its bands and could move published verdicts. Cross-lane reading
therefore happens at :data:`SHARED_RUNGS`, the rungs every lane has.

**An archived fit is a pilot, not saved compute** (:data:`ARCHIVED_REUSE_RULE`). The
profile rung reproduces each lane's archived control set column for column, but the
complete case drops rows, so the archived fit is on a different support. It is refitted
on the full support and required to reproduce the published values instead.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import bidirectional, overlap, responses
from .progression import identity_digest
from ..core import statistics
from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS, bootstrap_unit_floor
from ..interactions.pairwise_epistasis import ROSTER

#: The role vocabulary. A role is what a block *is* for this estimand; the letter that
#: carries it is the lane's own business.
ROLES = ('nuisance', 'profile', 'couplings', 'gemme')

#: Lane-local keys of the two added blocks. They are spelled out rather than lettered
#: because ``E`` and ``G`` already mean other things in other lanes, and a single
#: vocabulary that collides nowhere is cheaper than four per-lane exceptions.
COUPLINGS_BLOCK = 'POTTS'
GEMME_BLOCK = 'GEMME'
NEW_BLOCKS = {'couplings': COUPLINGS_BLOCK, 'gemme': GEMME_BLOCK}

#: The variant-effect ladder's own role binding. It is recorded here as the witness of
#: the collision this vocabulary removes -- ``G`` is GEMME there and the nonlinear
#: additive nuisance in three lanes below -- and as the comparability reference.
VARIANT_EFFECT_ROLES = {'nuisance': ('B', 'L'), 'profile': ('P',),
                        'couplings': ('E',), 'gemme': ('G',)}

#: How the variant-effect ladder maps onto this hierarchy, and where it does not.
#:
#: That lane is **not** re-run at this hierarchy and its control sets stay frozen in
#: :mod:`.statistical_ladder`. The reason is not only that the contract is frozen:
#: adding rungs would enlarge its multiplicity family, widen its simultaneous bands
#: and could move verdicts on findings that are already published, which is a
#: manuscript change made through the back door.
#:
#: Its profile family supplies rungs two through five with an *empty* nuisance set --
#: ``P``, ``P+E``, ``P+G``, ``P+E+G`` -- and its ``C0`` family supplies rungs two and
#: five, ``C0`` and ``C0+E+G``. It therefore lacks the nuisance-only rung and the two
#: intermediate ``C0`` rungs (``C0+E`` and ``C0+G``), and those three are not
#: estimated for it anywhere.
VARIANT_EFFECT_BRIDGE = {
    'lane': 'variant_effect',
    'declared_in': 'src/capability/extensions/statistical_ladder.py',
    'rerun_at_this_hierarchy': False,
    'rerun_refused_because': ('three added designs would enlarge that lane\'s declared '
                              'multiplicity family, widen its simultaneous bands and could move '
                              'already published verdicts; the published ladder stands as it is'),
    'profile_family': {'nuisance_set': 'empty',
                       'rungs': {'nuisance+profile': 'P',
                                 'nuisance+profile+couplings': 'PE',
                                 'nuisance+profile+gemme': 'PG',
                                 'nuisance+profile+couplings+gemme': 'PEG'}},
    'c0_family': {'nuisance_set': 'B + L',
                  'rungs': {'nuisance+profile': 'BPL',
                            'nuisance+profile+couplings+gemme': 'BPLEG'}},
    'absent_rungs': {'nuisance': 'the C0 family has no B+L-only control set',
                     'nuisance+profile+couplings': 'C0+E is not declared',
                     'nuisance+profile+gemme': 'C0+G is not declared'},
    'cross_lane_reading': ('cross-lane reading happens at the rungs every lane shares, which '
                           'are rungs two through five of the profile family: profile; profile '
                           'and couplings; profile and GEMME; profile, couplings and GEMME. A '
                           'lane\'s nuisance-only rung and its nuisance-conditioned rungs are '
                           'read within that lane only, because the nuisance sets differ'),
}
#: The rungs every lane shares, and the only ones read across lanes.
SHARED_RUNGS = ('nuisance+profile', 'nuisance+profile+couplings',
                'nuisance+profile+gemme', 'nuisance+profile+couplings+gemme')

#: The one hierarchy, in order: rung name -> the roles that rung carries.
RUNGS: dict[str, tuple[str, ...]] = {
    'nuisance': ('nuisance',),
    'nuisance+profile': ('nuisance', 'profile'),
    'nuisance+profile+couplings': ('nuisance', 'profile', 'couplings'),
    'nuisance+profile+gemme': ('nuisance', 'profile', 'gemme'),
    'nuisance+profile+couplings+gemme': ('nuisance', 'profile', 'couplings', 'gemme'),
}
#: The rung that reproduces each lane's archived primary control set exactly.
ARCHIVED_RUNG = 'nuisance+profile'
#: The rungs fitted on the full support: the nuisance-only sensitivity and the archived
#: reproduction pilot. Neither carries an added block, which is what lets a full-support
#: fit be bound to an identity that excludes the added-block vocabulary entirely
#: (:func:`fit_identity`). :func:`_check_declarations` asserts it rather than trusting it.
FULL_SUPPORT_RUNGS = ('nuisance', ARCHIVED_RUNG)

#: Three arm names per rung, taken from the shared directions so a pairing cannot
#: drift: the control arm, the pure model readout and their union.
CONTROL_ARM, JOINT_ARM = bidirectional.PRIMARY_DESIGNS['model_given_controls']
MODEL_ARM = bidirectional.PRIMARY_DESIGNS['controls_given_model'][0]
ARM_NAMES = (CONTROL_ARM, MODEL_ARM, JOINT_ARM)

#: Model channels: the block each adds, and the role it plays in the report. The
#: likelihood channel is the quantity the manuscript contrasts against explicit
#: statistics; a representation channel is reported separately and never merged in.
CHANNELS = {'likelihood': {'block': 'M', 'role': 'primary'},
            'representation': {'block': 'R', 'role': 'secondary'}}

#: Finding states, kept rigorously apart. The vocabulary is declared once in
#: :mod:`..core.statistics` and read here, never respelled.
SUPPORTED = statistics.SUPPORTED
BOUNDED_NEGATIVE = statistics.BOUNDED_NEGATIVE
UNRESOLVED = statistics.UNRESOLVED
NON_IDENTIFIABLE = statistics.NON_IDENTIFIABLE
STATES = statistics.FINDING_STATES
#: The shared verdict rule, relabelled onto the four states. ``bounded_negative`` is a
#: resolved negative *sign* here: these lanes declare no equivalence margin, so the
#: forward direction can never report a state for smallness and reports
#: :func:`forward_bound` instead.
VERDICT_STATE = {'positive': SUPPORTED, 'negative': BOUNDED_NEGATIVE,
                 'unresolved': UNRESOLVED}
#: What the forward direction's simultaneous upper limit is, and is not.
FORWARD_BOUND_READING = (
    'the forward direction never reports a state for smallness: it declares no '
    'equivalence margin, so an interval containing zero is unresolved however narrow. '
    'The quantitative claim is carried by the simultaneous upper limit instead -- the '
    'model adds at most this much beyond the explicit statistics, and at most this '
    'fraction of what those statistics already achieve on the identical support -- '
    'which is a bound on the increment rather than a failure to reject')

#: Upstream per-row status vocabularies of the two score tables, classified. An
#: ``available`` row carries a finite value, a ``refused`` row is an enumerable loss of
#: the declared kind, and a ``pending`` row means the wave has not landed. Any other
#: status -- a failed or unconverged background, a partly scored wild type -- is
#: neither, and the reader refuses rather than guessing which.
#:
#: ``no_homolog_support`` means the same thing on both rungs and is classified the same
#: way on both: the reference has fewer than two retained homolog rows, so it carries
#: no coupling model and no GEMME run, and neither block can ever value its rows. It is
#: a refusal rather than a pending row precisely because no run can be made.
AVAILABLE, REFUSED, PENDING = 'available', 'refused', 'pending'
ROW_STATUS = {
    'couplings': {'fitted': AVAILABLE, 'no_homolog_support': REFUSED,
                  'not_fitted': PENDING},
    'gemme': {'scored': AVAILABLE, 'refused_degenerate': REFUSED,
              'no_homolog_support': REFUSED, 'wildtype_pending': PENDING},
}
#: The score column and the upstream table schema each block is read from.
BLOCK_SOURCE = {
    'couplings': {'schema': 'r1_msa_baseline_scores_v1', 'column': 'potts_delta',
                  'producer': 'scripts/capability/baselines/r1_msa_baselines.py'},
    'gemme': {'schema': 'r1_gemme_scores', 'column': 'gemme_score',
              'producer': 'scripts/capability/baselines/score_r1_gemme.py'},
}

#: How an added block enters a double-mutant cycle estimand, declared.
#:
#: The score tables value *mutants*; a row of the interaction lane is a four-corner
#: cycle. The added block of a cycle is therefore the cycle contrast of its scored
#: corners, ``v(AB) - v(A) - v(B) + v(WT)`` with ``v(WT) = 0`` by construction, and a
#: cycle enters the complete case only when all three of its scored corners do.
#:
#: This is declared rather than left to be inferred because the lane's
#: non-identifiability finding rests on it: :data:`GEMME_CYCLE_REASON` holds only if the
#: block enters as this contrast, and a published result must not depend on a reading a
#: later reader has to reconstruct from the reason given for it.
CYCLE_CONTRAST = {
    'definition': 'v(AB) - v(A) - v(B) + v(WT), with v(WT) = 0 by construction',
    'corners': ('double', 'low single', 'high single'),
    'support_rule': ('a cycle enters the complete case only when every one of its three '
                     'scored corners does, because the contrast is undefined otherwise'),
    'why_declared': ('the non-identifiability of GEMME for this estimand is a consequence '
                     'of this definition, so the definition is declared beside it'),
}

#: Why the GEMME block cannot inform a double-mutant cycle contrast.
GEMME_CYCLE_REASON = (
    "the official GEMME -m score of a multi-mutant is the sum of its per-site scores "
    "(pred.R: normPred[i] = sum(predTMP * trace[myPos]), with no cross term), so the "
    "cycle contrast s(AB) - s(A) - s(B) + s(WT) with s(WT) = 0 by construction is "
    "identically zero; the block carries no non-additive component for this estimand "
    "and is recorded as non-identifiable rather than estimated at zero with a tight "
    "interval around it")

#: What the nuisance rung does and does not establish.
#:
#: ``G`` is a frozen block whose first stage reads the lane's standing columns, and those
#: include the profile columns in all three lanes beyond the added phenotypes. So the
#: nuisance rung drops the profile *columns* and still carries profile *information*
#: through ``G``. Refitting ``G`` without them would stop the profile rung reproducing
#: each lane's archive column for column, and would buy a decomposition this hierarchy
#: does not claim, so ``G`` stays exactly as the archive defined it.
#:
#: **What this does not threaten.** The leaked columns are themselves explicit
#: evolutionary statistics, so every rung's control set is exactly what the claim names
#: it, and the forward bound -- the model adds at most this much beyond the explicit
#: statistics -- is unaffected on every rung.
#:
#: **What it does threaten.** Attribution *among* rungs. How much the profile columns
#: alone contribute is not cleanly separable from what ``G`` already carries, so the
#: first step of the hierarchy reads as "plus the profile columns", never as "plus all
#: profile information", and a per-rung decomposition is not available from it.
NUISANCE_RUNG_LIMITATION = (
    'the nuisance rung drops the profile columns and still carries profile information '
    'through G, whose frozen first stage reads the lane\'s standing columns. This does not '
    'weaken any forward bound: the leaked columns are themselves explicit evolutionary '
    'statistics, so each control set is what the claim names it. What it costs is '
    'attribution among rungs -- the profile columns\' own contribution is not separable '
    'from what G already carries -- so the first step reads as "plus the profile columns" '
    'and never as "plus all profile information"')

#: The coupling fitter's own instrument bounds, declared separately and never pooled.
#:
#: The first is the fitter's single-host run-to-run spread on a per-variant score. The
#: second applies to a fit that stopped at its iteration cap and is about four times the
#: first. They are deliberately not averaged, because an average reports neither: a
#: comparison is read against whichever bound applies to the fits it rests on.
#:
#: **No comparison may resolve a difference below the applicable bound.** Both finished
#: cohorts carry zero non-converged backgrounds, so for the abundance and pairwise
#: cohorts the operative bound is the run-to-run one.
#:
#: This is documentation of the added block's measurement precision. It cannot change a
#: fitted number and is therefore outside :func:`fit_identity`, which is the whole point
#: of that function: recording an instrument bound must not cost a refit.
COUPLING_INSTRUMENT_BOUNDS = {
    'run_to_run_per_variant_score': 1.071e-2,
    'iteration_cap_fit': 4.303e-2,
    'pooling': 'declared separately and never averaged; an average reports neither',
    'rule': 'no comparison may resolve a difference below the applicable bound',
    'operative_for_finished_cohorts': 'run_to_run_per_variant_score, because the abundance '
                                      'and pairwise cohorts carry zero non-converged '
                                      'backgrounds',
}

#: What the two sides of every contrast actually are. Recorded in each panel because a
#: reader who misses it will take the result for a claim about model quality in general.
CONTROL_PROVENANCE = (
    'the control set is fitted per family on the target protein\'s own retained alignment: '
    'the profile columns, the coupling scalar and the GEMME scalar are all family-specific '
    'statistics of that protein\'s homologs, selected and tuned inside the training folds of '
    'this very cohort. The arm is a general model scored zero-shot, with one scalar '
    'likelihood difference per variant and no access to the target family. So a resolved '
    'reverse increment says that family-specific evolutionary statistics predict these '
    'endpoints better than a general model\'s sequence likelihood read out zero-shot; it is '
    'not a statement about model quality in general, about the models\' parameters, or about '
    'what a model fine-tuned on this family could do')

#: Why a per-column limit here is a statement about every arm at once.
SIMULTANEITY = (
    'one critical value is taken as the 95% quantile of the family-wise maximum absolute '
    'centered bootstrap deviation over every column of the family, and each column carries '
    'point +- that value. Each per-column interval is therefore simultaneous, not pointwise: '
    'a bound read off the largest per-arm upper limit holds for every arm of the family at '
    'once, including the strongest. Both directions of a stratum share the one critical '
    'value, so the forward and reverse limits are jointly valid')

#: Stable schema labels. A label is an identity, not a version counter: a reader
#: accepts the names it knows and refuses any other rather than stripping a suffix.
CONTRACT_SCHEMA = 'explicit_statistics_lane_contract'
SUPPORT_SCHEMA = 'explicit_statistics_lane_support'
CELLS_SCHEMA = 'explicit_statistics_lane_cells'
PANEL_SCHEMA = 'explicit_statistics_lane_panel'
CELLS_SCHEMAS = (CELLS_SCHEMA,)

#: Default file names inside a cohort's export directory.
ROWS_FILE = Path('inputs/variant_table.json')
ROWS_SCHEMA = 'r1_msa_variant_table_v1'
SUPPORT_FILE = Path('inputs/support.json')
UPSTREAM_SUPPORT_SCHEMA = 'r1_msa_support_v1'
REFUSALS_FILE = Path('inputs/refusals.json')
REFUSALS_SCHEMA = 'explicit_statistics_refusals_v1'

#: Who is the authority on a reference the export never scored.
#:
#: A block table enumerates the references *it* refused. A reference with too few
#: retained homolog rows was refused once, by the export, for both blocks at once, and
#: the export's own refusals document is where that is recorded. A loss of that kind is
#: therefore read from the export rather than re-enumerated in each block table: one
#: statement of one fact, where two could drift apart and only one of them be read.
#: :func:`complete_case` accepts a lost reference that either the block table or the
#: export enumerates, and refuses one that neither does.
EXPORT_REFUSAL_AUTHORITY = (
    'a reference with fewer than two retained homolog rows was refused by the cohort '
    'export for both blocks at once, so that loss is read from the export\'s own refusals '
    'document rather than re-enumerated per block table; a block table remains the '
    'authority on the refusals only it can see, such as a degenerate GEMME run')

#: The retained-alignment-row band every GEMME degeneracy observed so far has fallen
#: in: all seven degenerate R1 wild types had at most 14 retained rows and none at 15
#: or more, and two rows is the floor below which no alignment is retained at all. It
#: is a risk band read off the exported support, never a prediction and never a count
#: written into this module: :func:`gemme_risk_band` counts it from the cohort's own
#: support document so the number cannot drift from the export.
GEMME_RISK_BAND = (2, 14)


@dataclass(frozen=True)
class Stratum:
    """One endpoint of one lane: its units, its loss sign and its row support."""

    units: str
    #: +1 where the per-group statistic is a loss, -1 where it is a score, so that a
    #: paired reduction is positive when the joint arm is better in either reading.
    to_loss: float
    support: str
    #: Declared where this stratum's resampling units leave it no headroom over the
    #: floor. A support loss then reports the stratum unresolved; the floor never moves.
    fragile: str | None = None


@dataclass(frozen=True)
class Lane:
    """One lane's role bindings, strata, arms and declared families."""

    name: str
    cohort: str
    estimand: str
    unit: str
    #: The lane's own profile block keys, in its own order. The hierarchy carries them
    #: from the profile rung upwards whether or not the lane's archived forward
    #: qualification kept them.
    profile_blocks: tuple[str, ...]
    #: The arm-qualified tokenisation nuisance key, or None where the lane has none. It
    #: is read from the lane's archive and never re-qualified here.
    tokenisation_block: str | None
    strata: dict[str, Stratum]
    channels: tuple[str, ...]
    arms: tuple[str, ...]
    panel_arms: tuple[str, ...]
    declared_family: dict[str, int]
    full_panel_family: dict[str, int]
    non_identifiable: dict[str, str] = field(default_factory=dict)
    #: Readings of this lane that exist upstream and are deliberately not covered by
    #: this hierarchy, each with the reason, so that an absence is declared rather than
    #: left for a reader to notice.
    not_covered: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def rungs(self) -> tuple[str, ...]:
        """The identifiable rungs: the hierarchy minus any rung carrying a role this
        lane cannot identify."""

        return tuple(name for name, roles in RUNGS.items()
                     if not set(roles) & set(self.non_identifiable))

    @property
    def unidentifiable_rungs(self) -> tuple[str, ...]:
        return tuple(name for name in RUNGS if name not in self.rungs)

    def required_blocks(self) -> tuple[str, ...]:
        """The added blocks whose availability defines this lane's complete case."""

        return tuple(role for role in ('couplings', 'gemme')
                     if role not in self.non_identifiable)


#: The 33-arm panel every lane but the added-phenotype one has scored.
PANEL_ARMS = tuple(sorted(ROSTER))
#: The ten arms the added-phenotype lane has scored. The remaining arms are queued, so
#: its family is declared from the realised set and the declaration states what the
#: family becomes at full panel.
PHENOTYPE_ARMS = ('galactica-1.3b', 'gpt2-large', 'gpt2-xl', 'instructprotein',
                  'progen2-medium', 'progen2-small', 'progen2-xlarge', 'prollama',
                  'prollama-stage-1', 'protgpt2')


def _family_size(rungs: int, arms: int) -> int:
    return rungs * arms * len(bidirectional.DIRECTIONS)


LANES: dict[str, Lane] = {
    'abundance': Lane(
        name='abundance',
        cohort='abundance',
        estimand='held-out prediction of Domainome abundance effects of single substitutions',
        unit='family group; three split seeds averaged within group before any resample',
        profile_blocks=('prof', 'prof2'),
        tokenisation_block='T',
        strata={
            'squared_error': Stratum(
                units=('squared normalised fitness; group-equal mean squared error with '
                       'domains, then sites, then variants weighted equally inside a group'),
                to_loss=1., support='cohort rows'),
            'ranking': Stratum(
                units=('dimensionless within-domain Spearman rho, averaged over the domains '
                       'of a group; groups equal'),
                to_loss=-1., support='cohort rows'),
        },
        channels=('likelihood', 'representation'),
        arms=PANEL_ARMS, panel_arms=PANEL_ARMS,
        declared_family={'likelihood': _family_size(5, 33), 'representation': _family_size(5, 33)},
        full_panel_family={'likelihood': _family_size(5, 33),
                           'representation': _family_size(5, 33)},
        notes=('the archived qualified control set carries the training-fold nonlinear '
               'additive response G and, for the arms whose archived matched baseline is '
               'S_T, that arm\'s own tokenisation block; both are nuisances and neither '
               'reaches the model arm',
               'the nuisance control is arm-conditional by design: each arm\'s nuisance set '
               'has to absorb that arm\'s own tokenisation artefacts, so a control held common '
               'across arms would credit segmentation to the model and would be the error. The '
               'forward increment is therefore "the model beyond this arm\'s own controls"',
               'the declared representation channel cannot be fitted from this lane\'s archive: '
               'the restored per-arm extraction exports the scalar likelihood block and the '
               'tokenisation block and nothing else, so there is no representation block to put '
               'on a model arm and that family stays pending for that reason. It stays declared '
               'because the declared channels are part of this lane\'s fit identity, and '
               'withdrawing one would restamp every fitted cell of the lane as stale',)),
    'stability': Lane(
        name='stability',
        cohort='pairwise',
        estimand='held-out prediction of measured single-substitution stability change',
        unit='family group; three split seeds averaged within group before any resample',
        profile_blocks=('prof2',),
        tokenisation_block='T',
        strata={
            'ranking': Stratum(units='within-background Spearman rho', to_loss=-1.,
                               support='cohort rows'),
            'transfer_error': Stratum(units='squared kcal/mol; group-equal, site-equal MSE',
                                      to_loss=1., support='cohort rows'),
            'calibrated_error': Stratum(
                units='squared kcal/mol; site-calibrated group-equal MSE',
                to_loss=1., support='cohort rows'),
        },
        channels=('likelihood',),
        arms=PANEL_ARMS, panel_arms=PANEL_ARMS,
        declared_family={'likelihood': _family_size(5, 33)},
        full_panel_family={'likelihood': _family_size(5, 33)},
        notes=('the lane\'s archived primary control set is S2; its archived S baseline '
               'drops composition as well as profile and is therefore not this hierarchy\'s '
               'nuisance rung, which is S2 without the profile block',
               'the archived grid fitted no representation design for this endpoint, so the '
               'lane has a likelihood channel only',
               'this lane and the interaction lane read one cohort of reference sequences and '
               'have different complete-case supports, because GEMME is identifiable here and '
               'not there; the two supports are never substituted for one another')),
    'interaction': Lane(
        name='interaction',
        cohort='pairwise',
        estimand=('held-out prediction of the measured double-mutant cycle '
                  'y_AB - y_A - y_B + y_WT on the measured-single-assisted residual'),
        unit='family group; site pairs weighted equally inside a group',
        profile_blocks=('prof',),
        tokenisation_block=None,
        strata={
            'cycle_squared_error': Stratum(units='kcal^2/mol^2; group-equal, site-pair-equal MSE',
                                           to_loss=1., support='cycle corners'),
        },
        channels=('likelihood', 'representation'),
        arms=PANEL_ARMS, panel_arms=PANEL_ARMS,
        declared_family={'likelihood': _family_size(3, 33),
                         'representation': _family_size(3, 33)},
        full_panel_family={'likelihood': _family_size(3, 33),
                           'representation': _family_size(3, 33)},
        non_identifiable={'gemme': GEMME_CYCLE_REASON},
        notes=('the tokenisation block T is carried by every arm\'s control set in this '
               'lane rather than qualified per arm, so it is part of the archived control '
               'set the nuisance keys are read from',
               'only the coupling block can carry non-additive information into this '
               'estimand',
               'this lane and the stability lane read one cohort of reference sequences and '
               'have different complete-case supports, because GEMME restricts that lane\'s '
               'support and cannot restrict this one')),
    'phenotype': Lane(
        name='phenotype',
        cohort='phenotype',
        estimand=('held-out prediction of the added phenotype cohorts: catalytic activity, '
                  'binding and cellular fitness'),
        unit='sequence-homology group at 30% identity over 80% of both sequences',
        profile_blocks=('prof', 'prof2'),
        tokenisation_block=None,
        strata={
            'activity_venus-ranking': Stratum(units='within-background Spearman rho',
                                              to_loss=-1., support='activity cohort rows'),
            'binding_skempi-ranking': Stratum(units='within-background Spearman rho',
                                              to_loss=-1., support='binding cohort rows'),
            'binding_skempi-quantitative': Stratum(
                units='squared kcal/mol; group-equal, background-equal MSE',
                to_loss=1., support='binding cohort rows'),
            'cellular_fitness_sge-ranking': Stratum(
                units='within-background Spearman rho', to_loss=-1.,
                support='cellular fitness cohort rows',
                fragile=(f'nine homology groups against the {MINIMUM_BOOTSTRAP_UNITS}-unit floor: '
                         'one lost group leaves no headroom and two put the stratum under the '
                         'floor. A coupling or GEMME refusal landing here is reported as an '
                         'unresolved stratum; the floor is not lowered to keep it estimable')),
        },
        channels=('likelihood',),
        arms=PHENOTYPE_ARMS, panel_arms=PANEL_ARMS,
        declared_family={'likelihood': _family_size(5, 10)},
        full_panel_family={'likelihood': _family_size(5, 33)},
        not_covered=(
            ('binding_skempi-quantitative_calibrated',
             'the within-family recalibrated binding reading sits on a restricted 97-group '
             'support, so it is neither this stratum nor comparable with it on one support; it '
             'is not covered by this hierarchy and would need its own declaration'),),
        notes=('the lane\'s cohorts measure different quantities on disjoint rows, so each '
               'cohort/endpoint cell is its own stratum with its own support and its own '
               'family; the cells are never pooled',
               'the binding quantitative cell qualified no profile block, and the hierarchy '
               'carries the profile block there by declaration: including it is the entire '
               'purpose of the rung, and a forward qualification that dropped it is exactly '
               'the weak control this work replaces',
               'the frozen fitter declares no representation block for this lane')),
}


def lane(name: str) -> Lane:
    if name not in LANES:
        raise ValueError(f'unknown lane {name!r}; the declared lanes are {sorted(LANES)}')
    return LANES[name]


def stratum(lane_name: str, name: str) -> Stratum:
    declared = lane(lane_name)
    if name not in declared.strata:
        raise ValueError(f'{lane_name}: unknown stratum {name!r}; the declared strata are '
                         f'{sorted(declared.strata)}')
    return declared.strata[name]


def channel_block(lane_name: str, channel: str) -> str:
    """The block one channel's model arm carries, or a refusal naming lane and channel.

    This is the only place a channel request is checked, and it is the right place
    because every path that acts on a channel passes through it: :func:`arm_designs`
    for a design, :func:`declared_family` and :func:`panel` for a family. A guard at
    each of those would be the same question asked four times.

    The refusal is a stated one rather than a bare exception because the channel is
    the one declaration name on this path that arrives from outside the declaration
    -- a cells document's ``channel`` field, or a keyword argument whose default is
    silent -- so an undeclared pair is an operator's mistake and reaches an operator.
    A traceback out of the middle of a family assembly does not say which lane
    declined which channel; this does, and names what the lane does declare.
    """

    declared = lane(lane_name)
    if channel not in declared.channels:
        raise SystemExit(f'{lane_name}: no {channel!r} channel; this lane declares '
                         f'{list(declared.channels)}')
    return CHANNELS[channel]['block']


def role_blocks(lane_name: str, qualified: Sequence[str], *,
                tokenisation: bool = False) -> dict[str, tuple[str, ...]]:
    """This lane's role -> local block keys, given its archived qualified control set.

    ``qualified`` is read from the lane's own archive and is never re-qualified here.
    The nuisance role is everything in it that is not one of the lane's profile keys,
    plus the arm's own tokenisation block where its archive recorded one. The profile
    role is the lane's declared profile keys, in the archived order where the archive
    carries them and appended where it does not, because profile is a control of this
    hierarchy whether or not a lane's forward qualification kept it.
    """

    declared = lane(lane_name)
    qualified = tuple(qualified)
    if len(set(qualified)) != len(qualified):
        raise ValueError(f'{lane_name}: the qualified control set repeats a block: {qualified}')
    collisions = sorted(set(qualified) & set(NEW_BLOCKS.values()))
    if collisions:
        raise ValueError(f'{lane_name}: the qualified control set already carries {collisions}; '
                         'the added blocks must not collide with a lane block key')
    if tokenisation:
        if declared.tokenisation_block is None:
            raise ValueError(f'{lane_name}: this lane declares no tokenisation nuisance block')
        if declared.tokenisation_block in qualified:
            raise ValueError(f'{lane_name}: the qualified control set already carries '
                             f'{declared.tokenisation_block!r}')
    nuisance = tuple(name for name in qualified if name not in declared.profile_blocks)
    if tokenisation:
        nuisance += (declared.tokenisation_block,)
    if not nuisance:
        raise ValueError(f'{lane_name}: the nuisance rung would be empty, so the first step of '
                         'the hierarchy could not be an increment')
    profile = tuple(name for name in qualified if name in declared.profile_blocks)
    profile += tuple(name for name in declared.profile_blocks if name not in qualified)
    return {'nuisance': nuisance, 'profile': profile,
            'couplings': (COUPLINGS_BLOCK,), 'gemme': (GEMME_BLOCK,)}


def control_blocks(lane_name: str, rung: str, qualified: Sequence[str], *,
                   tokenisation: bool = False) -> tuple[str, ...]:
    """One rung's control columns, in the archived order with the added blocks appended.

    The profile rung reproduces the lane's archived primary control set column for
    column, which is what lets the archived control and joint fits be reused under the
    lane's own reproduction pilot; every other rung is a new fit.
    """

    declared = lane(lane_name)
    if rung not in RUNGS:
        raise ValueError(f'unknown rung {rung!r}; the hierarchy is {list(RUNGS)}')
    if rung in declared.unidentifiable_rungs:
        raise ValueError(f'{lane_name}: rung {rung!r} carries a role this lane cannot identify '
                         f'({sorted(set(RUNGS[rung]) & set(declared.non_identifiable))}); it is '
                         'recorded as non-identifiable and never estimated')
    roles = RUNGS[rung]
    blocks = role_blocks(lane_name, qualified, tokenisation=tokenisation)
    if 'profile' not in roles:
        columns = blocks['nuisance']
    else:
        # Keep the archived interleaving: a profile key the archive carries stays where the
        # archive put it, so this rung is the archived design matrix column for column, and
        # a profile key the archive dropped is appended rather than inserted.
        columns = tuple(qualified)
        if tokenisation:
            columns += (declared.tokenisation_block,)
        columns += tuple(name for name in declared.profile_blocks if name not in qualified)
    for added in ('couplings', 'gemme'):
        if added in roles:
            columns += blocks[added]
    if len(set(columns)) != len(columns):
        raise ValueError(f'{lane_name}: rung {rung!r} repeats a block: {columns}')
    return columns


def arm_designs(lane_name: str, rung: str, qualified: Sequence[str], *,
                channel: str = 'likelihood', tokenisation: bool = False) -> dict[str, tuple[str, ...]]:
    """The three arms of one rung: the control, the pure model readout and their union."""

    block = channel_block(lane_name, channel)
    columns = control_blocks(lane_name, rung, qualified, tokenisation=tokenisation)
    if block in columns:
        raise ValueError(f'{lane_name}: the control set already carries the {channel} block '
                         f'{block!r}, so the reverse direction would not read a pure readout')
    return {CONTROL_ARM: columns, MODEL_ARM: (block,), JOINT_ARM: (*columns, block)}


#: Why an archived fit is a reproduction gate here and not saved compute.
ARCHIVED_REUSE_RULE = (
    'the profile rung reproduces the lane\'s archived primary control set column for column, '
    'but the complete case drops the rows whose reference carries no coupling or GEMME value, so '
    'the archived fit sits on a different row set and is not reusable inside the complete-case '
    'family however exactly its columns match. The archived arms are instead refitted on the '
    'full support and required to reproduce the published values: that is the only check '
    'available on the whole fitting path before any added feature exists, and it is a pilot gate '
    'rather than compute saved')


def fit_plan(lane_name: str, qualified: Sequence[str], *, channel: str = 'likelihood',
             tokenisation: bool = False) -> list[dict[str, Any]]:
    """Every design the lane's fitter must produce, with the support it is fitted on.

    Three supports, three roles. The hierarchy is fitted on the complete-case rows,
    where every block of the hierarchy is available, and is the primary family. The
    nuisance rung needs neither added block, so it is also fitted on the full support
    and reported beside the hierarchy as a labelled sensitivity rather than inside its
    family. The archived rung is refitted on the full support as the pilot, for the
    reason :data:`ARCHIVED_REUSE_RULE` gives; it is not a family and is never read as
    one.
    """

    declared = lane(lane_name)
    plan = []
    for support, rungs, role in (('complete_case', declared.rungs, 'primary'),
                                 ('full', FULL_SUPPORT_RUNGS[:1], 'sensitivity'),
                                 ('full', FULL_SUPPORT_RUNGS[1:], 'pilot')):
        for rung in rungs:
            designs = arm_designs(lane_name, rung, qualified, channel=channel,
                                  tokenisation=tokenisation)
            plan.append({'support': support, 'rung': rung, 'channel': channel, 'role': role,
                         'reproduces_archived_control_set': rung == ARCHIVED_RUNG,
                         'compared_against_the_archive': role == 'pilot',
                         'reuses_archived_fit': False,
                         'designs': {name: list(columns) for name, columns in designs.items()}})
    return plan


#: What a fitted cell is bound to, and why that is narrower than this module's bytes.
FIT_IDENTITY_RULE = (
    'a cell is bound to the declared things that can change its own numbers, not to this '
    'module\'s bytes. Recording a limitation, a reason or an instrument bound cannot move '
    'any computed value, and a digest that cannot tell that apart makes writing one cost a '
    'refit -- which prices honesty backwards. The identity is also conditioned on the '
    'support: no full-support rung carries an added block, so the added-block vocabulary, '
    'rendering and admission rules are outside a full-support cell\'s identity and inside a '
    'complete-case cell\'s. Everything that decides which rows are admitted, which columns '
    'a rung carries, how an added block is rendered or which arms form a family is in it '
    'and does invalidate')


def fit_identity(lane_name: str, support: str) -> dict[str, Any]:
    """Every declared thing that can change one cell's numbers on one support.

    Names, letters, signs and numbers only: no prose, no reason and no limitation, for
    the reason :data:`FIT_IDENTITY_RULE` gives. The caller digests this and binds the
    cell to it.

    The residual this does not catch is named rather than hidden: a change to the *code*
    of this module's fitting-relevant helpers -- :func:`role_blocks`,
    :func:`control_blocks`, :func:`arm_designs`, :func:`prepare_scalar`,
    :func:`complete_case` -- is not a change to any field here, so a caller that wants
    that covered binds the module's bytes beside this and accepts the prose tax on them.
    """

    if support not in ('full', 'complete_case'):
        raise ValueError(f'unknown support {support!r}; a cell is fitted on the full support '
                         'or on the complete case')
    declared = lane(lane_name)
    identity: dict[str, Any] = {
        'lane': lane_name, 'support': support,
        'arms': list(ARM_NAMES),
        'channel_blocks': {name: CHANNELS[name]['block'] for name in declared.channels},
        'hierarchy': {name: list(roles) for name, roles in RUNGS.items()},
        'archived_rung': ARCHIVED_RUNG,
        'full_support_rungs': list(FULL_SUPPORT_RUNGS),
        'profile_blocks': list(declared.profile_blocks),
        'tokenisation_block': declared.tokenisation_block,
        'non_identifiable_roles': sorted(declared.non_identifiable),
        'strata_to_loss': {name: value.to_loss for name, value in declared.strata.items()},
        'panel_arms': list(declared.arms),
        'rows_schema': ROWS_SCHEMA,
    }
    if support == 'complete_case':
        identity['added_blocks'] = dict(NEW_BLOCKS)
        identity['row_status'] = {role: dict(vocabulary)
                                  for role, vocabulary in ROW_STATUS.items()}
        identity['block_source'] = {role: {'schema': source['schema'], 'column': source['column']}
                                    for role, source in BLOCK_SOURCE.items()}
        identity['added_block_rendering'] = 'one column each, reranked within assay'
        identity['cycle_contrast'] = CYCLE_CONTRAST['definition']
        identity['refusal_authority'] = 'block table or cohort export'
        identity['refusals_schema'] = REFUSALS_SCHEMA
    return identity


def prepare_scalar(values, assays) -> np.ndarray:
    """One added block: a single column, reranked within assay, as the R1 ladder enters it.

    The R1 ladder reranks the coupling and GEMME scalars within each retained assay on
    the frozen support; the same rendering is used here so the lanes and the ladder read
    one definition of how an explicit statistic enters a design.
    """

    values = np.asarray(values, dtype=float)
    assays = np.asarray(assays)
    if values.ndim != 1 or values.shape != assays.shape:
        raise ValueError('an added block is one value per row of the support')
    if not np.isfinite(values).all():
        raise ValueError('an added block is finite on the support it is fitted on; a refused '
                         'reference leaves the complete case rather than taking a substitute')
    return overlap.rerank(values, assays).reshape(-1, 1)


def read_rows(path) -> list[dict[str, Any]]:
    """The cohort's variant table: the row identities the lane's support is drawn from."""

    document = json.loads(Path(path).read_text())
    if document.get('schema') != ROWS_SCHEMA:
        raise ValueError(f'{path}: schema {document.get("schema")!r} is not {ROWS_SCHEMA!r}')
    rows = document.get('samples')
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{path} holds no sample rows')
    return rows


def read_refusals(path) -> dict[str, Any]:
    """The cohort export's own record of the references it refused to score.

    The export refuses a reference with too few retained homolog rows once, for both
    blocks at once, and this document is the single statement of that fact.
    """

    document = json.loads(Path(path).read_text())
    if document.get('schema') != REFUSALS_SCHEMA:
        raise ValueError(f'{path}: schema {document.get("schema")!r} is not '
                         f'{REFUSALS_SCHEMA!r}')
    refused = document.get('refused')
    if not isinstance(refused, dict):
        raise ValueError(f'{path} enumerates no refusals; an export that refused nothing '
                         'still declares an empty map rather than omitting one')
    if document.get('count') != len(refused):
        raise ValueError(f'{path} declares {document.get("count")!r} refusals and enumerates '
                         f'{len(refused)}')
    return dict(document, read_from=str(path))


def gemme_risk_band(support, *, band: tuple[int, int] = GEMME_RISK_BAND) -> dict[str, Any]:
    """References shallow enough for the official GEMME run to come back degenerate.

    ``support`` is the cohort's exported support document. The band is read off the
    retained row counts, so it is a property of the export rather than a number kept
    in this module; a reference inside it may still score normally, and one outside it
    has never been observed to degenerate.
    """

    if support.get('schema') != UPSTREAM_SUPPORT_SCHEMA:
        raise ValueError(f'support schema {support.get("schema")!r} is not '
                         f'{UPSTREAM_SUPPORT_SCHEMA!r}')
    low, high = band
    references = support.get('wildtypes')
    if not isinstance(references, dict) or not references:
        raise ValueError('the support document lists no references')
    inside = {}
    for name, entry in sorted(references.items()):
        rows = entry.get('rows')
        if not isinstance(rows, int):
            raise ValueError(f'{name}: the exported support records no retained row count')
        if low <= rows <= high:
            inside[name] = rows
    return {'cohort': support.get('cohort'), 'band': [low, high], 'references': len(references),
            'in_band': len(inside), 'rows_by_reference': inside,
            'rule': (f'every GEMME degeneracy observed so far had at most {high} retained '
                     f'alignment rows and none at {high + 1} or more; {low} rows is the floor '
                     'below which no alignment is retained at all'),
            'reading': ('a risk band, not a prediction: a reference inside it may score '
                        'normally, and a degenerate run is recorded as a refusal when it '
                        'happens rather than anticipated here'),
            'alignment_reuse_note': ('a reference whose sequence is byte-identical to a fitted '
                                     'R1 wild type takes R1\'s frozen alignment, which does not '
                                     'exempt it: GEMME is still re-run for this cohort\'s own '
                                     'mutation list and can still come back degenerate')}


def read_scores(path, role: str) -> dict[str, Any]:
    """One upstream score table, with its completion and its per-row status verified.

    A missing file, a wrong schema, an incomplete run, a pending row or an unclassified
    status is refused here. Nothing is imputed and no table is read past a refusal: the
    features arrive as their own wave and an absent wave is an absent wave.
    """

    if role not in BLOCK_SOURCE:
        raise ValueError(f'unknown added block role {role!r}')
    source = BLOCK_SOURCE[role]
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f'{role}: the score table {path} is absent; the {source["producer"]} wave has not '
            'landed for this cohort and no lane panel may be written without it')
    document = json.loads(path.read_text())
    if document.get('schema') != source['schema']:
        raise ValueError(f'{path}: schema {document.get("schema")!r} is not {source["schema"]!r}')
    counts = document.get('counts')
    if not isinstance(counts, dict):
        raise ValueError(f'{path}: the document declares no counts, so its completion cannot '
                         'be verified')
    if counts.get('complete') is not True:
        raise ValueError(f'{role}: {path} declares counts.complete={counts.get("complete")!r}; an '
                         'incomplete run is never read as a control')
    rows = document.get('samples')
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{path} holds no sample rows')
    return dict(document, read_from=str(path))


def _classify(rows: Sequence[Mapping[str, Any]], role: str, path: str) -> dict[str, Any]:
    """Per-row availability class of one score table, refusing anything unclassified."""

    source = BLOCK_SOURCE[role]
    vocabulary = ROW_STATUS[role]
    classes, unknown, pending, inconsistent = {}, {}, {}, []
    for row in rows:
        identifier = row.get('sample_id')
        if not identifier:
            raise ValueError(f'{path}: a score row carries no sample_id')
        if identifier in classes:
            raise ValueError(f'{path}: duplicate sample_id {identifier!r}')
        status = str(row.get('status'))
        state = vocabulary.get(status)
        if state is None:
            unknown[status] = unknown.get(status, 0) + 1
            classes[identifier] = None
            continue
        value = row.get(source['column'])
        try:
            finite = value is not None and bool(np.isfinite(float(value)))
        except (TypeError, ValueError) as error:
            raise ValueError(f'{path}: {identifier} carries a non-numeric '
                             f'{source["column"]} value {value!r}') from error
        if (state == AVAILABLE) != finite:
            inconsistent.append(identifier)
        if state == PENDING:
            pending[status] = pending.get(status, 0) + 1
        classes[identifier] = state
    if unknown:
        raise ValueError(
            f'{role}: {path} carries {sum(unknown.values())} rows whose status is neither an '
            f'available, a declared refusal nor a pending row: {unknown}. A failed or '
            'unconverged background and a partly scored wild type are not enumerable losses; '
            'they are resolved upstream, never classified here')
    if inconsistent:
        raise ValueError(f'{role}: {path} disagrees with itself on {len(inconsistent)} rows: a '
                         f'status and a {source["column"]} value that do not match, first '
                         f'{inconsistent[0]!r}')
    if pending:
        raise ValueError(f'{role}: {path} carries {sum(pending.values())} pending rows {pending}; '
                         'the wave is still running and a partial table is not a support')
    return classes


def complete_case(lane_name: str, rows: Sequence[Mapping[str, Any]], tables: Mapping[str, Any], *,
                  refusals: Mapping[str, Any] | None = None,
                  groups: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The rows every control set of one stratum is fitted on, and what was lost.

    ``tables`` maps an added-block role to the document :func:`read_scores` returned.
    A role this lane declares non-identifiable must not be supplied: it does not
    restrict the support, because the lane's hierarchy never carries it.

    ``refusals`` is the cohort export's own refusals document, read by
    :func:`read_refusals`. A lost reference is accepted as a declared loss when either
    the block table or the export enumerates it, for the reason
    :data:`EXPORT_REFUSAL_AUTHORITY` gives. Passing nothing leaves the block table as the
    only authority, so an export-refused reference the table does not re-enumerate is
    still refused: an absent document narrows what may be accepted and never widens it.

    Every loss is enumerated by reference, by count and by the upstream reason, and the
    groups a loss empties are named, because a lane that drops under the resampling
    floor is a finding and not an accident.
    """

    declared = lane(lane_name)
    required = declared.required_blocks()
    missing = [role for role in required if role not in tables]
    if missing:
        raise ValueError(f'{lane_name}: the complete case needs the {missing} score tables; '
                         'every control set is fitted on the rows where every block of this '
                         "lane's hierarchy is available")
    forbidden = sorted(set(tables) - set(required))
    if forbidden:
        raise ValueError(f'{lane_name}: {forbidden} is declared non-identifiable for this '
                         'estimand, so it must not restrict the support either')
    identifiers = [str(row['sample_id']) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f'{lane_name}: the cohort rows carry duplicate sample_id values')
    reference_of = {str(row['sample_id']): str(row['wildtype_id']) for row in rows}
    classes: dict[str, dict[str, str]] = {}
    losses: dict[str, Any] = {}
    for role in required:
        document = tables[role]
        source = document.get('read_from', role)
        classes[role] = _classify(document['samples'], role, source)
        unseen = [identifier for identifier in identifiers if identifier not in classes[role]]
        extra = sorted(set(classes[role]) - set(identifiers))
        if unseen or extra:
            raise ValueError(f'{role}: {source} does not cover the cohort rows: '
                             f'{len(unseen)} missing, {len(extra)} extra sample_id values')
        refused = document.get('refused') or {}
        exported = (refusals or {}).get('refused') or {}
        lost_rows = [identifier for identifier in identifiers
                     if classes[role][identifier] == REFUSED]
        lost_here = {reference_of[identifier] for identifier in lost_rows}
        orphan = sorted(lost_here - set(refused) - set(exported))
        if orphan:
            raise ValueError(
                f'{role}: {source} refuses rows of {len(orphan)} references that neither it nor '
                f'the cohort export enumerates as refusals, first {orphan[0]!r}. A loss is '
                'declared per reference, which is what keeps a multi-corner cycle from losing '
                'one corner on its own')
        enumerated = {}
        for name in sorted(lost_here):
            if name in refused:
                enumerated[name] = dict(refused[name], declared_by='block table')
            else:
                enumerated[name] = dict(exported[name], declared_by='cohort export')
        losses[role] = {
            'read_from': source, 'references': enumerated,
            'reference_count': len(enumerated), 'rows': len(lost_rows),
            'declared_by': {'block table': sum(1 for detail in enumerated.values()
                                               if detail['declared_by'] == 'block table'),
                            'cohort export': sum(1 for detail in enumerated.values()
                                                 if detail['declared_by'] == 'cohort export')}}
    kept, lost = [], []
    for identifier in identifiers:
        available = all(classes[role][identifier] == AVAILABLE for role in required)
        (kept if available else lost).append(identifier)
    lost_references = sorted({reference_of[identifier] for identifier in lost}
                             - {reference_of[identifier] for identifier in kept})
    record = {
        'schema': SUPPORT_SCHEMA, 'lane': lane_name, 'cohort': declared.cohort,
        'required_blocks': list(required),
        'non_identifiable': dict(declared.non_identifiable),
        'rule': ('one support per stratum: the rows where every block of this hierarchy is '
                 'available. A nested increment computed on shifting rows means nothing, so no '
                 'control set is fitted on rows another control set does not have'),
        'rows_offered': len(identifiers), 'rows_retained': len(kept), 'rows_lost': len(lost),
        'references_offered': len(set(reference_of.values())),
        'references_lost': lost_references,
        'refusal_authority': EXPORT_REFUSAL_AUTHORITY,
        'refusals_read_from': None if refusals is None else refusals.get('read_from'),
        'losses': losses,
        'sample_ids': kept,
        'support_sha256': identity_digest(kept),
        'full_support_sha256': identity_digest(identifiers),
        'full_support_sensitivity': ('the nuisance rung needs neither added block and is also '
                                     'fitted on the full support; it is reported beside the '
                                     'hierarchy and never inside its family'),
    }
    if groups is not None:
        offered = {str(identifier): str(groups[str(identifier)]) for identifier in identifiers}
        retained = {offered[identifier] for identifier in kept}
        emptied = sorted(set(offered.values()) - retained)
        floor = bootstrap_unit_floor(len(retained))
        record['groups'] = {'offered': len(set(offered.values())), 'retained': len(retained),
                            'emptied': emptied, 'floor': floor}
        if floor['degenerate']:
            record['groups']['status'] = UNRESOLVED
    return record


def declared_family(lane_name: str, channel: str, *, role: str = 'primary') -> dict[str, Any]:
    """The multiplicity family of one lane and channel: control sets x arms x directions."""

    declared = lane(lane_name)
    if role not in ('primary', 'sensitivity'):
        raise ValueError(f'unknown family role {role!r}')
    channel_block(lane_name, channel)
    rungs = declared.rungs if role == 'primary' else ('nuisance',)
    size = _family_size(len(rungs), len(declared.arms))
    if role == 'primary' and size != declared.declared_family[channel]:
        raise ValueError(f'{lane_name}: the realised {channel} family is {size} columns and the '
                         f'declaration says {declared.declared_family[channel]}')
    record = {
        'lane': lane_name, 'channel': channel, 'channel_role': CHANNELS[channel]['role'],
        'role': role, 'size': size, 'control_sets': list(rungs),
        'arms': list(declared.arms), 'directions': list(bidirectional.DIRECTIONS),
        'composition': (f'{len(rungs)} control sets x {len(declared.arms)} arms x '
                        f'{len(bidirectional.DIRECTIONS)} directions'),
        'declared_per': 'one family per lane, stratum and channel; lanes and channels are never '
                        'pooled, because their estimands and their units differ',
    }
    if declared.arms != declared.panel_arms:
        full = _family_size(len(rungs), len(declared.panel_arms))
        record['partial_panel'] = {
            'scored_arms': len(declared.arms), 'panel_arms': len(declared.panel_arms),
            'family_at_full_panel': full,
            'note': (f'{len(declared.arms)} of {len(declared.panel_arms)} arms are scored, so '
                     f'this family is {size} columns and becomes {full} when the queued arms '
                     'land; the declaration is amended then, not inferred at runtime')}
        if role == 'primary' and full != declared.full_panel_family[channel]:
            raise ValueError(f'{lane_name}: the full-panel {channel} family would be {full} '
                             f'columns and the declaration says '
                             f'{declared.full_panel_family[channel]}')
    if declared.unidentifiable_rungs:
        record['non_identifiable'] = {
            'rungs': list(declared.unidentifiable_rungs),
            'roles': dict(declared.non_identifiable),
            'note': 'recorded, never estimated; the rungs are outside the family'}
    return record


def read_cells(path) -> dict[str, Any]:
    """One lane runner's per-group held-out errors, as the panel reads them."""

    document = json.loads(Path(path).read_text())
    if document.get('schema') not in CELLS_SCHEMAS:
        raise ValueError(f'{path}: schema {document.get("schema")!r} is not one of '
                         f'{CELLS_SCHEMAS}')
    for key in ('lane', 'stratum', 'channel', 'support_sha256', 'groups', 'cells'):
        if key not in document:
            raise ValueError(f'{path}: a cells document declares {key}')
    return document


def _matrix(lane_name: str, stratum_name: str, cells: Mapping[tuple[str, str], Mapping[str, Any]],
            rungs: Sequence[str], *, groups: Sequence[Any], support_sha256: str):
    """Groups-by-columns reductions of every (rung, arm) cell, in declared order."""

    declared = lane(lane_name)
    sign = stratum(lane_name, stratum_name).to_loss
    expected = [(rung, arm) for rung in rungs for arm in declared.arms]
    unknown = sorted(set(cells) - set(expected))
    absent = [key for key in expected if key not in cells]
    if unknown or absent:
        raise ValueError(f'{lane_name}/{stratum_name}: the family is {len(expected)} cells of '
                         f'(control set, arm); {len(absent)} are absent and {len(unknown)} are '
                         f'undeclared (first absent {absent[:1]}, first undeclared {unknown[:1]})')
    columns, labels, control, model = [], [], {}, {}
    for rung, arm in expected:
        cell = cells[rung, arm]
        designs = {name: cell[name] for name in ARM_NAMES if name in cell}
        if set(designs) != set(ARM_NAMES):
            raise ValueError(f'{lane_name}/{stratum_name}/{rung}/{arm}: a cell carries the three '
                             f'arms {list(ARM_NAMES)}; it is missing '
                             f'{[name for name in ARM_NAMES if name not in designs]}')
        digest = cell.get('support_sha256')
        if digest is not None and digest != support_sha256:
            raise ValueError(f'{lane_name}/{stratum_name}/{rung}/{arm}: this cell was fitted on '
                             'support ' + str(digest)[:12] + ' and the stratum declares ' +
                             support_sha256[:12] + '; a control set is never compared against '
                             'anything on unmatched support')
        errors = {}
        for name in ARM_NAMES:
            values = np.asarray(designs[name], dtype=float)
            if values.shape != (len(groups),) or not np.isfinite(values).all():
                raise ValueError(f'{lane_name}/{stratum_name}/{rung}/{arm}/{name}: a cell carries '
                                 f'one finite value for each of the {len(groups)} groups')
            errors[name] = sign * values
        # Each reduced arm's own performance, in the stratum's native direction and on
        # these same groups: the denominators the two bounds are read against. The
        # forward bound divides by the control arm's, the reverse by the model arm's.
        control[rung, arm] = float(np.mean(np.asarray(designs[CONTROL_ARM], dtype=float)))
        model[rung, arm] = float(np.mean(np.asarray(designs[MODEL_ARM], dtype=float)))
        columns.append(bidirectional.directional_reductions(errors, bidirectional.PRIMARY_DESIGNS))
        labels.extend((rung, arm, direction) for direction in bidirectional.DIRECTIONS)
    return np.concatenate(columns, axis=1), labels, control, model


def control_set_label(lane_name: str, rung: str,
                      columns: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """What one rung actually is, in a form that cannot be read as a different rung.

    Every number this module reports is an increment over *one* rung, and the rungs
    differ by exactly the blocks whose value is in question. A label that names the
    rung's role in the run -- "sensitivity", "primary" -- leaves the reader to find the
    rung in another field, and a reader who does not will attribute a nuisance-rung
    increment to the archived control set. So the rung's name and its literal control
    columns travel with the number.
    """

    lane(lane_name)  # refuses an undeclared lane before anything is labelled with it
    if rung not in RUNGS:
        raise ValueError(f'unknown rung {rung!r}; the hierarchy is {list(RUNGS)}')
    record: dict[str, Any] = {
        'control_set': rung, 'roles': list(RUNGS[rung]),
        'is_archived_control_set': rung == ARCHIVED_RUNG,
        'carries_profile': 'profile' in RUNGS[rung],
        'carries_added_blocks': sorted(set(RUNGS[rung]) & set(NEW_BLOCKS)),
    }
    if columns is None:
        record['columns'] = None
        record['reads_as'] = (f'an increment over the {rung!r} rung of the {lane_name} lane; its '
                              'literal control columns were not supplied to this panel and are '
                              "recorded per arm in each cell's own designs")
        return record
    distinct: dict[tuple[str, ...], list[str]] = {}
    for arm, names in sorted(columns.items()):
        distinct.setdefault(tuple(names), []).append(arm)
    record['columns'] = {'/'.join(names): arms for names, arms in distinct.items()}
    shown = ', '.join(sorted(distinct, key=len)[0])
    if len(distinct) > 1:
        extra = sorted({name for names in distinct for name in names} - set(min(distinct, key=len)))
        record['arm_conditional_columns'] = {'added_by_some_arms': extra,
                                             'columns_per_arm_set': record['columns']}
        shown += f' (and {", ".join(extra)} for the arms whose archive qualified them)'
    note = '' if rung == ARCHIVED_RUNG else (
        f'. This is NOT the lane\'s archived control set, which is the {ARCHIVED_RUNG!r} rung')
    record['reads_as'] = (f'an increment over the {rung!r} rung of the {lane_name} lane, whose '
                          f'control columns are {shown}{note}')
    return record


def forward_bound(lane_name: str, stratum_name: str, labels, interval, control,
                  columns: Mapping[str, Mapping[str, Sequence[str]]] | None = None) -> dict[str, Any]:
    """The forward increment's simultaneous upper limit, in native units and as a fraction.

    This is the quantity the manuscript's overlap claim rests on, so it is reported as
    a first-class result rather than left to be read off an interval. For every forward
    column it gives the simultaneous upper limit in the stratum's own units and that
    limit as a fraction of the control set's own performance on the identical support:
    *the model adds at most this much beyond the explicit statistics, which is at most
    this fraction of what those statistics already achieve*. The fraction is the
    interpretable form because it needs no threshold and no unit.

    The denominator is the control arm's own group-equal mean in the stratum's native
    direction -- an error level where the stratum reads an error, a rank agreement where
    it reads one. For an error stratum that is the error the statistics *leave*, not
    their reduction against any null, because this module is handed no null arm; where
    the denominator is not a positive performance the fraction is withheld with its
    reason rather than rendered as a number.
    """

    reading = stratum(lane_name, stratum_name)
    native = np.asarray(interval, dtype=float)
    if native.shape != (len(labels), 2):
        raise ValueError('a forward bound needs one interval per declared column')
    sense = ('adding the model removes at most this fraction of the error the explicit '
             'statistics leave on these groups' if reading.to_loss > 0 else
             'adding the model adds at most this fraction of the rank agreement the explicit '
             'statistics already achieve on these groups')
    entries, summary = [], {}
    for position, (rung, arm, direction) in enumerate(labels):
        if direction != 'model_given_controls':
            continue
        limit = float(native[position, 1])
        denominator = control[rung, arm]
        fraction, withheld = None, None
        if denominator > 0:
            fraction = limit / denominator
        else:
            withheld = (f'the control arm\'s own performance on these groups is {denominator:g}, '
                        'which is not a positive performance to take a fraction of')
        # A ratio above one is still a true ratio but has stopped being a share of
        # anything: it says the bound on the increment exceeds the performance it is
        # divided by. Flagged rather than withheld, so the native limit stays primary.
        exceeds = fraction is not None and fraction > 1
        entries.append({'control_set': rung, 'arm': arm, 'upper_limit': limit,
                        'units': reading.units, 'control_performance': denominator,
                        'fraction_of_control_performance': fraction,
                        'fraction_withheld_because': withheld,
                        'fraction_exceeds_denominator': exceeds})
        held = summary.setdefault(rung, {'largest_upper_limit': limit,
                                         'largest_fraction_of_control_performance': fraction,
                                         'arms': 0, 'arms_without_a_fraction': 0,
                                         'arms_whose_fraction_exceeds_one': 0})
        held['arms'] += 1
        held['arms_whose_fraction_exceeds_one'] += int(exceeds)
        held['largest_upper_limit'] = max(held['largest_upper_limit'], limit)
        if fraction is None:
            held['arms_without_a_fraction'] += 1
        elif held['largest_fraction_of_control_performance'] is None:
            held['largest_fraction_of_control_performance'] = fraction
        else:
            held['largest_fraction_of_control_performance'] = max(
                held['largest_fraction_of_control_performance'], fraction)
    for rung, held in summary.items():
        held.update(control_set_label(lane_name, rung, (columns or {}).get(rung)))
    return {'direction': 'model_given_controls', 'reading': FORWARD_BOUND_READING,
            'denominator': ('the control arm\'s own group-equal mean in the stratum\'s native '
                            'direction, on the identical complete-case support'),
            'fraction_reading': sense,
            'limitation': ('an upper limit is not an equivalence statement and no margin is '
                           'declared; for an error stratum the denominator is the error the '
                           'controls leave, not their reduction against a null arm'),
            'fraction_limitation': ('the fraction is interpretable only while this rung\'s own '
                                    'performance is a meaningful denominator. Where the rung '
                                    'itself achieves almost nothing the ratio exceeds one and '
                                    'stops being a share; such columns are flagged and read in '
                                    'native units instead'),
            'columns': entries, 'per_control_set': summary}


#: What the reverse direction's simultaneous upper limit is, and when it matters.
REVERSE_BOUND_READING = (
    'the reverse increment is what separates shared information from model inferiority, '
    'so it needs both of its limits. Its simultaneous *lower* limit above zero resolves '
    'that the explicit statistics carry predictive information the model readout does '
    'not. Its simultaneous *upper* limit is the second half of a two-sided statement -- '
    'the statistics add at most this much beyond the pure model readout, and at most '
    'this fraction of what that readout already achieves. Overlap is a two-sided claim '
    'and needs both bounds small; a reverse increment resolved above zero and large is '
    'containment with resolved control superiority, not overlap')


def reverse_bound(lane_name: str, stratum_name: str, labels, interval, model,
                  columns: Mapping[str, Mapping[str, Sequence[str]]] | None = None) -> dict[str, Any]:
    """The reverse increment's simultaneous limits, in native units and as a fraction.

    The mirror of :func:`forward_bound`, against the *model* arm's own performance
    rather than the control set's, because the quantity being bounded is what the
    statistics add to the pure model readout. Both limits are reported: overlap is a
    two-sided statement and cannot be read off a lower limit alone.

    The band is simultaneous over the whole declared family and both directions share
    one critical value, so a limit here holds jointly with every other column's.
    """

    reading = stratum(lane_name, stratum_name)
    native = np.asarray(interval, dtype=float)
    if native.shape != (len(labels), 2):
        raise ValueError('a reverse bound needs one interval per declared column')
    sense = ('adding the explicit statistics removes at most this fraction of the error the '
             'model readout leaves on these groups' if reading.to_loss > 0 else
             'adding the explicit statistics adds at most this fraction of the rank agreement '
             'the model readout already achieves on these groups')
    entries, summary = [], {}
    for position, (rung, arm, direction) in enumerate(labels):
        if direction != 'controls_given_model':
            continue
        low, high = float(native[position, 0]), float(native[position, 1])
        denominator = model[rung, arm]
        fraction, withheld = None, None
        if denominator > 0:
            fraction = high / denominator
        else:
            withheld = (f'the model arm\'s own performance on these groups is {denominator:g}, '
                        'which is not a positive performance to take a fraction of')
        # A ratio above one is still a true ratio but has stopped being a share of
        # anything: it says the increment exceeds the performance it is divided by,
        # which happens when the model readout alone achieves almost nothing. Flagged
        # rather than withheld, so the native limit stays the primary statement.
        exceeds = fraction is not None and fraction > 1
        entries.append({'control_set': rung, 'arm': arm, 'lower_limit': low, 'upper_limit': high,
                        'units': reading.units, 'model_performance': denominator,
                        'fraction_of_model_performance': fraction,
                        'fraction_withheld_because': withheld,
                        'fraction_exceeds_denominator': exceeds,
                        'resolved_above_zero': low > 0})
        held = summary.setdefault(rung, {'largest_upper_limit': high,
                                         'largest_fraction_of_model_performance': fraction,
                                         'smallest_lower_limit': low, 'arms': 0,
                                         'arms_resolved_above_zero': 0,
                                         'arms_without_a_fraction': 0,
                                         'arms_whose_fraction_exceeds_one': 0})
        held['arms'] += 1
        held['arms_whose_fraction_exceeds_one'] += int(exceeds)
        held['largest_upper_limit'] = max(held['largest_upper_limit'], high)
        held['smallest_lower_limit'] = min(held['smallest_lower_limit'], low)
        held['arms_resolved_above_zero'] += int(low > 0)
        if fraction is None:
            held['arms_without_a_fraction'] += 1
        elif held['largest_fraction_of_model_performance'] is None:
            held['largest_fraction_of_model_performance'] = fraction
        else:
            held['largest_fraction_of_model_performance'] = max(
                held['largest_fraction_of_model_performance'], fraction)
    for rung, held in summary.items():
        held.update(control_set_label(lane_name, rung, (columns or {}).get(rung)))
    return {'direction': 'controls_given_model', 'reading': REVERSE_BOUND_READING,
            'denominator': ('the model arm\'s own group-equal mean in the stratum\'s native '
                            'direction, on the identical complete-case support'),
            'fraction_reading': sense,
            'two_sided': ('both limits are reported because overlap needs both small; a lower '
                          'limit above zero on its own resolves control superiority'),
            'columns_note': 'every entry names the rung it is an increment over',
            'fraction_limitation': ('the fraction is interpretable only while the model arm\'s own '
                                    'performance is a meaningful denominator. Where that arm alone '
                                    'achieves almost nothing -- a near-zero rank agreement, say -- '
                                    'the ratio exceeds one and stops being a share; such columns '
                                    'are flagged and read in native units instead'),
            'columns': entries, 'per_control_set': summary}


def panel(lane_name: str, stratum_name: str, channel: str,
          cells: Mapping[tuple[str, str], Mapping[str, Any]], *, groups: Sequence[Any],
          support_sha256: str, role: str = 'primary',
          control_columns: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
          draws: int = bidirectional.DRAWS,
          seed: int = bidirectional.SEED) -> dict[str, Any]:
    """The declared family of one lane, stratum and channel, with both directions.

    ``control_columns`` maps each rung to its literal control columns per arm, so that
    every number this returns carries the rung it is an increment over. Supplying it is
    how a reader is kept from attributing a nuisance-rung increment to the archived
    control set; without it the labels say so rather than guessing.

    Every column is a paired per-group reduction of the same joint arm, positive when
    the joint arm lowers the loss. The primary band is the shared native maximum
    absolute centered deviation; the project's fixed-standard-error studentised family
    over the same draws is retained as an explicitly labelled sensitivity. A stratum
    under the bootstrap-unit floor is unresolved rather than estimated.
    """

    declared = lane(lane_name)
    family = declared_family(lane_name, channel, role=role)
    rungs = declared.rungs if role == 'primary' else ('nuisance',)
    groups = [value.item() if isinstance(value, np.generic) else value for value in groups]
    if len(set(map(str, groups))) != len(groups):
        raise ValueError(f'{lane_name}/{stratum_name}: the resampling units repeat a label')
    matrix, labels, control, model = _matrix(lane_name, stratum_name, cells, rungs,
                                             groups=groups, support_sha256=support_sha256)
    if matrix.shape != (len(groups), family['size']):
        raise ValueError(f'{lane_name}/{stratum_name}: the realised family is '
                         f'{matrix.shape[1]} columns over {matrix.shape[0]} groups and the '
                         f'declaration says {family["size"]} columns')
    record = {
        'schema': PANEL_SCHEMA, 'lane': lane_name, 'stratum': stratum_name, 'channel': channel,
        'role': role, 'estimand': declared.estimand,
        'units': stratum(lane_name, stratum_name).units,
        'resampling_unit': declared.unit, 'groups': len(groups),
        'stratum_fragile': stratum(lane_name, stratum_name).fragile,
        'support': 'complete_case' if role == 'primary' else 'full',
        'support_sha256': support_sha256, 'family': family,
        'control_sets': {rung: control_set_label(lane_name, rung,
                                                 (control_columns or {}).get(rung))
                         for rung in rungs},
        'reads_as': '; '.join(
            control_set_label(lane_name, rung, (control_columns or {}).get(rung))['reads_as']
            for rung in rungs),
        'columns': [{'control_set': rung, 'arm': arm, 'direction': direction}
                    for rung, arm, direction in labels],
        'contrasts': {direction: f'loss({reduced}) - loss({joint})'
                      for direction, (reduced, joint) in bidirectional.PRIMARY_DESIGNS.items()},
        'point': matrix.mean(axis=0).tolist(),
        'control_provenance': CONTROL_PROVENANCE,
        'simultaneity': SIMULTANEITY,
        'state_rule': {SUPPORTED: 'the simultaneous interval lies strictly above zero',
                       BOUNDED_NEGATIVE: 'the simultaneous interval lies strictly below zero',
                       UNRESOLVED: ('the interval contains zero, however narrow, or the stratum '
                                    'is under the resampling floor; not equivalence, and no '
                                    'margin is declared'),
                       NON_IDENTIFIABLE: 'the block cannot inform this estimand at all'},
    }
    floor = bootstrap_unit_floor(len(groups))
    record['independence_floor'] = floor
    if floor['degenerate']:
        record.update(status=UNRESOLVED, native=None, standardized_maxT_sensitivity=None,
                      states=[UNRESOLVED] * len(labels),
                      forward_bound={'status': UNRESOLVED, 'reading': FORWARD_BOUND_READING,
                                     'reason': floor['degenerate_reason']},
                      reverse_bound={'status': UNRESOLVED, 'reading': REVERSE_BOUND_READING,
                                     'reason': floor['degenerate_reason']},
                      reason=floor['degenerate_reason'])
    else:
        native = bidirectional.paired_bands(matrix, draws=draws, seed=seed)
        verdicts = bidirectional.verdict(native['interval'])
        record.update(
            status='complete',
            native=dict(method=native['method'], draws=native['draws'], seed=native['seed'],
                        confidence=native['confidence'], family_size=native['family_size'],
                        joint_critical_max_absolute_deviation=native['critical_value'],
                        interval=native['interval'],
                        conditional_on_fitted_predictions=True),
            standardized_maxT_sensitivity=_sensitivity(matrix, draws=draws, seed=seed),
            states=[VERDICT_STATE[value] for value in verdicts],
            forward_bound=forward_bound(lane_name, stratum_name, labels, native['interval'],
                                        control, control_columns),
            reverse_bound=reverse_bound(lane_name, stratum_name, labels, native['interval'],
                                        model, control_columns))
    unidentifiable = []
    for rung in declared.unidentifiable_rungs:
        blocked = sorted(set(RUNGS[rung]) & set(declared.non_identifiable))
        for arm in declared.arms:
            for direction in bidirectional.DIRECTIONS:
                unidentifiable.append(
                    {'control_set': rung, 'arm': arm, 'direction': direction,
                     'state': NON_IDENTIFIABLE, 'roles': blocked,
                     'reasons': {name: declared.non_identifiable[name] for name in blocked}})
    record['non_identifiable'] = unidentifiable
    record['state_counts'] = state_counts(record['states'],
                                          non_identifiable=len(record['non_identifiable']))
    record['notes'] = list(declared.notes)
    return record


def _sensitivity(matrix, *, draws: int, seed: int) -> dict[str, Any]:
    """The project's fixed-standard-error studentised family, as a labelled sensitivity.

    A column with no spread across the resampling units takes a zero-width studentised
    interval, which would read as a resolved finding; such a column carries no state
    here and is named instead.
    """

    bands = responses.simultaneous_bands(np.asarray(matrix, dtype=float), draws=draws, seed=seed)
    spread = np.std(np.asarray(matrix, dtype=float), axis=0, ddof=1)
    zero = [int(index) for index in np.flatnonzero(spread <= 0)]
    verdicts = bidirectional.verdict(bands['interval'])
    states: list[str | None] = [VERDICT_STATE[value] for value in verdicts]
    for index in zero:
        states[index] = None
    return {'role': 'sensitivity', 'method': bidirectional.STANDARDIZED_METHOD,
            'draws': bands['draws'], 'seed': bands['seed'],
            'family_size': bands['family_size'],
            'joint_critical_standardized': bands['critical_value'],
            'interval': bands['interval'], 'states': states,
            'zero_standard_error_columns': zero,
            'note': ('declared sensitivity only; the native-unit family remains primary, and a '
                     'column with no group spread carries no state because its studentised '
                     'interval has no width')}


def state_counts(states: Sequence[str | None], *, non_identifiable: int = 0) -> dict[str, int]:
    """Tally over the four finding states, with unassigned columns counted apart."""

    values = list(states)
    unknown = sorted({value for value in values if value not in STATES and value is not None})
    if unknown:
        raise ValueError(f'unknown finding states {unknown}')
    counts = {name: values.count(name) for name in STATES}
    counts[NON_IDENTIFIABLE] += int(non_identifiable)
    counts['unassigned'] = sum(1 for value in values if value is None)
    return counts


def contract(lane_name: str) -> dict[str, Any]:
    """One declaration of the hierarchy, roles, directions, families and refusals."""

    declared = lane(lane_name)
    return {
        'schema': CONTRACT_SCHEMA, 'lane': lane_name, 'cohort': declared.cohort,
        'estimand': declared.estimand, 'resampling_unit': declared.unit,
        'roles': list(ROLES),
        'role_vocabulary': ('a role is what a block is for this estimand; the letter that '
                            'carries it is lane-local, because the same letter names different '
                            'blocks in different lanes'),
        'added_blocks': dict(NEW_BLOCKS),
        'added_block_rendering': ('one column each, reranked within assay on the fitted support, '
                                  'exactly as the R1 ladder enters the same two statistics'),
        'profile_blocks': list(declared.profile_blocks),
        'tokenisation_block': declared.tokenisation_block,
        'arm_conditional_control': (
            None if declared.tokenisation_block is None else
            'the nuisance control is arm-conditional by design, as the lane\'s archive made it: '
            f'an arm that qualified {declared.tokenisation_block!r} carries its own tokenisation '
            'descriptors in its control set, because that nuisance has to absorb that arm\'s own '
            'segmentation artefacts. A control held common across arms would credit segmentation '
            'to the model, so the forward increment is read as "the model beyond this arm\'s own '
            'controls" and the qualification is never redone here'),
        'hierarchy': {name: list(roles) for name, roles in RUNGS.items()},
        'identifiable_control_sets': list(declared.rungs),
        'shared_control_sets': list(SHARED_RUNGS),
        'variant_effect_bridge': dict(VARIANT_EFFECT_BRIDGE,
                                      roles={role: list(keys) for role, keys
                                             in VARIANT_EFFECT_ROLES.items()}),
        'cycle_contrast': dict(CYCLE_CONTRAST),
        'coupling_instrument_bounds': dict(COUPLING_INSTRUMENT_BOUNDS),
        'nuisance_rung_limitation': NUISANCE_RUNG_LIMITATION,
        'fit_identity_rule': FIT_IDENTITY_RULE,
        'fit_identity': {support: fit_identity(lane_name, support)
                         for support in ('full', 'complete_case')},
        'refusal_authority': EXPORT_REFUSAL_AUTHORITY,
        'archived_control_set': ARCHIVED_RUNG,
        'archived_control_set_note': ('this rung is the lane\'s archived primary control set, '
                                      'column for column'),
        'archived_reuse_rule': ARCHIVED_REUSE_RULE,
        'arms': {name: list(columns) for name, columns in
                 (('control', [CONTROL_ARM]), ('model', [MODEL_ARM]), ('joint', [JOINT_ARM]))},
        'directions': {direction: f'loss({reduced}) - loss({joint})'
                       for direction, (reduced, joint) in bidirectional.PRIMARY_DESIGNS.items()},
        'direction_reading': ('overlap is established only when the model adds little beyond the '
                              'statistics and the statistics add little beyond the model; the '
                              'reverse direction is what separates shared information from '
                              'model inferiority'),
        'channels': {name: dict(CHANNELS[name]) for name in declared.channels},
        'strata': {name: {'units': value.units, 'to_loss': value.to_loss,
                          'support': value.support, 'fragile': value.fragile}
                   for name, value in declared.strata.items()},
        'not_covered': {name: reason for name, reason in declared.not_covered},
        'forward_bound': {'quantity': ('the forward increment\'s simultaneous upper limit, in '
                                       'native units and as a fraction of the control set\'s own '
                                       'performance on the identical support'),
                          'reading': FORWARD_BOUND_READING,
                          'never': 'the forward direction never reports a state for smallness'},
        'gemme_risk_band': {'band': list(GEMME_RISK_BAND),
                            'counted_from': str(SUPPORT_FILE),
                            'rule': ('read off the cohort\'s exported retained row counts by '
                                     'gemme_risk_band(); never a count kept in this module')},
        'families': {channel: declared_family(lane_name, channel) for channel in declared.channels},
        'sensitivity_families': {channel: declared_family(lane_name, channel, role='sensitivity')
                                 for channel in declared.channels},
        'support_rule': ('one complete-case support per stratum, where every block of the '
                         'hierarchy is available; every loss is enumerated by name, count and '
                         'upstream reason'),
        'uncertainty': {'primary': bidirectional.NATIVE_METHOD,
                        'sensitivity': bidirectional.STANDARDIZED_METHOD,
                        'draws': bidirectional.DRAWS, 'seed': bidirectional.SEED,
                        'confidence': bidirectional.CONFIDENCE,
                        'floor': MINIMUM_BOOTSTRAP_UNITS,
                        'floor_rule': 'a lane or stratum under the floor is unresolved, not '
                                      'estimated'},
        'states': list(STATES),
        'non_identifiable': dict(declared.non_identifiable),
        'notes': list(declared.notes),
        'limitations': list(LIMITATIONS),
    }


LIMITATIONS = [
    'Increments are predictive contrasts under declared blocks, training-only scaling and a '
    'tuned ridge. They are not exclusive biological knowledge, mutual information or a causal '
    'decomposition, and the two directions are not an additive partition.',
    'The coupling block enters as one assay-reranked energy difference and the GEMME block as '
    'one assay-reranked official score, so neither carries per-position structure and only the '
    'coupling block is non-additive over sites.',
    'Every method reads the same retained homolog support at R1\'s own depth cap, so these '
    'contrasts do not estimate any method at an ideal alignment. The later cohorts searched at a '
    'deeper --max-target-seqs than R1, which no re-analysis can remove.',
    'An unresolved simultaneous interval is not equivalence. No post-hoc equivalence margin is '
    'declared, so "bounded negative" means a resolved negative sign and nothing weaker.',
    'Intervals condition on fitted held-out predictions; they exclude refitting and '
    'training-history uncertainty, and split seeds are sensitivities rather than replicates.',
    'Lanes, strata and channels are never pooled. Their estimands, units and resampling units '
    'differ, and a verdict that differs between two of them is a difference between estimands.',
    'A control set that carries more blocks is not thereby a stronger control: only a set that '
    'actually improves held-out prediction may be called stronger.',
    'The model enters as one scalar sequence likelihood difference per variant, so what is '
    'bounded is the evolutionary memory *as read out by sequence likelihood*, not the '
    'information content of the parameters. A probe on internal representations asks a '
    'different question and nothing here answers it; the representation channel, where a lane '
    'declares one, is reported separately and never merged into the likelihood channel.',
    'The two directions are not symmetric in predictors added. The reverse adds a whole '
    'family-fitted control set to one scalar; the forward adds one scalar to that control set. '
    'A resolved reverse beside an unresolved forward is therefore partly an asymmetry in power '
    'rather than in information. That does not rescue the forward side: the forward result is '
    'not a wide interval but a bound, at a few per cent of the control set\'s own performance '
    'on identical support, and a power asymmetry would matter only if the forward interval were '
    'merely too wide to resolve. It does constrain the reverse side, where the resolved sign '
    'should not be read as a measure of how much more information the statistics hold.',
    CONTROL_PROVENANCE[0].upper() + CONTROL_PROVENANCE[1:] + '.',
    f'The coupling block carries its own instrument precision: a single-host run-to-run spread '
    f'of {COUPLING_INSTRUMENT_BOUNDS["run_to_run_per_variant_score"]:g} on a per-variant score, '
    f'and {COUPLING_INSTRUMENT_BOUNDS["iteration_cap_fit"]:g} for a fit that stopped at its '
    'iteration cap. The two are reported separately rather than averaged, and no comparison '
    'resolves a difference below whichever bound applies to the fits it rests on.',
    'The nuisance rung is not profile-free: ' + NUISANCE_RUNG_LIMITATION + '.',
]


def _check_declarations() -> None:
    """Assert the declarations against each other at import, not at runtime."""

    if tuple(bidirectional.PRIMARY_DESIGNS) != bidirectional.DIRECTIONS:
        raise RuntimeError('the two directions must be read in the declared order')
    if bidirectional.PRIMARY_DESIGNS['model_given_controls'][1] != \
            bidirectional.PRIMARY_DESIGNS['controls_given_model'][1]:
        raise RuntimeError('both directions are paired reductions of one joint arm')
    if len(set(ARM_NAMES)) != 3:
        raise RuntimeError('the three arms of a rung must have distinct names')
    for name, roles in RUNGS.items():
        if roles[0] != 'nuisance' or len(set(roles)) != len(roles) or set(roles) - set(ROLES):
            raise RuntimeError(f'rung {name!r} is not a set of declared roles over the nuisances')
    if tuple(RUNGS) != tuple(sorted(RUNGS, key=lambda name: (len(RUNGS[name]), name))):
        raise RuntimeError('the hierarchy is declared from the smallest control set upwards')
    for key in NEW_BLOCKS.values():
        if key in {CHANNELS[name]['block'] for name in CHANNELS}:
            raise RuntimeError(f'the added block {key!r} collides with a model block')
    from . import statistical_ladder
    if set(VARIANT_EFFECT_ROLES) != set(ROLES):
        raise RuntimeError('the variant-effect binding must cover the same roles')
    letters = {letter for keys in VARIANT_EFFECT_ROLES.values() for letter in keys}
    if letters != set(statistical_ladder.BLOCK_ORDER) - {'M'}:
        raise RuntimeError('the variant-effect binding no longer covers that ladder\'s blocks')
    if letters & set(NEW_BLOCKS.values()):
        raise RuntimeError('an added block key collides with a variant-effect ladder letter, so '
                           'one vocabulary could not cover both')
    if set(FULL_SUPPORT_RUNGS) - set(RUNGS):
        raise RuntimeError('a full-support rung must be a rung of the hierarchy')
    for rung in FULL_SUPPORT_RUNGS:
        if set(RUNGS[rung]) & set(NEW_BLOCKS):
            raise RuntimeError(
                f'the full-support rung {rung!r} carries an added block, so a full-support '
                'cell could no longer be bound to an identity that excludes the added-block '
                'vocabulary; fit_identity would be unsound')
    if SHARED_RUNGS != tuple(name for name, roles in RUNGS.items() if 'profile' in roles):
        raise RuntimeError('the rungs read across lanes are the ones that carry profile')
    bridge = VARIANT_EFFECT_BRIDGE
    if set(bridge['profile_family']['rungs']) != set(SHARED_RUNGS):
        raise RuntimeError('the variant-effect profile family must supply every shared rung')
    if not set(bridge['c0_family']['rungs']) <= set(SHARED_RUNGS):
        raise RuntimeError('the variant-effect C0 family cannot supply an unshared rung')
    if set(bridge['absent_rungs']) | set(bridge['c0_family']['rungs']) != set(RUNGS):
        raise RuntimeError('the variant-effect bridge must account for every rung of the '
                           'hierarchy, as supplied or as absent')
    if bridge['rerun_at_this_hierarchy'] is not False:
        raise RuntimeError('the variant-effect ladder is not re-run at this hierarchy')
    for name, declared in LANES.items():
        if declared.name != name:
            raise RuntimeError(f'lane {name!r} is declared under another name')
        if not declared.profile_blocks:
            raise RuntimeError(f'{name}: profile is a control of every lane and needs a binding')
        if set(declared.profile_blocks) & set(NEW_BLOCKS.values()):
            raise RuntimeError(f'{name}: a profile key collides with an added block key')
        if declared.tokenisation_block in NEW_BLOCKS.values():
            raise RuntimeError(f'{name}: the tokenisation key collides with an added block key')
        if set(declared.non_identifiable) - set(ROLES):
            raise RuntimeError(f'{name}: a non-identifiable role is not a declared role')
        if 'nuisance' in declared.non_identifiable or 'profile' in declared.non_identifiable:
            raise RuntimeError(f'{name}: the nuisance and profile rungs are always identifiable')
        if not declared.strata or not declared.channels:
            raise RuntimeError(f'{name}: a lane declares at least one stratum and one channel')
        for value in declared.strata.values():
            if value.to_loss not in (1., -1.):
                raise RuntimeError(f'{name}: a stratum reads as a loss (+1) or a score (-1)')
        for entry in declared.not_covered:
            if len(entry) != 2 or not all(isinstance(part, str) and part for part in entry):
                raise RuntimeError(f'{name}: an uncovered reading is a (name, reason) pair')
            if entry[0] in declared.strata:
                raise RuntimeError(f'{name}: {entry[0]!r} is declared both covered and not')
        if set(declared.channels) - set(CHANNELS):
            raise RuntimeError(f'{name}: an undeclared model channel')
        if declared.channels[0] != 'likelihood':
            raise RuntimeError(f'{name}: the likelihood channel is the primary one')
        if set(declared.arms) - set(declared.panel_arms):
            raise RuntimeError(f'{name}: a scored arm is outside the panel roster')
        if set(declared.declared_family) != set(declared.channels) or \
                set(declared.full_panel_family) != set(declared.channels):
            raise RuntimeError(f'{name}: every channel declares a family size')
        for channel in declared.channels:
            size = _family_size(len(declared.rungs), len(declared.arms))
            full = _family_size(len(declared.rungs), len(declared.panel_arms))
            if size != declared.declared_family[channel]:
                raise RuntimeError(
                    f'{name}/{channel}: the realised family is {len(declared.rungs)} control sets '
                    f'x {len(declared.arms)} arms x {len(bidirectional.DIRECTIONS)} directions = '
                    f'{size} columns, and the declaration says '
                    f'{declared.declared_family[channel]}; at the full '
                    f'{len(declared.panel_arms)}-arm panel it would be {full}')
            if full != declared.full_panel_family[channel]:
                raise RuntimeError(
                    f'{name}/{channel}: at the full {len(declared.panel_arms)}-arm panel the '
                    f'family would be {full} columns and the declaration says '
                    f'{declared.full_panel_family[channel]}')


_check_declarations()


# --------------------------------------------------------------------------- #
# Entry point. ``declare`` needs no wave at all; ``support`` and ``panel`` refuse
# clearly while the coupling and GEMME features are still being produced.
# --------------------------------------------------------------------------- #

def _write(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False,
                                    default=bidirectional.json_default) + '\n')
    temporary.replace(path)


def _emit(value, out) -> None:
    if out is None:
        print(json.dumps(value, indent=2, allow_nan=False, default=bidirectional.json_default))
    else:
        _write(out, value)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('mode', choices=['declare', 'support', 'panel'])
    parser.add_argument('--lane', required=True, choices=sorted(LANES))
    parser.add_argument('--out', type=Path, help='write the record here instead of stdout')
    parser.add_argument('--export', type=Path,
                        help="support: the cohort's export directory, which carries its "
                             'variant table')
    parser.add_argument('--rows', type=Path, help='support: the variant table, if not under --export')
    parser.add_argument('--couplings', type=Path, help='support: the coupling score table')
    parser.add_argument('--gemme', type=Path, help='support: the GEMME score table')
    parser.add_argument('--cells', type=Path, help='panel: the per-group held-out errors')
    args = parser.parse_args(argv)
    declared = lane(args.lane)
    if args.mode == 'declare':
        _emit(contract(args.lane), args.out)
        return
    if args.mode == 'support':
        if args.rows is None and args.export is None:
            parser.error('support needs --export or --rows')
        rows = read_rows(args.rows if args.rows is not None else args.export / ROWS_FILE)
        tables = {}
        for role, path in (('couplings', args.couplings), ('gemme', args.gemme)):
            if role in declared.non_identifiable:
                continue
            if path is None:
                parser.error(f'support needs --{role} for lane {args.lane}')
            tables[role] = read_scores(path, role)
        _emit(complete_case(args.lane, rows, tables), args.out)
        return
    if args.cells is None:
        parser.error('panel needs --cells')
    document = read_cells(args.cells)
    if document['lane'] != args.lane:
        raise ValueError(f'{args.cells} carries lane {document["lane"]!r}, not {args.lane!r}')
    cells = {(cell['rung'], cell['arm']): cell for cell in document['cells']}
    _emit(panel(args.lane, document['stratum'], document['channel'], cells,
                groups=document['groups'], support_sha256=document['support_sha256'],
                role=document.get('role', 'primary')), args.out)


if __name__ == '__main__':
    main()
