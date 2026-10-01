"""The pattern every gate already follows, written down once.

Each gate of the capability map runs the same six steps in order and reports at
whatever step it has reached: endpoint qualification, baseline and control
qualification, native model behaviour, representation readout, held-family and
remote-family generalization, and causal validation where justified. That
sequence is declared in :doc:`archive/docs/shared/capability-map-plan.md`; what it does not
have is a form a reader or a test can check a gate against, so a gate that
reported a representation verdict as final, or a control whose own contribution
carries no interval, would read as complete.

This module is that form and nothing more. It is the smallest abstraction the
existing gates can be expressed in, and it deliberately does **not** refactor
them: each gate keeps its own modules, scripts and record, and what is declared
here is a read-only transcription of the elements its record already carries,
checked against the contract's rules and, where a record is supplied, against
that record's own bytes.

Four rules carry the substance, and each exists because a gate record already
shows why:

**A control must carry its own contribution with an interval.** The stability
gate keeps a control block whose own contribution is +0.00390 [-0.00068,
+0.00864], an interval containing zero, and records that this makes *beyond
controls* weaker than the phrase suggests. A contract that accepted a control
without an interval would hide exactly that.

**A representation verdict is not available while the readout reassessment is
open.** Every representation-level verdict is provisional until the reassessment
completes, so an element at that step may be provisional, not applicable or
absent, and a gate declaring it reported diverges from the contract.

**A likelihood element is not deferred.** A native likelihood enters as one
scalar column, so a linear readout of it is close to the full function space over
that feature; the asymmetry is what makes the sequencing rule tractable and is
expressed here rather than left to each gate.

**A step not reached is named, not omitted.** *Unresolved*, *not detected*, *not
applicable* and *absent* are four different states, and a gate that stopped at
step 1 records the later steps as absent with the reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from .gate_retention import GATE_RETENTION

#: The six steps, in the order the capability map declares them. Imported by name
#: rather than re-numbered, so a gate's report is indexable against the plan.
CONTRACT_STEPS = (
    'endpoint_qualification',
    'control_qualification',
    'native_likelihood',
    'representation_readout',
    'generalization',
    'causal_validation',
)

#: What a gate can report at a step. ``unresolved`` and ``not_detected`` are
#: never merged: the first says the instrument, support or readout could not
#: decide, the second says a qualified measurement with qualified controls
#: resolved the quantity against zero or below it.
STATUSES = ('reported', 'not_detected', 'unresolved', 'provisional', 'not_applicable', 'absent')

#: Statuses that assert a measurement and therefore require a quantity and a bound.
MEASURED_STATUSES = ('reported', 'not_detected', 'unresolved', 'provisional')

#: Statuses that assert no measurement and therefore require a reason.
UNMEASURED_STATUSES = ('not_applicable', 'absent')

#: Forms a control's own qualification can take. ``interval`` is the form every
#: gate with a fitted readout uses: the control's paired contribution to the
#: baseline it augments, with a resampling interval, so that a control
#: indistinguishable from zero is visible rather than implied. The other two are
#: for a control that is a *generator* rather than a feature block, where what has
#: to be established is that it emits the statistics it declares and nothing more.
#: ``exactness`` is a decidable property checked on every row, which is a census
#: with no residual; ``order_statistic`` is a signed position on a declared axis,
#: read against a like-for-like reference. Both are tighter than an interval on
#: the quantity they establish, and an interval there would describe sampling
#: variation a census does not have or answer a question the control need not pass.
QUALIFICATION_FORMS = ('interval', 'exactness', 'order_statistic')

#: True while the readout-class sweep and the depth sweep are open. It gates one
#: rule -- a representation element may not be ``reported`` -- and is a single
#: switch rather than a per-gate declaration, because the sequencing rule is
#: programme-level. When the reassessment lands this becomes False and the gates
#: whose representation elements are provisional are the ones to recompute.
READOUT_REASSESSMENT_OPEN = True


@dataclass(frozen=True)
class ControlContribution:
    """One control's own contribution, in the form its own record publishes it.

    ``form`` names which of the three declared qualification forms the record
    uses. An ``interval`` contribution carries its interval and says whether it
    resolves away from zero. An ``exactness`` or ``order_statistic`` contribution
    carries neither -- there is no sampling variation in a census and no zero to
    resolve against on an axis -- and must instead say what its qualification
    ``establishes``, so that a non-interval form is a declared difference in kind
    rather than a missing interval.
    """

    name: str
    contribution: str
    interval: str = ''
    resolved_away_from_zero: bool | None = None
    form: str = 'interval'
    establishes: str = ''

    def __post_init__(self):
        if not self.name or not self.contribution:
            raise ValueError('a control contribution names the control and its own change')
        if self.form not in QUALIFICATION_FORMS:
            raise ValueError(f'{self.name}: {self.form!r} is not a declared qualification form')
        if self.form == 'interval':
            if not self.interval:
                raise ValueError(f'{self.name}: an interval-form contribution publishes its '
                                 'interval, so that a control indistinguishable from zero is '
                                 'visible rather than implied')
            if self.resolved_away_from_zero is None:
                raise ValueError(f'{self.name}: an interval-form contribution says whether it '
                                 'resolves away from zero')
            if self.interval not in self.contribution:
                raise ValueError(f'{self.name}: the published interval must be part of the '
                                 'quantity the record states, so one string is checked against '
                                 'the record')
        else:
            if self.interval or self.resolved_away_from_zero is not None:
                raise ValueError(f'{self.name}: a {self.form} contribution publishes no interval '
                                 'and nothing resolves away from zero in it')
            if not self.establishes:
                raise ValueError(f'{self.name}: a {self.form} contribution states what its '
                                 'qualification establishes, so the form is a declared '
                                 'difference in kind rather than a missing interval')


@dataclass(frozen=True)
class GateElement:
    """One gate's report at one contract step."""

    step: str
    status: str
    quantities: tuple[str, ...] = ()
    bound: str = ''
    reason: str = ''
    contributions: tuple[ControlContribution, ...] = ()
    remote: str = ''

    def __post_init__(self):
        if self.step not in CONTRACT_STEPS:
            raise ValueError(f'{self.step!r} is not a contract step')
        if self.status not in STATUSES:
            raise ValueError(f'{self.step}: {self.status!r} is not a declared status')


@dataclass(frozen=True)
class GateDeclaration:
    """One gate's elements, transcribed from its own record and pointing at it."""

    gate: str
    record: str
    elements: tuple[GateElement, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def by_step(self) -> dict[str, GateElement]:
        return {element.step: element for element in self.elements}


def _divergences(declaration: GateDeclaration) -> list[str]:
    """Where one gate's declared elements depart from the contract."""
    found: list[str] = []
    steps = [element.step for element in declaration.elements]
    for step in CONTRACT_STEPS:
        if steps.count(step) != 1:
            found.append(f'{step} appears {steps.count(step)} times, not once')
    for element in declaration.elements:
        where = f'{element.step}'
        if element.status in MEASURED_STATUSES:
            if not element.quantities:
                found.append(f'{where} is {element.status} and names no quantity')
            if not element.bound:
                found.append(f'{where} is {element.status} and states no bound on its reading')
        if element.status in UNMEASURED_STATUSES:
            if not element.reason:
                found.append(f'{where} is {element.status} and gives no reason')
            if element.quantities:
                found.append(f'{where} is {element.status} yet names quantities')
        if element.step == 'control_qualification' and element.status in MEASURED_STATUSES:
            if not element.contributions:
                found.append(f'{where} is {element.status} and declares no control contribution, '
                             'so no control is shown competent in its own right')
            for contribution in element.contributions:
                if contribution.form == 'interval' and not contribution.interval:
                    found.append(
                        f'{where}: the record publishes no interval for the own contribution of '
                        f'{contribution.name!r}, and declares no other qualification form for it')
        if element.step == 'representation_readout':
            if READOUT_REASSESSMENT_OPEN and element.status in ('reported', 'not_detected'):
                found.append(f'{where} is {element.status} while the readout reassessment is '
                             'open; a representation-level verdict is provisional until it lands')
        if element.step == 'generalization' and element.status in MEASURED_STATUSES:
            if not element.remote:
                found.append(f'{where} is {element.status} and says nothing about remote-family '
                             'generalization, which is the half a held-family fold does not reach')
        if element.step == 'causal_validation' and element.status in MEASURED_STATUSES:
            if not element.quantities:
                found.append(f'{where} claims a causal validation without naming the resolved and '
                             'generalising increment it intervened on')
    return found


def transcription_check(declaration: GateDeclaration, root: Path | str = Path('.')) -> dict:
    """Whether every quantity the declaration names occurs verbatim in the record.

    This is what makes the conformance check read-only and still meaningful: no
    gate is altered, and a declared quantity that has drifted from the document
    it is transcribed from is reported rather than trusted. A record that cannot
    be read is a failure of the check, not a pass.
    """
    path = Path(root) / declaration.record
    if not path.is_file():
        return dict(record=str(path), readable=False, checked=0, missing=[],
                    reason='the record document is not present under this root')
    text = path.read_text(encoding='utf-8')
    missing, checked = [], 0
    for element in declaration.elements:
        for quantity in element.quantities:
            checked += 1
            if quantity not in text:
                missing.append(dict(step=element.step, quantity=quantity))
    for element in declaration.elements:
        for contribution in element.contributions:
            checked += 1
            if contribution.contribution not in text:
                missing.append(dict(step=element.step, quantity=contribution.contribution,
                                    control=contribution.name))
    return dict(record=str(path), readable=True, checked=checked, missing=missing,
                conforms=not missing)


def conform(gate: str, *, root: Path | str | None = None) -> dict:
    """One gate's conformance to the contract, optionally against its own record."""
    declaration = GATE_DECLARATIONS[gate]
    divergences = _divergences(declaration)
    record = dict(gate=gate, record=declaration.record,
                  steps={element.step: dict(status=element.status,
                                            quantities=list(element.quantities),
                                            bound=element.bound, reason=element.reason,
                                            remote=element.remote,
                                            contributions=[dict(name=c.name,
                                                                contribution=c.contribution,
                                                                form=c.form,
                                                                interval=c.interval or None,
                                                                establishes=c.establishes or None,
                                                                resolved_away_from_zero=
                                                                c.resolved_away_from_zero)
                                                           for c in element.contributions])
                         for element in declaration.elements},
                  notes=list(declaration.notes),
                  divergences=divergences, conforms=not divergences)
    if root is not None:
        record['transcription'] = transcription_check(declaration, root)
    return record


def contract_digest() -> str:
    """Content digest of the contract and every gate's declared elements."""
    payload = {
        'steps': list(CONTRACT_STEPS), 'statuses': list(STATUSES),
        'readout_reassessment_open': READOUT_REASSESSMENT_OPEN,
        'gates': {gate: dict(record=declaration.record, notes=list(declaration.notes),
                             elements=[dict(step=e.step, status=e.status,
                                            quantities=list(e.quantities), bound=e.bound,
                                            reason=e.reason, remote=e.remote,
                                            contributions=[dict(name=c.name,
                                                                contribution=c.contribution,
                                                                form=c.form,
                                                                resolved=c.resolved_away_from_zero)
                                                           for c in e.contributions])
                                       for e in declaration.elements])
                  for gate, declaration in sorted(GATE_DECLARATIONS.items())}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def conformance(root: Path | str | None = None) -> dict:
    """Every declared gate's conformance, with the gates that are not declared named."""
    gates = {gate: conform(gate, root=root) for gate in sorted(GATE_DECLARATIONS)}
    undeclared = sorted(set(GATE_RETENTION) - set(GATE_DECLARATIONS))
    return dict(schema='d1_gate_contract_v1', contract_sha256=contract_digest(),
                steps=list(CONTRACT_STEPS), statuses=list(STATUSES),
                readout_reassessment_open=READOUT_REASSESSMENT_OPEN,
                gates=gates,
                conforming=sorted(gate for gate, record in gates.items() if record['conforms']),
                diverging={gate: record['divergences'] for gate, record in gates.items()
                           if not record['conforms']},
                not_declared=dict(
                    gates=undeclared,
                    reason=('a retention entry with no record document to transcribe from: the '
                            'three unlaunched gates, and the upstream studies and the transplant '
                            'experiment, which are not gates of the capability map')))


_ABSENT_CAUSAL = 'no resolved and generalising increment has been judged worth intervening on; ' \
                 'the step is conditional rather than a routine final one'


#: Each gate's elements, transcribed from its own record. Read-only: no gate is
#: expressed on this contract in its own code, and nothing here is a second
#: source for any quantity -- :func:`transcription_check` requires every quantity
#: named here to occur verbatim in the record it points at.
GATE_DECLARATIONS: dict[str, GateDeclaration] = {
    'local_context': GateDeclaration(
        gate='local_context',
        record='archive/docs/R1/gate-local-context.md',
        elements=(
            GateElement('endpoint_qualification', 'reported',
                        quantities=('201-assay, 163-cluster, 25,728-variant anchor',),
                        bound='the endpoint is the admitted Readout anchor, inherited unchanged '
                              'through the same call sites rather than re-qualified here'),
            GateElement('control_qualification', 'reported',
                        quantities=('+0.47906 to +0.48290',),
                        bound='a candidate qualifies only by not lowering both baselines at every '
                              'one of the three split seeds; the rule is a refusal applied to '
                              'control-versus-control correlations only',
                        contributions=(
                            ControlContribution('wall over C', '+0.03387 [+0.02228, +0.04622]',
                                                '[+0.02228, +0.04622]', True),
                            ControlContribution('wall over C+P', '+0.02973 [+0.02058, +0.03926]',
                                                '[+0.02058, +0.03926]', True),
                            ControlContribution('rf3 over C', '+0.01763 [+0.00456, +0.03056]',
                                                '[+0.00456, +0.03056]', True),
                            ControlContribution('rf3 over C+P, unresolved at two of three seeds',
                                                '+0.00082 [−0.00545, +0.00701]',
                                                '[−0.00545, +0.00701]', False),
                            ControlContribution('wsub over C, discarded',
                                                '−0.01245 [−0.02260, −0.00291]',
                                                '[−0.02260, −0.00291]', True),
                        )),
            GateElement('native_likelihood', 'reported',
                        quantities=('+0.02328 to +0.02503', '14 of the 33 arms'),
                        bound='a likelihood enters as one scalar column, so the readout-class '
                              'question barely bites on it; the increment is bounded by the '
                              'declared controls and the 163-cluster label budget'),
            GateElement('representation_readout', 'provisional',
                        quantities=('+0.02605 to +0.03033',),
                        bound='one fixed compressed linear family: 1,024 fixed Gaussian '
                              'projection coordinates of four pooled hidden-state summaries at '
                              'the two retained depths, under linear ridge on the admitted grid'),
            GateElement('generalization', 'reported',
                        quantities=('99 cells',),
                        bound='held-out wild-type clusters at 50% identity under five outer and '
                              'four inner folds at three split seeds',
                        remote='none: the grouping unit is the existing single-linkage wild-type '
                               'cluster at 50% identity with 80% coverage, which is not a '
                               'certified Pfam-family or CATH-superfamily holdout and does not '
                               'exclude overlap with pretraining corpora'),
            GateElement('causal_validation', 'absent', reason=_ABSENT_CAUSAL),
        ),
        notes=('the representation side is the half this gate expects to be recomputed; its '
               'likelihood side and its control qualification do not depend on the readout class',)),
    'folding_stability': GateDeclaration(
        gate='folding_stability',
        record='archive/docs/R4/gate-stability.md',
        elements=(
            GateElement('endpoint_qualification', 'reported',
                        quantities=('0.2337 [0.2209, 0.2458] kcal/mol',
                                    '3.86 [3.61, 4.13]'),
                        bound='two proteases are two assay channels over the same sequences, '
                              'library and stability-inference model, so the shared component is '
                              'an upper bound on reproducible signal and the discordance a lower '
                              'bound on per-channel noise'),
            GateElement('control_qualification', 'reported',
                        quantities=('49.7% to 50.4%',),
                        bound='a candidate is kept only if its paired reduction in group-equal '
                              'held-out mean squared error over the standing set is positive at '
                              'every one of the three split seeds',
                        contributions=(
                            ControlContribution('nonlinear-additive response',
                                                '+0.02529 [+0.01639, +0.03460]',
                                                '[+0.01639, +0.03460]', True),
                            ControlContribution('local chemistry, kept by the sign rule alone',
                                                '+0.00390 [−0.00068, +0.00864]',
                                                '[−0.00068, +0.00864]', False),
                            ControlContribution('composition, discarded',
                                                '+0.00007 [−0.01645, +0.01589]',
                                                '[−0.01645, +0.01589]', False),
                            ControlContribution('mutation-local profile, discarded',
                                                '+0.01876 [−0.00883, +0.04521]',
                                                '[−0.00883, +0.04521]', False),
                        )),
            GateElement('native_likelihood', 'reported',
                        quantities=('4 of 33 arms', '+0.02260 [+0.01333, +0.03144]'),
                        bound='against a qualified control set that already removes half of the '
                              'no-effect null, the resolved increments are 1.2% to 3.5% of the '
                              'remaining error, so the quantity is resolvable and small'),
            GateElement('representation_readout', 'provisional',
                        quantities=('+0.06121',),
                        bound='one compressed linear readout of four pooled block summaries at '
                              'the retained depths; the full-width states are retained so the '
                              'revisit is a refit rather than a re-extraction'),
            GateElement('generalization', 'reported',
                        quantities=('101 family groups',),
                        bound='held family groups under the frozen grouping contract, five outer '
                              'and four inner folds at three split seeds',
                        remote='two declared purges at 20% identity with 80% and with 50% '
                               'coverage; every observed alignment-edge count lies inside a '
                               'composition-preserving shuffle null, so the purge bounds '
                               'dependence on a chance-level nearest-neighbour set and certifies '
                               'no remote-family disjointness'),
            GateElement('causal_validation', 'absent', reason=_ABSENT_CAUSAL),
        )),
    'residue_interactions': GateDeclaration(
        gate='residue_interactions',
        record='archive/docs/R3/pairwise-epistasis-results.md',
        elements=(
            GateElement('endpoint_qualification', 'reported',
                        quantities=('8,192',),
                        bound='the four-state cycle endpoint, admitted by its own label '
                              'instrument; 99.98% of admitted cycles sit on site pairs the source '
                              'study selected as suspected interactions',
                        ),
            GateElement('control_qualification', 'unresolved',
                        quantities=('admissible on 45 of 64 groups',),
                        bound='the pairwise comparator is admissible on 45 of the 64 held groups '
                              'and absent on the other 19, where the gate is unresolved rather '
                              'than negative and no weaker estimator is substituted',
                        contributions=(
                            ControlContribution('Q, the fitted pairwise comparator, over the '
                                                'matched baseline at seed 20260923',
                                                '+0.00578 [+0.00004, +0.01269]',
                                                '[+0.00004, +0.01269]', True),
                            ControlContribution('Q at seed 20260924, where its own contribution '
                                                'does not resolve',
                                                '−0.00323 [−0.03625, +0.02782]',
                                                '[−0.03625, +0.02782]', False),
                            ControlContribution('the tokenisation block T over C+G, ProtGPT2',
                                                '+0.00393 [−0.00129, +0.00896]',
                                                '[−0.00129, +0.00896]', False),
                        )),
            GateElement('native_likelihood', 'reported',
                        quantities=('+0.00812 [+0.00373, +0.01287]',),
                        bound='the first-order likelihood difference is a single-mutant quantity, '
                              'so surviving an interaction control does not make the information '
                              'pairwise',
                        ),
            GateElement('representation_readout', 'provisional',
                        quantities=('1,024 columns on 64 held groups',),
                        bound='four pools projected to 256 coordinates each, with the ridge grid '
                              'running out of shrinkage in 43 of 360 selections; full-width '
                              'per-state block outputs are retained so a different projection '
                              'replays without model inference'),
            GateElement('generalization', 'reported',
                        quantities=('64',),
                        bound='held family groups, with the site pair as the independent unit and '
                              'the cycle count never entering as a sample size',
                        remote='not tested here: the retrieval-and-memorization gate owns the '
                               'identity-band, retrieval-depth and near-duplicate stratification '
                               'of this same gain'),
            GateElement('causal_validation', 'absent', reason=_ABSENT_CAUSAL),
        )),
    'higher_order': GateDeclaration(
        gate='higher_order',
        record='archive/docs/R3/gate-higher-order.md',
        elements=(
            GateElement('endpoint_qualification', 'unresolved',
                        quantities=('0.464', '0.436', '1.06'),
                        bound='two site pairs and two family groups, below the eight-unit '
                              'resampling floor, so every quantity is a point estimate with no '
                              'interval and the propagated floor is inherited rather than '
                              'measured on these units'),
            GateElement('control_qualification', 'absent',
                        reason='the endpoint did not clear its own measurement-noise floor, so no '
                               'control was offered and no design was fitted'),
            GateElement('native_likelihood', 'absent',
                        reason='no model was scored, so the gate makes no statement about any '
                               'model\'s likelihood assignments'),
            GateElement('representation_readout', 'absent',
                        reason='no readout was fitted; a fold-identity and a held-out-label '
                               'leakage test on a fitted readout are recorded as absent rather '
                               'than as satisfied'),
            GateElement('generalization', 'absent',
                        reason='the frozen five-outer, four-inner held-group design cannot be '
                               'instantiated on two family groups under any fold assignment'),
            GateElement('causal_validation', 'absent',
                        reason='the gate stopped at endpoint qualification'),
        )),
    'higher_order_extended': GateDeclaration(
        gate='higher_order_extended',
        record='archive/docs/R3/gate-higher-order-extended.md',
        elements=(
            GateElement('endpoint_qualification', 'unresolved',
                        quantities=('0.276 [0.000, 0.496]', '0.064 [0.000, 0.214]'),
                        bound='one assay carries a measured floor and one a synonymous-genotype '
                              'channel; the ratio is an upper bound over a lower bound, because '
                              'error common to the channels inflates the shared component and '
                              'cancels in their difference'),
            GateElement('control_qualification', 'absent',
                        reason='the endpoint failed its floor at every order from three upwards '
                               'on both assays where a floor is measurable, so no control was '
                               'offered'),
            GateElement('native_likelihood', 'absent',
                        reason='the 33-arm roster was not run and no arm was inspected, scored or '
                               'selected, so the roster stays unconditioned on any outcome'),
            GateElement('representation_readout', 'absent',
                        reason='no readout was fitted, so no fold-identity or leakage condition '
                               'on one exists; that is recorded as absent rather than satisfied'),
            GateElement('generalization', 'absent',
                        reason='no model comparison was fitted, so there is no increment to '
                               'stratify'),
            GateElement('causal_validation', 'absent',
                        reason='the gate stopped at endpoint qualification'),
        )),
    'structural_constraints': GateDeclaration(
        gate='structural_constraints',
        record='archive/docs/R3/gate-structure-contact.md',
        elements=(
            GateElement('endpoint_qualification', 'unresolved',
                        quantities=('+0.192 [−0.061, +0.412]', '−0.001 [−0.095, +0.075]'),
                        bound='111 retained contacts against 53 matched controls at a Kish '
                              'effective 25.1 control site pairs; a pure-noise negative control '
                              'built from the two protease channels returns +0.031 [−0.007, '
                              '+0.064] on the same support'),
            GateElement('control_qualification', 'reported',
                        quantities=('+0.285 [+0.130, +0.434]',),
                        bound='coarsened exact matching on sequence separation, relative '
                              'accessibility and wild-type residue class, with residual '
                              'within-cell imbalance at −0.220 and −0.182 standardized mean '
                              'difference after weighting',
                        contributions=(
                            ControlContribution('the burial positive control, which qualifies the '
                                                'structure instrument by a route independent of '
                                                'the contact contrast',
                                                '+0.285 [+0.130, +0.434]',
                                                '[+0.130, +0.434]', True),
                        )),
            GateElement('native_likelihood', 'absent',
                        reason='part 2 was not run, so no model-side contact number exists for '
                               'any arm; it was not run rather than run and reported as '
                               'uninformative'),
            GateElement('representation_readout', 'absent',
                        reason='part 2 was conditional on part 1, which is not passed'),
            GateElement('generalization', 'absent',
                        reason='no model-side increment was measured, so there is nothing to '
                               'stratify; the two-stage resampling of held groups and their site '
                               'pairs applies to the measured endpoint alone'),
            GateElement('causal_validation', 'absent',
                        reason='the gate stopped at part 1'),
        )),
    'evolution_adaptation': GateDeclaration(
        gate='evolution_adaptation',
        record='archive/docs/R5/gate-retrieval-memorization.md',
        elements=(
            GateElement('endpoint_qualification', 'reported',
                        quantities=('1.11 × 10⁻¹⁶',),
                        bound='neither gain is refitted: a stratum estimate is the same paired '
                              'statistic over the same fitted held-out predictions, and the '
                              'decomposition identity is computed rather than assumed'),
            GateElement('control_qualification', 'not_applicable',
                        reason='this gate qualifies no control of its own: it re-reads two '
                               'studies\' fitted held-out predictions on the units each stratum '
                               'holds, so the controls are those studies\' own qualified ones, '
                               'inherited unchanged. What it declares instead are the six '
                               'stratifications, frozen with no free band edge, and its refit of '
                               'the stability arm is refused unless the row identity, fold '
                               'identity, support counts, design widths, selected penalties and '
                               'every whole-support increment reproduce the admitted record, '
                               'which they do to 7.0 × 10⁻¹⁶ squared kcal/mol'),
            GateElement('native_likelihood', 'reported',
                        quantities=('10 of 33', '5 of 33'),
                        bound='the bands are properties of a search against one UniRef50 '
                              'snapshot, not of any model\'s training corpus; no detected '
                              'alignment is not certified corpus exclusion'),
            GateElement('representation_readout', 'provisional',
                        quantities=('+0.03231 [+0.00275, +0.06355]',),
                        bound='a stratified representation null cannot be separated from a limit '
                              'of the compressed linear readout class until the class sweep and '
                              'the full-depth re-extraction land'),
            GateElement('generalization', 'reported',
                        quantities=('163', '179.6'),
                        bound='six declared stratifications over 20 populated bands, with 20,097 '
                              'pointwise unadjusted intervals in total, so every stratum verdict '
                              'rests on holding at all three seeds and on the whole-support '
                              'finding it decomposes',
                        remote='the binding limitation: 49 of 64 stability groups and 140 of 163 '
                               'ranking clusters sit at or above 95% identity, so the '
                               'remote-identity end is unresolved in both directions and neither '
                               'gain has been tested against remote homology'),
            GateElement('causal_validation', 'absent', reason=_ABSENT_CAUSAL),
        )),
    'generation_control': GateDeclaration(
        gate='generation_control',
        record='archive/docs/R6/gate-generative-control.md',
        elements=(
            GateElement('endpoint_qualification', 'reported',
                        quantities=('0.892 [0.871, 0.910]', '0.835 [0.811, 0.857]'),
                        bound='a curated-profile assignment is a threshold on a bit score against '
                              'a curated model, and a generator with direct access to the '
                              'scoring database misses roughly one attempt in nine'),
            GateElement('control_qualification', 'reported',
                        quantities=('0.154 KD units',),
                        bound='a generator enters only if it reproduces the order statistics it '
                              'declares, measured rather than asserted; the pre-declared '
                              'order-axis margin was declared against a wrong premise and both '
                              'readings are reported',
                        contributions=(
                            ControlContribution(
                                'fragment: a contiguous substring of its named corpus record at '
                                'the exact length, and the binding control on both endpoints',
                                '0 containment failures and 0 length failures of 14,467',
                                form='exactness',
                                establishes='that every emitted sequence is a contiguous '
                                            'substring of its named corpus record at its parent '
                                            'attempt\'s exact length -- a decidable property '
                                            'checked on every one of the 14,467 pairs, so the '
                                            'qualification is a census with no residual and an '
                                            'interval would describe sampling variation it does '
                                            'not have'),
                            ControlContribution(
                                'hydropathy: the attempt\'s own mean Kyte-Doolittle hydropathy',
                                '0.154 KD units mean absolute deviation',
                                form='exactness',
                                establishes='that no sequence misses its own per-sequence target '
                                            'by more than the tolerance declared before the '
                                            'measurement, 0.25 KD units, at a realised mean '
                                            'absolute deviation of 0.154 and a median of 0.117; a '
                                            'bound on every row rather than an interval on their '
                                            'mean'),
                            ControlContribution(
                                'markov_4: the corpus conditional at order 4',
                                '−0.0007 nats/residue at order 4',
                                form='order_statistic',
                                establishes='that the cohort sits at its declared order on the '
                                            'corpus\'s own scale and picks up none of the '
                                            'structure the corpus gains one order above it, as a '
                                            'signed comparison against a length-matched corpus '
                                            'reference; a position on an axis, which has no zero '
                                            'for an interval to resolve against'),
                        )),
            GateElement('native_likelihood', 'not_applicable',
                        reason='the endpoints are properties of emitted sequences; a native '
                               'likelihood is not read against them at this gate, and scoring, '
                               'generation and context utilisation have already been shown not to '
                               'collapse onto one axis'),
            GateElement('representation_readout', 'not_applicable',
                        reason='none is applicable: reading a frozen representation against these '
                               'endpoints would need an instrument this gate does not build'),
            GateElement('generalization', 'unresolved',
                        quantities=('4,000 resamples',),
                        bound='the intervals cover variation across the 800 attempts of one arm '
                              'at one decoding configuration and one batch-seed stream, and not '
                              'the configuration, the seed stream or retraining',
                        remote='each attempt carries a reported maximum identity to the searched '
                               'snapshot in declared bands, which bounds what the endpoint means '
                               'rather than measuring retrieval: no training set is accessible '
                               'and the snapshot is not any arm\'s training corpus'),
            GateElement('causal_validation', 'absent', reason=_ABSENT_CAUSAL),
        )),
    'readout_panel': GateDeclaration(
        gate='readout_panel',
        record='archive/docs/R2/readout-results.md',
        elements=(
            GateElement('endpoint_qualification', 'reported',
                        quantities=('201 assays, 163 wild-type identity clusters and 25,728 '
                                    'variants',),
                        bound='the anchor is a finite assay support and not a population sample; '
                              'estimates from different panels are not directly comparable model '
                              'ranks'),
            GateElement('control_qualification', 'reported',
                        quantities=('+0.48569 [+0.45550, +0.51519]',),
                        bound='every predictor is fitted on identical training labels and '
                              'weights, so the controls are equally supervised rather than merely '
                              'present; a gain over unsupervised likelihood alone cannot '
                              'distinguish a better readout from the benefit of new labels',
                        contributions=(
                            ControlContribution(
                                'the matched supervised baseline B, fitted alone on held-out '
                                'clusters, on the ProteinGLM-7B-CLM row of the anchor panel',
                                '+0.48569 [+0.45550, +0.51519]',
                                '[+0.45550, +0.51519]', True),
                            ControlContribution(
                                'the same baseline over the arm\'s own native likelihood, which '
                                'is what makes it a matched control rather than a stronger one',
                                '+0.09586 [+0.07642, +0.11471]',
                                '[+0.07642, +0.11471]', True),
                        )),
            GateElement('native_likelihood', 'reported',
                        quantities=('+0.38984 [+0.36161, +0.41669]',),
                        bound='raw likelihood ranking needs no folds, no penalty grid and no '
                              'projection, so the readout-class question barely bites on it; the '
                              'matched supervised baseline already exceeds it on this arm'),
            GateElement('representation_readout', 'provisional',
                        quantities=('+0.01970 [+0.00942, +0.02971]', '+0.02255'),
                        bound='one fixed compressed linear family at two extraction depths: '
                              '1,024 fixed Gaussian coordinates of four pooled summaries, so no '
                              'negative result here reaches other layers, nonlinear readouts or '
                              'uncompressed features'),
            GateElement('generalization', 'reported',
                        quantities=('163 wild-type identity clusters',),
                        bound='five outer and four inner wild-type-cluster folds at three split '
                              'seeds, with the label budget at 163 clusters on the anchor and 30 '
                              'on the conditioned panel',
                        remote='wild-type clustering at 50% identity does not guarantee '
                               'remote-superfamily disjointness or exclude pretraining overlap; '
                               'the evolution-and-adaptation gate owns the stratification'),
            GateElement('causal_validation', 'absent',
                        reason='establishing a claim about the representation rather than about a '
                               'readout requires removing the information from the states and '
                               'measuring the behavioural change, which this execution does not '
                               'authorize'),
        ),
        notes=('not a gate of the capability map: the upstream study whose retained arrays four '
               'of the gates read instead of their own extraction',)),
    'crossed_controls': GateDeclaration(
        gate='crossed_controls',
        record='archive/docs/shared/hierarchy-crossed-controls.md',
        elements=(
            GateElement('endpoint_qualification', 'reported',
                        quantities=('+0.30024 [+0.27330, +0.32841]',),
                        bound='the same anchor and the same folds as the admitted readout study, '
                              'reached through its own call sites rather than reimplemented'),
            GateElement('control_qualification', 'reported',
                        quantities=('+0.25277 [+0.22628, +0.27928]',),
                        bound='a control is judged on control-versus-control correlations alone, '
                              'so it cannot be tuned against the increment it is meant to absorb',
                        contributions=(
                            ControlContribution(
                                'C, the 444-coordinate composition control, fitted alone on '
                                'held-out clusters',
                                '+0.30024 [+0.27330, +0.32841]',
                                '[+0.27330, +0.32841]', True),
                            ControlContribution(
                                'C+L, the same control with the declared local-pattern block '
                                'added, which is lower than the C it extends',
                                '+0.25277 [+0.22628, +0.27928]',
                                '[+0.22628, +0.27928]', True),
                        )),
            GateElement('native_likelihood', 'reported',
                        quantities=('+0.0076 to +0.0118',),
                        bound='the likelihood enters as one scalar column over each control set, '
                              'so its increment is bounded by those controls and the label budget '
                              'rather than by the readout class'),
            GateElement('representation_readout', 'provisional',
                        quantities=('+0.10635 [+0.08910, +0.12391]',),
                        bound='the same fixed compressed linear family at the two retained '
                              'depths; the mutation-local profile block is the binding control and '
                              'absorbs the larger part of the increment'),
            GateElement('generalization', 'reported',
                        quantities=('33, 33, 33, 32 and 32',),
                        bound='held-out wild-type clusters under the admitted fold map, verified '
                              'identical to the admitted readout fits at all three split seeds',
                        remote='not addressed here: the clusters are the admitted 50%-identity '
                               'groups, and the evolution-and-adaptation gate carries the '
                               'identity-band and retrieval-depth stratification of this gain'),
            GateElement('causal_validation', 'absent', reason=_ABSENT_CAUSAL),
        ),
        notes=('not a gate of the capability map: the upstream study the evolution-and-adaptation '
               'gate re-aggregates per assay',)),
    'conformational_dynamics': GateDeclaration(
        gate='conformational_dynamics',
        record='archive/docs/shared/capability-map-plan.md',
        elements=tuple(
            GateElement(step, 'absent',
                        reason=('the conformational dynamics and allostery gate is not launched: its admission condition is a '
                                'qualified endpoint and controls competent on held-out groups, '
                                'and neither exists, so every step of the sequence is absent '
                                'rather than unresolved. It is opened early only if existing data '
                                'make a qualified endpoint immediately tractable.'))
            for step in CONTRACT_STEPS),
        notes=('declared so that an unlaunched gate is a recorded state rather than an omission; '
               'the question is whether a frozen model carries information about state transitions and long-range regulatory coupling',)),
    'molecular_recognition': GateDeclaration(
        gate='molecular_recognition',
        record='archive/docs/shared/capability-map-plan.md',
        elements=tuple(
            GateElement(step, 'absent',
                        reason=('the molecular recognition and function gate is not launched: its admission condition is a '
                                'qualified endpoint and controls competent on held-out groups, '
                                'and neither exists, so every step of the sequence is absent '
                                'rather than unresolved. It is opened early only if existing data '
                                'make a qualified endpoint immediately tractable.'))
            for step in CONTRACT_STEPS),
        notes=('declared so that an unlaunched gate is a recorded state rather than an omission; '
               'the question is whether a frozen model carries information about binding, catalysis and substrate specificity beyond family membership and retrieval support',)),
    'cellular_regulation': GateDeclaration(
        gate='cellular_regulation',
        record='archive/docs/shared/capability-map-plan.md',
        elements=tuple(
            GateElement(step, 'absent',
                        reason=('the cellular regulation and environment gate is not launched: its admission condition is a '
                                'qualified endpoint and controls competent on held-out groups, '
                                'and neither exists, so every step of the sequence is absent '
                                'rather than unresolved. It is opened early only if existing data '
                                'make a qualified endpoint immediately tractable.'))
            for step in CONTRACT_STEPS),
        notes=('declared so that an unlaunched gate is a recorded state rather than an omission; '
               'the question is whether a frozen model carries information about expression, localisation, degradation, post-translational modification and membrane context',)),
    'remote_homology': GateDeclaration(
        gate='remote_homology',
        record='archive/docs/R5/gate-remote-homology.md',
        elements=(
            GateElement('endpoint_qualification', 'reported',
                        quantities=('1.566 [1.301, 1.847]', '+0.1340 [+0.0799, +0.1921]'),
                        bound='the reproducible component exceeds the per-channel discordance by '
                              'about half the factor the development endpoint reaches, so a model '
                              'quantity has roughly half as much resolvable room here'),
            GateElement('control_qualification', 'reported',
                        quantities=('36.4% to 36.8%',),
                        bound='a candidate is kept only if its paired reduction in group-equal '
                              'held-out error over the standing set is positive at every one of '
                              'the three split seeds; two of five candidates were discarded',
                        contributions=(
                            ControlContribution('local chemistry',
                                                '+0.00472 [+0.00064, +0.00853]',
                                                '[+0.00064, +0.00853]', True),
                            ControlContribution('mutation-local profile, kept by the sign rule '
                                                'alone',
                                                '+0.00268 [−0.00100, +0.00618]',
                                                '[−0.00100, +0.00618]', False),
                            ControlContribution('the bounded profile restatement',
                                                '+0.00440 [+0.00002, +0.00908]',
                                                '[+0.00002, +0.00908]', True),
                            ControlContribution('composition, discarded',
                                                '−0.00286 [−0.01098, +0.00509]',
                                                '[−0.01098, +0.00509]', False),
                            ControlContribution('the nonlinear-additive response, discarded',
                                                '−0.02845 [−0.03894, −0.01750]',
                                                '[−0.03894, −0.01750]', True),
                        )),
            GateElement('native_likelihood', 'unresolved',
                        quantities=('0 of 33 arms',),
                        bound='the close-identity positive control did not fire, and a stratum on '
                              'which the pipeline resolves nothing gates the reading of the one '
                              'on which two arms do'),
            GateElement('representation_readout', 'provisional',
                        quantities=('33 of 33, with 0 failures and 0 no-record cells',),
                        bound='one compressed linear readout of four pooled block summaries at '
                              'the retained depths; the full-width states are retained so the '
                              'revisit is a refit rather than a re-extraction'),
            GateElement('generalization', 'reported',
                        quantities=('100 family groups are entirely remote and 66 entirely close',),
                        bound='both strata clear the eight-unit percentile floor, and the mixed '
                              'stratum holds 45% of the variants on 7% of the units, so it is '
                              'reported and read as neither',
                        remote='this gate is the remote stratification: its cohort populates the '
                               'identity bands every earlier cohort left at 1, 2 and 12 groups, '
                               'which is what it exists to do'),
            GateElement('causal_validation', 'absent', reason=_ABSENT_CAUSAL),
        ),
        notes=('one of the four gates whose representation cells are retained at the two hooked '
               'depths only, so a depth selection cannot be applied to it without a fresh '
               'extraction of its own cohort',)),
    'external_confirmation': GateDeclaration(
        gate='external_confirmation',
        record='archive/docs/R1/external-confirmation.md',
        elements=(
            GateElement('endpoint_qualification', 'reported',
                        quantities=('+0.0159 [−0.0710, +0.1087]',),
                        bound='the replicate offset is indistinguishable from zero, so the two '
                              'replicates are interchangeable in level; the endpoint is a '
                              'one-estimate level rather than a two-estimate difference, because '
                              'the wild-type reference is identically zero by construction'),
            GateElement('control_qualification', 'reported',
                        quantities=('52.5% to 52.9%', '+0.10096 [+0.09227, +0.10928]'),
                        bound='every one of the five candidates qualified and nothing was '
                              'discarded, so the two licensed cells coincide here and the crossed '
                              'sensitivities the development gate reports do not arise',
                        contributions=(
                            ControlContribution('the mutation-local profile, this endpoint\'s '
                                                'strongest control',
                                                '+0.00892 [+0.00639, +0.01132]',
                                                '[+0.00639, +0.01132]', True),
                            ControlContribution('composition, kept by the sign rule alone',
                                                '+0.00104 [-0.00048, +0.00253]',
                                                '[-0.00048, +0.00253]', False),
                        )),
            GateElement('native_likelihood', 'absent',
                        reason='the extraction is in flight, 8 of 33 arms complete when last '
                               'probed, so no per-arm cell exists yet; what the likelihood cell '
                               'awaits is wall clock alone, the cohort, plan, controls and '
                               'profiles being frozen and pushed'),
            GateElement('representation_readout', 'absent',
                        reason='as for the likelihood cell, the extraction is in flight; every '
                               'cell carries --keep-full-features so a readout-class '
                               'recomputation will be a refit rather than a second forward pass, '
                               'and a depth selection will still need a fresh extraction'),
            GateElement('generalization', 'reported',
                        quantities=('350, 115',),
                        bound='the screen removes 28 sequence-overlapping domains entirely and a '
                              'further 66 on a 50%/80% rule against both development sources, '
                              'which is stricter than the frozen contract\'s own 30% edge; at '
                              'that weaker threshold 140 of the 494 would cross',
                        remote='the negative statement rests on exhaustive query and subject sets '
                               '-- every retained wild type against every development target -- '
                               'which is what such a statement requires; homology below the rule '
                               'is undetected'),
            GateElement('causal_validation', 'absent', reason=_ABSENT_CAUSAL),
        ),
        notes=('one of the four gates whose representation cells are retained at the two hooked '
               'depths only; its extraction is also the long pole, at 4.2 times the stability '
               'cohort\'s residue workload per arm',)),
    'global_additive_context': GateDeclaration(
        gate='global_additive_context',
        record='archive/docs/R5/gate-global-context.md',
        elements=(
            GateElement('endpoint_qualification', 'reported',
                        quantities=('6.7e-16 kcal²/mol²',),
                        bound='nothing is re-extracted and no model inference runs; the fit '
                              'refuses any run whose row-identity or fold-identity digest differs '
                              'from the admitted per-arm fit it must compose with'),
            GateElement('control_qualification', 'unresolved',
                        quantities=('+0.00434 [−0.00391, +0.01268]',),
                        bound='the declared control is measurably informative only at the same '
                              'order of magnitude as the model increments it is meant to absorb, '
                              'and its own contribution is not resolved above zero at all three '
                              'seeds',
                        contributions=(
                            ControlContribution('X, the global-context block, over C+G+T',
                                                '+0.00434 [−0.00391, +0.01268]',
                                                '[−0.00391, +0.01268]', False),
                            ControlContribution('W, the bounded comparator, over C+G+T, which '
                                                'lowers its own baseline',
                                                '−0.01033 [−0.01745, −0.00401]',
                                                '[−0.01745, −0.00401]', True),
                        )),
            GateElement('native_likelihood', 'reported',
                        quantities=('+0.00792 [+0.00365, +0.01264]',),
                        bound='the block absorbs between nothing and about four percent of the '
                              'first-order gain, so a surviving gain survives a linear long-range '
                              'additive representation rather than every additive one'),
            GateElement('representation_readout', 'not_applicable',
                        reason='the gate is declared over model likelihood only: GATE_ADDITIONS '
                               'holds the likelihood interaction and nothing else, and no design '
                               'reads the projected representation blocks, so the representation '
                               'contrast is not measured rather than measured and reported'),
            GateElement('generalization', 'reported',
                        quantities=('129.2',),
                        bound='power is set by 217 site pairs and a Kish effective 129.2 on the '
                              '64-group support, never by the 8,192 cycle count',
                        remote='not addressed here: separation is a sequence-distance definition '
                               'and not a structural-contact annotation, and 99.98% of admitted '
                               'cycles sit on source-selected site pairs'),
            GateElement('causal_validation', 'absent', reason=_ABSENT_CAUSAL),
        )),
}
