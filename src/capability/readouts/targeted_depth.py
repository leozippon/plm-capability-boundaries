"""A targeted three-depth extraction for the two gates that cannot resolve depth.

The representation extractions of the folding-and-stability gate and the
residue-interaction gate each hooked exactly two blocks -- ``(L-1)//2`` and
``L-1`` -- on their own cohorts, so no other block's state was ever written for
either. A readout reassessment that selects a depth other than those two can be
applied to neither, and their representation verdicts are therefore conditional
on the admitted depth choice rather than on depth generally (audit L53).

This module plans the smallest extraction that removes that conditionality: at
each arm's **selected** depth, plus the two admitted depths retained alongside so
the new cells are comparable with the published ones on the same arrays. Three
depths per arm, not a full stack. The depth sweep's own measurement is what makes
that affordable -- capturing more blocks costs the same single forward pass per
sequence, and the additional cost is storage and fitting rather than inference --
so the GPU estimate below is the *measured* wall clock of the extraction that
already ran on each cohort, and the storage estimate is the only quantity that
grows with the number of depths.

Nothing here dispatches. The plan is parameterised on the reassessment's own
per-arm selection and refuses without it. Whether its lanes are queueable is read
from each extractor's own source rather than assumed: both now accept the
repeatable depth option, so the lanes parse, and the check stays because the same
wave whose numbers this module quotes lost three of its four first cells to
argparse for naming an option its snapshot did not define.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

#: The two gates this plan exists for, with the cohort each extraction covers.
#: ``states`` is the number of sequences one arm's extraction scores, read from
#: the retained manifests' own ``sequences_scored``, so the estimate rests on the
#: work that was done rather than on a recount of the cohort.
@dataclass(frozen=True)
class Cohort:
    gate: str
    #: Short tag that enters every lane label. The campaign queue derives a cell's
    #: output directory from the run id and the lane label alone, so two cohorts'
    #: cells for one arm would share a directory -- and the second would be
    #: reported `skipped-complete` against the first's artifact, writing nothing
    #: while looking complete. Measured: the pairwise gate cell did exactly that
    #: against the stability gate cell's manifest before this tag existed.
    tag: str
    label: str
    states: int
    groups: int
    rows: str
    #: Extraction plan, relative to the cluster project root the lane runs under
    #: (``campaign_rows`` renders it as ``{project_root}/{plan}``). These are the
    #: allocation's own paths, not this repository's: the four resolve in-pod and
    #: three of the four do not exist under the repository's later ``results/R*``
    #: and ``results/shared`` layout.
    plan: str
    plan_sha256: str
    #: Bare basename of the extraction stage. Resolved through
    #: :func:`scripts.capability.entrypoints.stage_path`, which finds it wherever
    #: it currently lives under ``scripts/capability``; the basename is what the
    #: campaign lane names and what :func:`declaration_digest` covers.
    extractor: str
    #: Retained-state destination, relative to the same cluster project root.
    output_root: str


#: **Frozen declaration.** :func:`declaration_digest` hashes every field below,
#: and ``archive/logs/shared/recomputation_20260924/targeted_depth_estimate.json``
#: records that digest. A mechanical path or package rewrite must not reach these
#: strings: they name the allocation the estimate was made against, not the
#: current repository layout. One such rewrite did reach them and was recovered
#: from a frozen pre-rename package.
COHORTS: dict[str, Cohort] = {
    'folding_stability': Cohort(
        gate='folding_stability',
        tag='fs',
        label='the frozen 101-background single-mutant stability cohort',
        states=25_957,
        groups=101,
        rows='25,856 single substitutions and 101 wild types',
        plan='results/gate_stability_20260924/extraction_plan.json',
        plan_sha256='283b4503ea166a3d61052bf68818c8aef162692ed3eb436a7a9864759a48763d',
        extractor='extract_stability_singles.py',
        output_root='results/external_baseline/<wave>/depth3-<tag>-<arm>/'),
    'remote_homology': Cohort(
        gate='remote_homology',
        tag='rh',
        label='the frozen 179-family-group MGnify-derived single-mutant cohort',
        states=6_596,
        groups=179,
        rows='6,291 single substitutions over 6,289 mutated sites and 305 backgrounds',
        plan='results/remote_homology_20260924/extraction_plan.json',
        plan_sha256='aa63ff6dc4b66ea51c9a495e1ca173ab22d622cbc4fe48bc8f92b6b012d05e48',
        extractor='extract_stability_singles.py',
        output_root='results/external_baseline/<wave>/depth3-<tag>-<arm>/'),
    'external_confirmation': Cohort(
        gate='external_confirmation',
        tag='ec',
        label='the frozen 96-family-group Domainome abundance cohort',
        states=109_996,
        groups=96,
        rows='428 domains at a 256-substitution cap each, 19 to 97 residues per domain',
        plan='results/external_confirmation_20260924/extraction_plan.json',
        plan_sha256='805adabe10200fdfb5992796c40d78c328a4f510da80ce56f0f663ee923578aa',
        extractor='extract_stability_singles.py',
        output_root='results/external_baseline/<wave>/depth3-<tag>-<arm>/'),
    'residue_interactions': Cohort(
        gate='residue_interactions',
        tag='ri',
        label='the frozen 64-background four-state double-mutant cycle cohort',
        states=12_977,
        groups=64,
        rows='8,192 cycles over 217 site pairs, as 12,977 measured states',
        plan='data/pairwise_epistasis/extraction_plan.json',
        plan_sha256='e338420f5df70143ccfc8d16ec5479a67b330ec35d36ea7c5965ea31a59e179c',
        extractor='extract_pairwise_epistasis.py',
        output_root='results/external_baseline/<wave>/depth3-<tag>-<arm>/'),
}

#: Bytes per retained state coordinate. Both waves wrote float32 features, and
#: the two bfloat16 arms cast to float32 on the way out, which the measured
#: archive shapes confirm.
BYTES_PER_COORDINATE = 4

#: Pooled summaries retained per depth: the arithmetic mean over the target's
#: residue-bearing tokens and the last residue-bearing token's state, which are
#: the two rules both admitted extractions used. The targeted extraction changes
#: which depths are hooked and nothing else.
SUMMARIES_PER_DEPTH = 2


@dataclass(frozen=True)
class ArmCost:
    """One arm's measured extraction cost on both cohorts.

    Transcribed from the retained manifests of the two waves that already ran:
    the stability gate's full-width retention pass and the pairwise extraction.
    ``final_block`` is read from each manifest's ``block_indices``, so the block
    count is the number the extraction actually saw rather than a declared one.
    """

    arm: str
    hidden_width: int
    middle_block: int
    final_block: int
    dtype: str
    stability_seconds: float
    pairwise_seconds: float

    @property
    def blocks(self) -> int:
        return self.final_block + 1

    @property
    def admitted_depths(self) -> tuple[int, int]:
        return (self.middle_block, self.final_block)

    def seconds(self, gate: str) -> float:
        """This arm's measured extraction wall clock on one cohort, in seconds.

        Only the two cohorts whose extractions completed arm by arm carry a
        per-arm measurement. The two later gates share this extractor and this
        panel but their per-arm seconds are not transcribed here, so a cost for
        them is derived from their state count against these measurements rather
        than invented per arm, and :func:`cohort_gpu_hours` is the only place
        that scaling happens.
        """
        if gate == 'folding_stability':
            return self.stability_seconds
        if gate == 'residue_interactions':
            return self.pairwise_seconds
        raise ValueError(f'{gate} carries no per-arm measured wall clock; use cohort_gpu_hours')


def _cost(arm, width, middle, final, dtype, stability, pairwise) -> ArmCost:
    return ArmCost(arm=arm, hidden_width=width, middle_block=middle, final_block=final,
                   dtype=dtype, stability_seconds=stability, pairwise_seconds=pairwise)


#: The 33 arms of both panels, with the hidden width and hooked blocks read from
#: the retained archives and manifests, and the measured wall clock of each
#: cohort's own completed extraction at batch size one on one H200. ZymCTRL is
#: absent because neither gate's panel carries it.
ARM_COSTS: dict[str, ArmCost] = {cost.arm: cost for cost in (
    _cost('bygpt5-base-en', 1536, 2, 5, 'float32', 597.1, 113.5),
    _cost('bygpt5-medium-en', 1536, 5, 11, 'float32', 522.2, 157.3),
    _cost('bygpt5-small-en', 1472, 1, 3, 'float32', 574.3, 101.4),
    _cost('dialogpt-small', 768, 5, 11, 'float32', 248.4, 69.3),
    _cost('galactica-1.3b', 2048, 11, 23, 'float32', 1745.3, 710.0),
    _cost('galactica-125m', 768, 5, 11, 'float32', 1004.5, 575.5),
    _cost('galactica-30b', 7168, 23, 47, 'float32', 5611.6, 2851.6),
    _cost('galactica-6.7b', 4096, 15, 31, 'float32', 2582.8, 1043.5),
    _cost('gpt2', 768, 5, 11, 'float32', 369.6, 64.5),
    _cost('gpt2-large', 1280, 17, 35, 'float32', 829.2, 144.3),
    _cost('gpt2-medium', 1024, 11, 23, 'float32', 533.4, 105.9),
    _cost('gpt2-xl', 1600, 23, 47, 'float32', 960.7, 209.3),
    _cost('instructprotein', 2048, 11, 23, 'float32', 1709.7, 648.6),
    _cost('llama-2-7b', 4096, 15, 31, 'float32', 1344.4, 691.8),
    _cost('llama-3.2-3b', 3072, 13, 27, 'float32', 583.6, 255.7),
    _cost('progen2-base', 1536, 13, 26, 'float32', 939.2, 296.9),
    _cost('progen2-large', 2560, 15, 31, 'float32', 1594.0, 434.6),
    _cost('progen2-medium', 1536, 13, 26, 'float32', 1274.5, 305.0),
    _cost('progen2-small', 1024, 5, 11, 'float32', 714.8, 136.7),
    _cost('progen2-xlarge', 4096, 15, 31, 'float32', 1917.8, 612.3),
    _cost('progen3-112m', 384, 4, 9, 'bfloat16', 1257.9, 367.5),
    _cost('progen3-3b', 1280, 11, 23, 'bfloat16', 3636.5, 832.4),
    _cost('prollama', 4096, 15, 31, 'float32', 1923.1, 690.4),
    _cost('prollama-stage-1', 4096, 15, 31, 'float32', 1710.8, 687.3),
    _cost('proteinglm-7b-clm', 4096, 17, 35, 'float32', 1772.9, 713.1),
    _cost('protgpt2', 1280, 17, 35, 'float32', 930.2, 179.8),
    _cost('protgpt3-1.3b', 1024, 8, 16, 'float32', 1577.4, 450.8),
    _cost('qwen2.5-0.5b', 896, 11, 23, 'float32', 614.2, 209.9),
    _cost('qwen2.5-0.5b-instruct', 896, 11, 23, 'float32', 798.5, 213.1),
    _cost('qwen2.5-32b', 5120, 31, 63, 'float32', 3571.3, 1612.3),
    _cost('qwen2.5-7b', 3584, 13, 27, 'float32', 1171.8, 405.3),
    _cost('qwen3-8b-base', 4096, 17, 35, 'float32', 1460.6, 459.7),
    _cost('rita-xl', 2048, 11, 23, 'float32', 3925.5, 194.8),
)}

#: Where the measured numbers come from, bound into every plan so a reader can
#: check them against the artefacts rather than against this table.
COST_PROVENANCE = {
    'remote_homology': ('results/external_baseline/20260924081450_45cbf26d2945/rh-<arm>/'
                        'manifest_<arm>.json: 33 of 33 cells complete, 6,596 sequences scored per '
                        'arm, 18,101.9 s in total measured in-pod'),
    'external_confirmation': ('results/external_baseline/20260924085613_5f8d32cf0368/ec-<arm>/'
                              'manifest_<arm>.json: 8 of 33 cells complete when measured, 109,996 '
                              'sequences scored per arm, 61,132.3 s over those 8. Lane order is a '
                              'declared descending parameter scale, so the completed 8 are the '
                              'largest arms and a per-arm mean over them overstates the remaining '
                              '25; the 33-arm total is estimated by scaling the stability cohort '
                              'instead, at 4.2 times its residue workload'),
    'folding_stability': ('results/external_baseline/20260924024010_7caaa5ac6216/full-<arm>/'
                          'manifest_<arm>.json: status complete, 101 of 101 backgrounds and '
                          '25,957 sequences scored for all 33 arms, elapsed_seconds and '
                          'block_indices per arm, features shaped (rows, 4, hidden width)'),
    'residue_interactions': ('results/pairwise_epistasis_20260924/extraction/<arm>/'
                             'manifest_<arm>.json: status complete, 12,977 sequences scored for '
                             'all 33 arms, elapsed_seconds and block_indices per arm'),
}

#: The option a lane needs to hook a depth beyond the admitted pair. Both
#: extractors now accept it, repeatably, through the one shared rule
#: ``readout_extraction.hooked_block_indices``: with the option absent the hooked
#: set is bit-identically what it was, the admitted pair stays at feature
#: positions 0 to 3, extras are deduplicated and appended in ascending order, the
#: resulting set is recorded in the receipt as ``block_indices`` and
#: ``feature_blocks``, and an index outside the stack is refused rather than
#: clamped. :func:`extractor_accepts_depth` still reads each stage's source rather
#: than assuming it, because the queueable marking must follow the code.
REQUIRED_EXTRACTOR_OPTION = '--extra-block-index'


def depths_for(arm: str, selected) -> tuple[int, ...]:
    """The depths one cell hooks, in the order the extractor hooks them.

    ``selected`` is one index for a targeted cell or a sequence of them for an
    explicit set, and either way the two admitted depths are included and come
    first, so an archive presents the admitted four feature blocks at positions 0
    to 3 whatever else it holds. Deduplicated, so a selection that lands on an
    admitted depth gives two depths rather than writing one block twice, and the
    plan reports how many it actually holds.

    The rule is :func:`~src.capability.readouts.readout_extraction.hooked_block_indices`, the
    one the extractors apply, rather than a second copy of it: a plan that
    computed the hooked set differently from the stage it dispatches would
    mis-report which blocks an archive holds. The index is validated against the
    arm own block count, read from the retained manifest rather than declared.
    """
    from .readout_extraction import hooked_block_indices
    cost = ARM_COSTS[arm]
    requested = [selected] if isinstance(selected, int) else list(selected)
    for index in requested:
        if not 0 <= int(index) < cost.blocks:
            raise ValueError(f'{arm}: depth {index} is outside 0..{cost.blocks - 1}; the arm has '
                             f'{cost.blocks} transformer blocks')
    return hooked_block_indices(cost.blocks, requested)


def every_block(arm: str) -> tuple[int, ...]:
    """Every zero-based block of one arm's stack, the depth-agnostic set.

    The reason this is worth pricing rather than dismissing: the GPU cost is the
    forward pass and does not follow the number of hooked blocks, so hooking all
    of them costs exactly what hooking three costs and differs only in storage.
    A pass over every block removes the depth conditionality permanently -- the
    gate becomes recomputable at any depth this or any later analysis selects --
    and it does not wait for the per-arm selection, which is otherwise the single
    blocking input for the whole recomputation.
    """
    return tuple(range(ARM_COSTS[arm].blocks))


def retained_bytes(gate: str, arm: str, depths) -> int:
    """Bytes of retained state for one arm at one depth set, at full hidden width.

    One float32 coordinate per state per summary per depth, which is the shape the
    retained archives already carry: ``(rows, depths x summaries, hidden width)``.
    """
    cohort = COHORTS[gate]
    cost = ARM_COSTS[arm]
    return (cohort.states * SUMMARIES_PER_DEPTH * len(tuple(depths))
            * cost.hidden_width * BYTES_PER_COORDINATE)


def arm_estimate(gate: str, arm: str, depths=None) -> dict:
    """One arm's GPU-hour and storage estimate for a targeted extraction.

    The GPU estimate does not depend on how many depths are hooked, because every
    depth is read from the same forward pass; it is the measured wall clock of the
    extraction that already ran on this cohort. Only the storage grows.
    """
    cost = ARM_COSTS[arm]
    hooked = tuple(sorted(set(depths))) if depths is not None else (-1, -2, -3)
    count = 3 if depths is None else len(hooked)
    # A cohort without per-arm wall clock reports none per arm rather than a
    # scaled one: the scaling is a cohort-level statement and splitting it over
    # arms would present a derived number in the shape of a measured one.
    seconds = cost.seconds(gate) if gate in MEASURED_COHORTS else None
    return dict(arm=arm, gate=gate, hidden_width=cost.hidden_width, blocks=cost.blocks,
                dtype=cost.dtype, admitted_depths=list(cost.admitted_depths),
                selected_depths=None if depths is None else list(hooked),
                depths_hooked=count,
                measured_seconds=seconds,
                gpu_hours=None if seconds is None else round(seconds / 3600.0, 4),
                retained_bytes=(COHORTS[gate].states * SUMMARIES_PER_DEPTH * count
                                * cost.hidden_width * BYTES_PER_COORDINATE),
                gpu_estimate_basis=(
                    'the measured wall clock of this cohort\'s completed extraction for this '
                    'arm, at batch size one on one H200; hooking a further block is read from the '
                    'same forward pass and adds no inference'
                    if seconds is not None else
                    'this cohort carries no per-arm wall clock here; its cost is a cohort-level '
                    'figure from cohort_gpu_hours and is not split over arms'))


def depth_agnostic_estimate() -> dict:
    """What a pass over every block of both cohorts costs, against a targeted one.

    Quoted beside the subsets so a reader can see what a fraction of the storage
    would buy. A subset is cheaper and reintroduces exactly the conditionality
    this pass exists to remove: a selected depth that falls on an unhooked block
    leaves the gate where L53 found it.
    """
    variants = {
        'every block': lambda cost: tuple(range(cost.blocks)),
        'every second block': lambda cost: tuple(range(0, cost.blocks, 2)),
        'every fourth block': lambda cost: tuple(range(0, cost.blocks, 4)),
        'the two admitted depths': lambda cost: cost.admitted_depths,
    }
    rows = {}
    for label, rule in variants.items():
        per_gate, total = {}, 0
        for gate in sorted(COHORTS):
            gate_bytes = sum(retained_bytes(gate, arm, rule(cost))
                             for arm, cost in ARM_COSTS.items())
            per_gate[gate] = round(gate_bytes / 2 ** 30, 2)
            total += gate_bytes
        rows[label] = dict(per_gate_gib=per_gate, total_bytes=total,
                           total_gib=round(total / 2 ** 30, 2),
                           depths_hooked=sum(len(rule(cost)) for cost in ARM_COSTS.values()),
                           resolves_any_depth=label == 'every block')
    costs = {gate: cohort_gpu_hours(gate) for gate in sorted(COHORTS)}
    hours = {gate: record['gpu_hours'] for gate, record in costs.items()}
    calibrated = {gate: record.get('gpu_hours_calibrated', record['gpu_hours'])
                  for gate, record in costs.items()}
    return dict(schema='d1_depth_agnostic_estimate_v1', variants=rows, gpu_hours=hours,
                gpu_cost=costs, gpu_hours_calibrated=calibrated,
                combined_gpu_hours=round(sum(hours.values()), 3),
                combined_gpu_hours_calibrated=round(sum(calibrated.values()), 3),
                blocks_over_the_panel=sum(cost.blocks for cost in ARM_COSTS.values()),
                per_depth_gib=round(sum(retained_bytes(gate, arm, (0,))
                                        for gate in COHORTS for arm in ARM_COSTS) / 2 ** 30, 2),
                gpu_invariance=('the GPU cost is identical for every variant, because every '
                                'hooked block is read from the same forward pass; only storage '
                                'follows the depth count'))


#: Cohorts whose per-arm wall clock is transcribed in :data:`ARM_COSTS`, so a
#: per-arm estimate is measured rather than derived.
MEASURED_COHORTS = ('folding_stability', 'residue_interactions')

#: Measured cohort-level extraction totals, in seconds over the 33 arms, read
#: from each wave's own manifests in-pod. The first two are the sums of the
#: per-arm figures in :data:`ARM_COSTS`. The third completed 33 of 33 cells and
#: its total is measured even though its per-arm seconds are not transcribed
#: here. The fourth is absent because its extraction is in flight, so its cost is
#: scaled and labelled so.
#:
#: The third entry is what calibrates that scaling, and it is why the scaling is
#: reported as first-order rather than as a figure: scaling the stability
#: cohort's total by the ratio of sequences scored predicts 3.530 GPU-hours for
#: the remote-homology cohort against the 5.028 measured, so the state-count
#: ratio understates by a factor of about 1.42 on a cohort whose targets are
#: longer per state. A scaled estimate elsewhere should be read as a lower-leaning
#: bound for the same reason.
MEASURED_COHORT_SECONDS = {
    'folding_stability': 50_007.8,
    'residue_interactions': 16_544.1,
    'remote_homology': 18_101.9,
}


def cohort_gpu_hours(gate: str) -> dict:
    """One cohort's extraction cost, measured where it can be and scaled where not.

    A targeted or depth-agnostic pass costs one forward pass per sequence, which
    is what the cohort's own completed extraction already cost, so the measured
    cohorts quote their own wall clock. The two later gates ran the same extractor
    over the same 33 arms on their own cohorts; their per-arm seconds are not
    transcribed here, so their cost is the stability cohort's measured total
    scaled by the ratio of sequences scored. That is a scaling and is labelled
    one: a longer target costs more per forward than a shorter one, so the ratio
    of state counts is a first-order estimate and not a measurement.
    """
    if gate in MEASURED_COHORT_SECONDS:
        seconds = MEASURED_COHORT_SECONDS[gate]
        return dict(gate=gate, gpu_hours=round(seconds / 3600.0, 3), basis='measured',
                    seconds=round(seconds, 1), per_arm_measured=gate in MEASURED_COHORTS,
                    note=('the wall clock this cohort\'s own completed extraction recorded, at '
                          'batch size one on one H200'))
    reference = 'folding_stability'
    seconds = MEASURED_COHORT_SECONDS[reference]
    ratio = COHORTS[gate].states / COHORTS[reference].states
    # The one cohort that is both scaled-predictable and measured shows the
    # scaling's direction: it understates by about 1.42, so the same correction is
    # carried here as a range rather than folded silently into the point.
    correction = (MEASURED_COHORT_SECONDS['remote_homology']
                  / (seconds * COHORTS['remote_homology'].states / COHORTS[reference].states))
    return dict(gate=gate, gpu_hours=round(seconds * ratio / 3600.0, 3), basis='scaled',
                scaled_from=reference, state_ratio=round(ratio, 3),
                calibration_factor=round(correction, 3),
                gpu_hours_calibrated=round(seconds * ratio * correction / 3600.0, 3),
                note=('scaled from the stability cohort\'s measured total by the ratio of '
                      'sequences scored, because this cohort\'s extraction is still in flight. '
                      'The one cohort that is both scaled-predictable and measured shows the '
                      'state-count ratio understating by about 1.42, so the calibrated figure is '
                      'the one to plan against and the plain scaling is the lower bound'))


def estimate(gate: str, selection: dict[str, int] | None = None) -> dict:
    """The whole targeted extraction's estimate for one gate.

    With no selection the storage is quoted at three depths per arm, which is the
    upper bound: a selection that coincides with an admitted depth hooks two.
    """
    cohort = COHORTS[gate]
    # A selection covering part of the panel estimates that part. A gate cell and
    # the wave that follows it are complementary arm lists, and an estimate that
    # silently spanned the whole panel would misreport either one.
    roster = sorted(ARM_COSTS if selection is None else selection)
    arms = []
    for arm in roster:
        depths = None if selection is None else depths_for(arm, selection[arm])
        arms.append(arm_estimate(gate, arm, depths))
    cohort_hours = cohort_gpu_hours(gate)
    total_hours = cohort_hours['gpu_hours']
    total_bytes = sum(row['retained_bytes'] for row in arms)
    return dict(gate=gate, cohort=cohort.label, states_per_arm=cohort.states,
                groups=cohort.groups, rows=cohort.rows, arms=len(arms),
                depths_per_arm='three: the selected depth and the two admitted ones'
                               if selection is not None else
                               'three, quoted as the upper bound; a selection that lands on an '
                               'admitted depth hooks two',
                total_gpu_hours=round(total_hours, 3),
                total_retained_bytes=total_bytes,
                total_retained_gib=round(total_bytes / 2 ** 30, 2),
                longest_arm=(max(arms, key=lambda row: row['gpu_hours'])['arm']
                             if gate in MEASURED_COHORTS else None),
                cost_provenance=COST_PROVENANCE[gate], gpu_cost=cohort_hours,
                per_arm=arms)


def extractor_accepts_depth(source: Path | str) -> bool:
    """Whether an extractor already takes the depth option a targeted lane needs.

    Read from the source rather than assumed. A lane that names an option its
    stage does not define is refused by argparse with exit 2, which is how three
    of the four cells of the first full-width retention dispatch were lost; a
    manifest builder that could not see the difference would repeat it. A source
    that cannot be read is refused rather than reported as not accepting the
    option, because those two answers are not the same answer.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(
            f'extractor source not found, so whether it accepts '
            f'{REQUIRED_EXTRACTOR_OPTION} cannot be read: {path}')
    return REQUIRED_EXTRACTOR_OPTION in path.read_text(encoding='utf-8')


def campaign_rows(gate: str, selection: dict[str, int], *, project_root: str,
                  runtime: str, gpu: int = 0, gpus=None,
                  order_longest_first: bool = True) -> list[dict]:
    """One lane row per arm, in the campaign manifest's own column order.

    Cells are ordered longest-first by the measured wall clock of the same arm's
    completed extraction. That order reads a measured duration and no fitted
    outcome, and it is the order both existing waves used.

    ``gpus`` spreads the cells over several cards: the arms are dealt
    longest-first across the given card indices, so slot *s* holds one cell per
    card and the queue's own rule -- one card at most once per slot -- holds by
    construction. With one card the manifest is the serial one, which is what a
    gate cell wants; with four it is about a quarter of the wall clock, which is
    what a 32-arm wave wants. Neither changes any measurement: the card a cell
    lands on is execution metadata outside the measurement identity.
    """
    cohort = COHORTS[gate]
    # The roster is the selection's own arms, so a gate cell and the wave that
    # follows it are complementary lists and no arm is extracted twice.
    roster = list(selection)
    # Longest-first by the measured wall clock where this cohort has one, and by
    # the stability cohort's where it does not: the ordering reads a measured
    # duration and no fitted outcome either way, and an arm's relative cost is the
    # same ordering on any cohort because the panel and the extractor are the same.
    reference = gate if gate in MEASURED_COHORTS else 'folding_stability'
    arms = sorted(roster, key=lambda arm: -ARM_COSTS[arm].seconds(reference)) \
        if order_longest_first else sorted(roster)
    cards = [int(index) for index in (gpus or (gpu,))]
    if not cards or len(set(cards)) != len(cards):
        raise ValueError(f'card indices must be distinct and non-empty, got {cards}')
    rows = []
    for position, arm in enumerate(arms):
        slot, card = position // len(cards) + 1, cards[position % len(cards)]
        depths = depths_for(arm, selection[arm])
        extra = [depth for depth in depths if depth not in ARM_COSTS[arm].admitted_depths]
        args = [f'--plan {project_root}/{cohort.plan}']
        if cohort.plan_sha256:
            args.append(f'--expect-plan-sha256 {cohort.plan_sha256}')
        args += [f'--arm {arm}', '--keep-full-features']
        args += [f'{REQUIRED_EXTRACTOR_OPTION} {depth}' for depth in extra]
        rows.append(dict(slot=slot, key='depth3', gpu=card, stage=cohort.extractor,
                         label=f'depth3-{cohort.tag}-{arm}', env=runtime,
                         expect=f'manifest_{arm}.json', args=' '.join(args),
                         depths=list(depths), extra_depths=extra,
                         measured_seconds=(ARM_COSTS[arm].seconds(gate)
                                           if gate in MEASURED_COHORTS else None),
                         ordering_reference=reference))
    return rows


def render_campaign(rows: list[dict], header: str) -> str:
    """The campaign TSV, with its header comment and the declared column order."""
    lines = [line if line.startswith('#') else f'# {line}' for line in header.splitlines()]
    lines.append('# slot\tkey\tgpu\tstage\tlabel\tenv\texpect\targs')
    for row in rows:
        lines.append('\t'.join((str(row['slot']), row['key'], str(row['gpu']), row['stage'],
                                row['label'], row['env'], row['expect'], row['args'])))
    return '\n'.join(lines) + '\n'


def declaration_digest() -> str:
    """Content digest of the cohorts, the measured cost table and the depth rule."""
    payload = dict(
        cohorts={gate: dict(tag=c.tag, label=c.label, states=c.states, groups=c.groups, rows=c.rows,
                            plan=c.plan, plan_sha256=c.plan_sha256, extractor=c.extractor,
                            output_root=c.output_root)
                 for gate, c in sorted(COHORTS.items())},
        arms={arm: dict(hidden_width=c.hidden_width, middle_block=c.middle_block,
                        final_block=c.final_block, dtype=c.dtype,
                        stability_seconds=c.stability_seconds,
                        pairwise_seconds=c.pairwise_seconds)
              for arm, c in sorted(ARM_COSTS.items())},
        bytes_per_coordinate=BYTES_PER_COORDINATE, summaries_per_depth=SUMMARIES_PER_DEPTH,
        required_extractor_option=REQUIRED_EXTRACTOR_OPTION,
        cost_provenance=COST_PROVENANCE)
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()
