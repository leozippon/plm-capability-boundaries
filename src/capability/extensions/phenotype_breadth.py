"""Independent phenotype cohorts: admission, qualified controls, separated endpoints.

This module asks which *additional* biological phenotypes a checkpoint's native
likelihood can predict, under the admission discipline the anchor cohort and the
nested gates already use. Three rules carry the whole design.

1. **An independence floor comes before any model cost.** A cohort's independent
   unit is a sequence-homology group at the frozen 30% identity / 80% mutual
   coverage edge rule, not an assay file, a gene, a PDB complex or an
   exact-sequence component. :data:`INDEPENDENCE_FLOOR` is the project's own
   percentile-interval floor (:data:`..core.statistics.MINIMUM_BOOTSTRAP_UNITS`),
   the same floor that closed the DHFR (1 group) and TEV (7 group) candidates. A
   cohort below it is refused *and the refusal is recorded*; it is never quietly
   scored.

2. **A likelihood increment is admitted only over a qualified baseline.** The
   candidate control blocks are the nested gate's own
   (:data:`CANDIDATE_BLOCKS`): composition, mutation-local chemistry windows and
   the mutation-local evolutionary profile in its raw and bounded restatements.
   Each is offered in declared order to the standing set and kept only if its
   paired reduction in group-equal held-out error is positive at every split
   seed. A cohort with no evolutionary-profile coverage cannot clear the
   evolutionary-statistics explanation and is reported as such rather than
   admitted on local controls alone.

3. **Ranking and quantitative prediction are separate endpoints.** A likelihood
   difference has an arbitrary scale, so the ranking endpoint uses the
   within-background standardized-rank target the project uses everywhere, while
   the quantitative endpoint is licensed only when the cohort's label unit is
   traced and shared across its backgrounds, and its predictor is fitted on the
   endpoint's own units using training groups only. Rank-target error is never
   reported as physical phenotype error and labels in different units are never
   pooled into one error metric.

Nothing here reads a model output while a support, a grouping or a control set is
being decided: :func:`qualify` and :func:`qualify_controls` are label- and
model-blind in that order, and :func:`panel` refuses a model block whose state
coverage is not exact.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import rankdata

from ..context.profile_increment import correlation
from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS, bootstrap_unit_floor
from ..readouts.readout_analysis import ALPHAS, family_folds, ridge_predict, row_weights
from ..stability import stability_gate as gate
from . import phenotype_homology as homology
from .overlap import rerank

SCHEMA = 'phenotype_breadth_v1'
PLAN_SCHEMA = 'phenotype_breadth_state_plan_v1'

#: The independence floor, taken from the project's percentile-interval floor so
#: that the number is declared in exactly one place.
INDEPENDENCE_FLOOR = MINIMUM_BOOTSTRAP_UNITS

#: Outer/inner partition shape and split seeds, as the frozen panels use them.
OUTER_FOLDS, INNER_FOLDS = 5, 4
SPLIT_SEEDS = (20260923, 20260924, 20260925)

#: Ranking needs at least this many retained rows in a background before a
#: within-background rank correlation is defined at all. Label-blind.
MIN_BACKGROUND_ROWS = 3

#: Packed-token budget of every frozen interface. A state longer than this
#: cannot be scored under the admitted budget and is refused before any GPU cost.
RESIDUE_BUDGET = 1000

#: Label-blind cap on retained variants per independent group, with the
#: stable-hash draw seed. The value is the external-confirmation panel's own
#: ``VARIANT_CAP``, so the breadth cohorts carry no more rows per unit than the
#: frozen abundance cohort does. A group-unit interval is not limited by the
#: number of rows inside a group, and an uncapped cohort would spend its whole
#: scoring budget on its three largest proteins. Declared before any model
#: effect is read.
VARIANT_CAP = 256
DRAW_SEED = 20261008

#: Control blocks, in the nested gate's own declaration. ``ident`` and ``geom``
#: are the base every candidate is qualified over; the candidates are competing
#: non-model explanations, never mechanisms. ``prof``/``prof2`` are the raw and
#: bounded restatements of the same mutation-local evolutionary profile.
BASE_BLOCKS = ('ident', 'geom')
CANDIDATE_BLOCKS = ('comp', 'chem', 'prof', 'prof2')
CONTROL_BLOCKS = BASE_BLOCKS + CANDIDATE_BLOCKS

#: The model addition: one scalar column. ``M_nats`` is the native likelihood of
#: the mutant state minus that of its wild type, in nats, which removes the
#: protein-level offset by construction. ``M_rank`` is its within-background
#: standardized rank, the rendering every frozen ranking panel uses.
MODEL_COLUMNS = ('M_nats', 'M_rank')

#: Simultaneous-inference settings. The band is the 95th percentile of the
#: maximum absolute centered bootstrap deviation over the declared contrast
#: family under one shared draw of the independent groups, which is the rule
#: ``extensions.progression.joint_bootstrap`` and ``extensions.overlap`` apply.
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261008

#: Measured-quantity classes of the VenusMutHub activity partitions and the
#: monotone direction of each. The direction is a property of the named physical
#: quantity, not of a dataset title: a larger turnover number is faster
#: catalysis. A quantity whose direction depends on an unrecorded objective is
#: refused rather than guessed, and no quantity here shares a unit across
#: partitions, so the activity cohort licenses ranking only.
#: The keywords are tested in this order, so a preference ratio is recognised
#: before the efficiency it is a ratio of and a ``kcatkm`` partition is never
#: read as a ``km`` one.
ACTIVITY_DIRECTIONS: dict[str, dict[str, Any]] = {
    'ratio': {'direction': None, 'reason': (
        'an enantiomeric or substrate preference ratio has no phenotype direction without the '
        'objective the authors optimised; a larger ratio is not "more active"')},
    'kcatkm': {'direction': 1, 'reason': 'catalytic efficiency kcat/Km; larger is more efficient'},
    'kcat': {'direction': 1, 'reason': 'turnover number; larger is faster catalysis'},
    'vmax': {'direction': 1, 'reason': 'maximal velocity; larger is faster catalysis'},
    'km': {'direction': None, 'reason': (
        'a Michaelis constant is an affinity, and lower Km is tighter binding but not more '
        'catalysis; the activity direction is not determined by Km alone')},
    'activity': {'direction': 1, 'reason': (
        'relative or specific catalytic activity as the partition reports it; larger is more active')},
}

LIMITATIONS = [
    'A sequence-homology group at 30% identity over 80% of both sequences is a conservative '
    'independence unit, not an independent evolutionary or experimental history; assay files, '
    'genes, PDB complexes, conditions and exact-sequence components are not families.',
    'Within-background transforms of the model feature (the standardized rank) and of the '
    'ranking target are label-blind but read every row of a background, including held-out rows; '
    'this is the established convention of the frozen panels, not an independent choice.',
    'Intervals condition on fitted out-of-fold predictions. They carry no training, tuning or '
    'checkpoint-selection uncertainty, and split seeds are sensitivity analyses, not replicates.',
    'A qualified control set is a competent predictor of held-out groups, not a complete account '
    'of non-model explanations. Where the evolutionary-profile block fails qualification an '
    'evolutionary-statistics explanation of the increment is not excluded.',
    'Rank-target squared error is not physical phenotype error, and the quantitative endpoint is '
    'reported only where the label unit is traced and shared across the cohort backgrounds.',
]


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def array_digest(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def stable_draw_key(*parts: str) -> int:
    """Deterministic label-blind draw order; depends only on identity strings."""
    digest = hashlib.sha256(('|'.join((str(DRAW_SEED), *parts))).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big')


@dataclass(frozen=True)
class PhenotypeRow:
    """One measured single substitution of one background, with its identity.

    ``background`` is the ranking background (the experimental partition inside
    which a rank is meaningful). ``unit`` is the identity whose homology decides
    the independent group; it is a wild-type sequence hash, never a file name.
    """

    cohort: str
    row_id: str
    background: str
    unit: str
    wildtype: str
    mutant_sequence: str
    position: int
    wt_aa: str
    mt_aa: str
    label: float
    direction: int
    label_unit: str | None
    source_sha256: str

    def oriented_label(self) -> float:
        """The label in the declared 'more of the phenotype is larger' direction."""
        return float(self.direction) * float(self.label)

    def identity(self) -> dict[str, Any]:
        return {'cohort': self.cohort, 'row_id': self.row_id, 'background': self.background,
                'unit': self.unit, 'position': self.position, 'wt_aa': self.wt_aa,
                'mt_aa': self.mt_aa, 'wildtype_sha256': sha_text(self.wildtype),
                'mutant_sha256': sha_text(self.mutant_sequence),
                'source_sha256': self.source_sha256}


@dataclass
class Cohort:
    """A candidate independent phenotype cohort and everything decided about it."""

    key: str
    phenotype: str
    endpoint: str
    source: str
    rows: list[PhenotypeRow] = field(default_factory=list)
    quantitative_unit: str | None = None
    construct_status: str = 'verified'
    #: A written justification for measuring on a construct that is not verified
    #: against the one the labels were measured on. Where it is absent an
    #: unverified construct is a blocker; where it is present the mismatch is
    #: recorded as an irreducible limitation and the cohort may proceed. Making
    #: the tolerance explicit is the point: it cannot be granted by silence.
    construct_tolerated: str | None = None
    notes: list[str] = field(default_factory=list)
    refusals: dict[str, int] = field(default_factory=dict)


def validate_row(row: PhenotypeRow) -> None:
    """Exact single-substitution identity; an identity error is never an exclusion."""
    homology.canonical(row.wildtype)
    homology.canonical(row.mutant_sequence)
    if len(row.wildtype) != len(row.mutant_sequence):
        raise ValueError(f'{row.row_id}: substitution changes sequence length')
    if not 1 <= row.position <= len(row.wildtype):
        raise ValueError(f'{row.row_id}: position outside the wild type')
    index = row.position - 1
    if row.wildtype[index] != row.wt_aa or row.mutant_sequence[index] != row.mt_aa:
        raise ValueError(f'{row.row_id}: declared substitution disagrees with the sequences')
    if row.wt_aa == row.mt_aa:
        raise ValueError(f'{row.row_id}: identity substitution is not a mutation')
    differing = [i for i, (a, b) in enumerate(zip(row.wildtype, row.mutant_sequence)) if a != b]
    if differing != [index]:
        raise ValueError(f'{row.row_id}: not a strict single substitution')
    if row.direction not in (-1, 1):
        raise ValueError(f'{row.row_id}: direction must be declared as -1 or +1')
    if not np.isfinite(row.label):
        raise ValueError(f'{row.row_id}: nonfinite label')


def independence_groups(units: dict[str, str], runtime: Path) -> dict[str, Any]:
    """Connected components of the frozen 30%/80% family edge, over wild types.

    ``units`` maps a unit identity to its wild-type sequence. The edge rule and
    the exact alignment are the committed phenotype-homology ones, so a cohort's
    independence is measured by the same instrument as every other cohort's.
    """

    names = sorted(units)
    aligner = homology.ExactAligner(runtime)
    aligner.prepare([units[name] for name in names])
    union = homology.Union(names)
    edges = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            identity = homology.relation(units[left], units[right])
            if identity is not None:
                union.join(left, right, identity)
                edges.append({'a': left, 'b': right, 'kind': identity})
                continue
            alignment = aligner.align(units[left], units[right])
            if alignment.family():
                union.join(left, right, 'family_edge')
                edges.append({'a': left, 'b': right, 'kind': 'family_edge',
                              'identity': round(alignment.identity, 4),
                              'coverage': [round(alignment.coverage_a, 4),
                                           round(alignment.coverage_b, 4)]})
    assignment = {name: union.find(name) for name in names}
    sizes = Counter(assignment.values())
    return {'rule': ('30% identity over 80% of both sequences with at least '
                     f'{homology.MIN_ALIGNED_RESIDUES} paired residues, plus exact and '
                     'containment identity'),
            'backend': aligner.provenance, 'units': len(names),
            'groups': len(sizes), 'assignment': assignment, 'edges': edges,
            'largest_group_units': max(sizes.values()) if sizes else 0}


def cap_rows(rows: Sequence[PhenotypeRow], groups: dict[str, str],
             *, cap: int = VARIANT_CAP) -> tuple[list[PhenotypeRow], dict[str, Any]]:
    """Label-blind retention of at most ``cap`` rows per independent group.

    The draw order is a stable hash of the row identity only, so it does not
    depend on a label, a model output or the iteration order of a dict.
    """

    if cap <= 0:
        raise ValueError('a positive cap is required')
    by_group: dict[str, list[PhenotypeRow]] = defaultdict(list)
    for row in rows:
        by_group[groups[row.unit]].append(row)
    kept: list[PhenotypeRow] = []
    dropped = 0
    for group, members in sorted(by_group.items()):
        ordered = sorted(members, key=lambda r: (stable_draw_key(r.cohort, r.row_id), r.row_id))
        kept.extend(ordered[:cap])
        dropped += max(len(ordered) - cap, 0)
    kept.sort(key=lambda r: (r.cohort, r.background, r.row_id))
    return kept, {'cap': cap, 'draw_seed': DRAW_SEED, 'dropped_rows': dropped,
                  'draw': 'stable sha256 of (seed, cohort, row id); label- and model-blind'}


def screen_backgrounds(rows: Sequence[PhenotypeRow]) -> tuple[list[PhenotypeRow], dict[str, int]]:
    """Drop states over the token budget, repeated variants and unrankable backgrounds.

    A ranking background that holds one substitution more than once has an
    undeclared replicate, endpoint or condition structure: its rank is not
    defined until the source's own structure is declared. Every row of such a key
    is dropped, label-blind, and counted; nothing is averaged, because choosing a
    summary of discordant replicates is a declaration the source has to make.
    """

    refusals: Counter[str] = Counter()
    long = [r for r in rows if len(r.wildtype) > RESIDUE_BUDGET]
    refusals['state_over_residue_budget'] = len(long)
    retained = [r for r in rows if len(r.wildtype) <= RESIDUE_BUDGET]
    variant_counts = Counter((r.background, r.unit, r.position, r.mt_aa) for r in retained)
    repeated = {key for key, count in variant_counts.items() if count > 1}
    if repeated:
        refusals['repeated_background_variant_rows'] = sum(
            variant_counts[key] for key in repeated)
        retained = [r for r in retained
                    if (r.background, r.unit, r.position, r.mt_aa) not in repeated]
    counts = Counter(r.background for r in retained)
    small = [r for r in retained if counts[r.background] < MIN_BACKGROUND_ROWS]
    refusals['background_below_minimum_rows'] = len(small)
    retained = [r for r in retained if counts[r.background] >= MIN_BACKGROUND_ROWS]
    constant = set()
    for background, members in group_by(retained, lambda r: r.background).items():
        if len({round(r.oriented_label(), 12) for r in members}) < 2:
            constant.add(background)
    refusals['background_constant_label'] = sum(
        1 for r in retained if r.background in constant)
    retained = [r for r in retained if r.background not in constant]
    return retained, {k: int(v) for k, v in refusals.items() if v}


def group_by(rows: Iterable[Any], key) -> dict[Any, list[Any]]:
    out: dict[Any, list[Any]] = defaultdict(list)
    for row in rows:
        out[key(row)].append(row)
    return dict(out)


def qualify(cohort: Cohort, runtime: Path, *, floor: int = INDEPENDENCE_FLOOR) -> dict[str, Any]:
    """Decide admission of one cohort before any model quantity exists.

    Returns ``(record, retained_rows, grouping)``. The record always carries the
    support counts, the measured independent group count, the floor verdict and,
    for a refusal, the named recoverable blockers. A refused cohort still gets a
    full record: "too few independent units" is a finding about the cohort, not a
    reason to omit it.
    """

    for row in cohort.rows:
        validate_row(row)
    retained, refusals = screen_backgrounds(cohort.rows)
    refusals = {**cohort.refusals, **refusals}
    units = {row.unit: row.wildtype for row in retained}
    if len({row.unit: sha_text(row.wildtype) for row in retained}) != len(units):
        raise ValueError('a unit identity maps to more than one wild-type sequence')
    grouping = independence_groups(units, runtime) if units else {
        'units': 0, 'groups': 0, 'assignment': {}, 'edges': [], 'largest_group_units': 0,
        'rule': 'not evaluated: no retained row', 'backend': None}
    # A ranking background must sit inside one independent group. If it does not,
    # a rank is being taken across two biologically independent units and the
    # group weighting is not defined; that is a defect in how the cohort names its
    # background, not something to average away.
    spanning = {}
    for background, members in group_by(retained, lambda row: row.background).items():
        reached = sorted({grouping['assignment'][row.unit] for row in members})
        if len(reached) > 1:
            spanning[background] = reached
    if spanning:
        example = sorted(spanning)[0]
        raise ValueError(
            f'{cohort.key}: {len(spanning)} ranking backgrounds span more than one independent '
            f'group, for example {example!r} over {spanning[example]}')
    capped, cap_record = cap_rows(retained, grouping['assignment'])
    capped, post_refusals = screen_backgrounds(capped)
    for key, value in post_refusals.items():
        refusals[f'after_cap_{key}'] = value
    floor_record = bootstrap_unit_floor(grouping['groups'], minimum_units=floor)
    blockers: list[str] = []
    limitations: list[str] = []
    if floor_record['degenerate']:
        blockers.append(
            f"{grouping['groups']} independent homology groups is below the {floor}-group floor: "
            f"{floor_record['degenerate_reason']}")
    elif grouping['groups'] == floor:
        limitations.append(
            f'{floor} independent groups is exactly the floor and leaves no margin: one group '
            'lost to any later screen would make every interval on this cohort degenerate')
    if cohort.construct_status != 'verified':
        if cohort.construct_tolerated is None:
            blockers.append(f'construct identity is {cohort.construct_status}, and the cohort '
                            'declares no justification for measuring on it anyway')
        else:
            limitations.append(f'construct identity is {cohort.construct_status}; tolerated '
                               f'because {cohort.construct_tolerated}')
    directions = sorted({row.direction for row in capped})
    if not directions:
        blockers.append('no row survives the identity, budget and background screens')
    quantitative_reason = None
    if cohort.quantitative_unit is None:
        quantitative_reason = (
            'no traced label unit shared across backgrounds, so a squared-error metric would pool '
            'heterogeneous units; the quantitative endpoint is refused, not approximated')
    status = 'admitted' if not blockers else 'refused'
    return {
        'cohort': cohort.key, 'phenotype': cohort.phenotype, 'endpoint': cohort.endpoint,
        'source': cohort.source, 'status': status, 'blockers': blockers,
        'support': {'rows': len(capped), 'backgrounds': len({r.background for r in capped}),
                    'units': len({r.unit for r in capped}),
                    'independent_groups': grouping['groups'],
                    'source_rows': len(cohort.rows)},
        'independence': {k: v for k, v in grouping.items() if k != 'assignment'},
        'independence_floor': floor_record,
        'variant_cap': cap_record,
        'refusals': refusals,
        'ranking_licensed': status == 'admitted',
        'quantitative_licensed': status == 'admitted' and cohort.quantitative_unit is not None,
        'quantitative_unit': cohort.quantitative_unit,
        'quantitative_refusal': quantitative_reason,
        'construct_status': cohort.construct_status,
        'construct_tolerated': cohort.construct_tolerated,
        'cohort_limitations': limitations,
        'notes': list(cohort.notes),
        'rows_sha256': sha_text(json.dumps([r.identity() for r in capped], sort_keys=True)),
    }, capped, grouping


def state_plan(cohorts: dict[str, Sequence[PhenotypeRow]]) -> dict[str, Any]:
    """One deduplicated state table for every admitted cohort.

    A wild type shared by all of its mutants is scored once, and a sequence
    shared by two cohorts is scored once; the per-cohort index lists bind each
    row back to its two states by exact sequence bytes.
    """

    index: dict[str, int] = {}
    states: list[dict[str, Any]] = []

    def register(sequence: str) -> int:
        digest = sha_text(sequence)
        if digest not in index:
            index[digest] = len(states)
            states.append({'index': len(states), 'sequence': sequence,
                           'sequence_sha256': digest, 'length': len(sequence)})
        return index[digest]

    records = []
    for key in sorted(cohorts):
        rows = cohorts[key]
        pairs = []
        for row in rows:
            pairs.append({'row_id': row.row_id,
                          'wildtype_state': register(row.wildtype),
                          'mutant_state': register(row.mutant_sequence)})
        records.append({'cohort': key, 'rows': len(pairs), 'pairs': pairs})
    if any(state['length'] > RESIDUE_BUDGET for state in states):
        raise ValueError('a planned state exceeds the declared residue budget')
    return {'schema': PLAN_SCHEMA, 'residue_budget': RESIDUE_BUDGET,
            'states': states, 'cohorts': records,
            'states_sha256': sha_text(json.dumps([s['sequence_sha256'] for s in states])),
            'measurement': ('native next-token likelihood of each state under each checkpoint, '
                            'summed over the interface scored span, in nats')}


def profile_catalogue(cohorts: dict[str, Sequence[PhenotypeRow]]) -> tuple[list[dict], dict]:
    """Query catalogue and background plan for the committed profile pipeline.

    The evolutionary-profile control is built by the repository's own
    ``search_pairwise_homologs.py`` and ``build_pairwise_profile_features.py``,
    against the retained UniRef50 DIAMOND index, so this module declares the
    queries and never restates the profile definition.
    """

    wildtypes: dict[str, str] = {}
    for key in sorted(cohorts):
        for row in cohorts[key]:
            name = f'{key}:{row.unit[:16]}'
            previous = wildtypes.setdefault(name, row.wildtype)
            if previous != row.wildtype:
                raise ValueError(f'{name}: two wild-type sequences under one profile name')
    catalogue = [{'WT_name': name, 'sequence': wildtypes[name]} for name in sorted(wildtypes)]
    plan = {'schema': 'phenotype_breadth_profile_plan_v1',
            'backgrounds': [{'name': name, 'wildtype': wildtypes[name]} for name in sorted(wildtypes)]}
    return catalogue, plan


def profile_name(cohort: str, unit: str) -> str:
    return f'{cohort}:{unit[:16]}'


def control_blocks(rows: Sequence[PhenotypeRow], profiles: dict[str, Any] | None) -> dict[str, np.ndarray]:
    """The declared control blocks, built by the stability gate's own functions."""
    blocks: dict[str, list] = {name: [] for name in CONTROL_BLOCKS}
    for row in rows:
        index = row.position - 1
        profile = None if profiles is None else profiles.get(profile_name(row.cohort, row.unit))
        blocks['ident'].append(gate.identity_block(row.wildtype, index, row.mt_aa))
        blocks['geom'].append(gate.geometry_block(index, len(row.wildtype)))
        blocks['comp'].append(gate.composition_block(row.wildtype, row.mutant_sequence))
        blocks['chem'].append(gate.chemistry_block(row.wildtype, index, row.mt_aa))
        blocks['prof'].append(gate.profile_block(profile, row.wildtype, index, row.mt_aa))
        blocks['prof2'].append(gate.profile_bounded_block(profile, row.wildtype, index, row.mt_aa))
    out = {}
    for name, values in blocks.items():
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) != len(rows) or not np.isfinite(matrix).all():
            raise ValueError(f'invalid {name} control block')
        out[name] = matrix
    return out


def read_arm_scores(directories: Iterable[Path]) -> dict[str, dict[str, float]]:
    """Per arm, the exact state-sequence hash to native likelihood in nats.

    A scoring directory without a completion record, with a non-complete status,
    with a truncated plan or with an archive whose digest disagrees with its own
    record is refused: a partial scoring run is an interface check, not an input
    to a fit.
    """

    from ..core.io import sha256_file

    scores: dict[str, dict[str, float]] = {}
    for directory in directories:
        directory = Path(directory)
        completion = directory / 'phenotype_scoring.json'
        if not completion.is_file():
            raise ValueError(f'{directory}: no completion record; a partial run is refused')
        record = json.loads(completion.read_text(encoding='utf-8'))
        if record.get('status') != 'complete':
            raise ValueError(f'{directory}: scoring status is {record.get("status")!r}')
        if record.get('partial_plan'):
            raise ValueError(f'{directory}: scored a truncated plan; interface checks are not fits')
        for arm_record in record['arms']:
            archive = directory / arm_record['archive']
            if sha256_file(archive) != arm_record['archive_sha256']:
                raise ValueError(f'{archive}: digest disagrees with the scoring record')
            with np.load(archive, allow_pickle=False) as data:
                hashes = [str(value) for value in data['sequence_sha256']]
                values = data['likelihood_nats'].astype(np.float64)
            table = dict(zip(hashes, (float(v) for v in values)))
            if len(table) != len(hashes):
                raise ValueError(f'{archive}: duplicate state hash')
            if arm_record['arm'] in scores:
                raise ValueError(f'{arm_record["arm"]}: scored in two input directories')
            scores[arm_record['arm']] = table
    if not scores:
        raise ValueError('no arm scores supplied')
    return scores


def prune_constant_columns(blocks: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    """Drop control columns that are constant on this cohort's own support.

    A column with no variation inside the fitted support carries no information
    the unpenalised intercept does not already carry, so removing it leaves every
    prediction unchanged. It is removed anyway, because the ridge solves through a
    symmetric eigendecomposition and a design padded with many exactly identical
    zero columns is the one input on which that decomposition fails to converge.
    On a small cohort the substitution-identity block is mostly such columns: the
    400 directed substitution types cannot all be observed in a few thousand rows.

    The census is returned so the removal is visible: it is a property of the
    support, decided without reading a label or a model output.
    """

    pruned, census = {}, {}
    for name, matrix in blocks.items():
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or not len(matrix):
            raise ValueError(f'invalid {name} block')
        keep = np.flatnonzero(~np.all(matrix == matrix[0], axis=0))
        fully_constant = not keep.size
        if fully_constant:
            # A block with no variation anywhere is kept as a single column so
            # that a design naming it still resolves. One constant column is
            # numerically harmless; a hundred identical ones are not. The
            # evolutionary-profile blocks look like this when no profile file was
            # supplied, which the fit records separately as an absent control.
            keep = np.asarray([0])
        census[name] = {'columns': int(matrix.shape[1]), 'retained': int(keep.size),
                        'constant_columns_removed': int(matrix.shape[1] - keep.size),
                        'fully_constant': bool(fully_constant)}
        pruned[name] = matrix[:, keep]
    census['rule'] = ('a column with no variation on the fitted support is removed before the '
                      'ridge; it is absorbed by the unpenalised intercept and changes no '
                      'prediction, and its presence is what makes the eigendecomposition '
                      'ill-conditioned on a small cohort')
    return pruned, census


def model_block(rows: Sequence[PhenotypeRow], scores: dict[str, float],
                backgrounds: np.ndarray) -> dict[str, np.ndarray]:
    """``M_nats`` and ``M_rank`` from exact state likelihoods; no partial coverage."""
    values = np.empty(len(rows), dtype=np.float64)
    for position, row in enumerate(rows):
        wild = scores.get(sha_text(row.wildtype))
        mutant = scores.get(sha_text(row.mutant_sequence))
        if wild is None or mutant is None:
            raise ValueError(f'{row.row_id}: the arm has no exact likelihood for both states')
        values[position] = mutant - wild
    if not np.isfinite(values).all():
        raise ValueError('nonfinite likelihood difference')
    return {'M_nats': values.reshape(-1, 1),
            'M_rank': rerank(values, backgrounds).reshape(-1, 1)}


def outer_signature(groups: np.ndarray, seed: int) -> list[tuple]:
    """Five group-disjoint outer folds, each with four inner validation folds."""
    unique = sorted({g.item() if isinstance(g, np.generic) else g for g in groups})
    if len(unique) < OUTER_FOLDS:
        raise ValueError(f'{OUTER_FOLDS} outer folds require at least that many groups')
    signature = []
    for position, held in enumerate(family_folds(unique, OUTER_FOLDS, seed)):
        training = [g for g in unique if g not in set(held)]
        if len(training) < INNER_FOLDS:
            raise ValueError('too few training groups for the inner partition')
        inner = family_folds(training, INNER_FOLDS, seed + 1 + position)
        signature.append((position, sorted(held), training, inner))
    return signature


def fold_predictions(x: np.ndarray, target: np.ndarray, backgrounds: np.ndarray,
                     groups: np.ndarray, signature: Sequence[tuple]) -> tuple[np.ndarray, list[dict]]:
    """Group-held-out ridge with the nested inner penalty choice, as the panels do."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(target) or not np.isfinite(x).all():
        raise ValueError('unaligned or nonfinite design')
    prediction = np.full(len(x), np.nan)
    records = []
    for fold, held, training, inner in signature:
        train = np.flatnonzero(np.isin(groups, training))
        test = np.flatnonzero(np.isin(groups, held))
        if not len(train) or not len(test):
            raise ValueError('an empty outer fold')
        losses = np.zeros(len(ALPHAS))
        for validation in inner:
            valid = np.flatnonzero(np.isin(groups, validation))
            fit = train[~np.isin(groups[train], validation)]
            if not len(valid) or not len(fit):
                raise ValueError('an empty inner fold')
            predicted = ridge_predict(x[fit], target[fit],
                                      row_weights(backgrounds[fit], groups[fit]), x[valid], ALPHAS)
            weights = row_weights(backgrounds[valid], groups[valid])
            losses += len(validation) * (weights[:, None] * (predicted - target[valid, None]) ** 2).sum(0)
        losses /= len(training)
        # Largest penalty among the minima, the frozen tie rule.
        best = len(losses) - 1 - int(np.argmin(losses[::-1]))
        alpha = ALPHAS[best]
        prediction[test] = ridge_predict(x[train], target[train],
                                         row_weights(backgrounds[train], groups[train]),
                                         x[test], [alpha])[:, 0]
        records.append({'fold': fold, 'held_groups': list(held), 'alpha': alpha,
                        'inner_group_error': losses.tolist()})
    if not np.isfinite(prediction).all():
        raise ValueError('incomplete out-of-fold prediction')
    return prediction, records


def group_error(prediction: np.ndarray, target: np.ndarray, backgrounds: np.ndarray,
                groups: np.ndarray) -> float:
    """Group-equal, background-equal weighted squared error."""
    weights = row_weights(backgrounds, groups)
    return float((weights * (np.asarray(prediction) - np.asarray(target)) ** 2).sum() / weights.sum())


def design(blocks: dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    if not names:
        raise ValueError('a design needs at least one block')
    return np.column_stack([blocks[name] for name in names])


def qualify_controls(blocks: dict[str, np.ndarray], target: np.ndarray, backgrounds: np.ndarray,
                     groups: np.ndarray, *, seeds: Sequence[int] = SPLIT_SEEDS,
                     base: Sequence[str] = BASE_BLOCKS,
                     candidates: Sequence[str] = CANDIDATE_BLOCKS) -> dict[str, Any]:
    """Forward qualification of the candidate controls on this endpoint's own target.

    A candidate enters the standing set only if its paired reduction in
    group-equal held-out error over that standing set is positive at every split
    seed. Each candidate's own contribution is reported whether or not it is
    kept, and so is the qualified set's contribution over the base, because an
    increment measured over a control that predicts nothing is uninformative.
    """

    signatures = {seed: outer_signature(groups, seed) for seed in seeds}

    def error(names: Sequence[str]) -> dict[int, float]:
        matrix = design(blocks, names)
        return {seed: group_error(fold_predictions(matrix, target, backgrounds, groups, signatures[seed])[0],
                                  target, backgrounds, groups) for seed in seeds}

    standing = list(base)
    base_error = error(standing)
    offers = []
    for candidate in candidates:
        trial = error([*standing, candidate])
        reduction = {seed: base_error[seed] - trial[seed] for seed in seeds}
        kept = all(value > 0 for value in reduction.values())
        offers.append({'candidate': candidate, 'kept': kept,
                       'standing_before': list(standing),
                       'reduction_by_seed': {str(k): v for k, v in reduction.items()},
                       'own_contribution': float(np.mean(list(reduction.values())))})
        if kept:
            standing.append(candidate)
            base_error = trial
    null_error = {seed: group_error(np.zeros(len(target)), target, backgrounds, groups) for seed in seeds}
    return {'rule': ('offered in declared order; kept only if the paired group-equal held-out error '
                     'reduction over the standing set is positive at every split seed'),
            'base': list(base), 'candidates': list(candidates),
            'qualified': list(standing), 'offers': offers,
            'qualified_error_by_seed': {str(k): v for k, v in base_error.items()},
            'no_effect_error_by_seed': {str(k): v for k, v in null_error.items()},
            'qualified_contribution_over_no_effect': float(
                np.mean([null_error[s] - base_error[s] for s in seeds])),
            'profile_blocks_qualified': [b for b in standing if b.startswith('prof')]}


def background_metrics(predictions: dict[str, np.ndarray], target: np.ndarray,
                       backgrounds: np.ndarray, groups: np.ndarray, *, metric: str) -> list[dict]:
    """Per-background metric of each design; ranking and error never mixed."""
    if metric not in ('spearman', 'error'):
        raise ValueError('unknown metric')
    rows = []
    for background in sorted(set(backgrounds.tolist())):
        mask = backgrounds == background
        scores = {}
        for name, prediction in predictions.items():
            if metric == 'spearman':
                scores[name] = correlation(rankdata(prediction[mask]), target[mask])
            else:
                scores[name] = float(np.mean((prediction[mask] - target[mask]) ** 2))
        rows.append({'background': str(background), 'group': str(groups[mask][0]),
                     'rows': int(mask.sum()), 'metric': metric, 'scores': scores})
    return rows


def simultaneous_band(matrix: np.ndarray, *, draws: int = BOOTSTRAP_DRAWS,
                      seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Shared-draw simultaneous band over one declared contrast family.

    ``matrix`` is (independent groups x contrasts) of already group-aggregated
    contrast values. The band is the 95th percentile of the maximum absolute
    centered deviation across the family under one shared resample of the groups,
    which is the rule the frozen panels and ``extensions.progression`` apply; the
    pointwise quantiles are reported beside it, never instead of it.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError('invalid contrast matrix')
    n_groups = len(matrix)
    floor_record = bootstrap_unit_floor(n_groups)
    if draws < 100:
        raise ValueError('at least 100 bootstrap draws required')
    rng = np.random.default_rng(seed)
    counts = np.asarray([np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)
                         for _ in range(draws)], dtype=np.float64)
    point = matrix.mean(0)
    resampled = counts @ matrix / n_groups
    low, high = np.quantile(resampled, [0.025, 0.975], axis=0)
    critical = float(np.quantile(np.max(np.abs(resampled - point), axis=1), 0.95))
    return {'groups': n_groups, 'draws': draws, 'seed': seed,
            'independence_floor': floor_record,
            'shared_draw_sha256': array_digest(counts),
            'critical_max_absolute_deviation': critical,
            'point': point.tolist(),
            'pointwise': np.stack([low, high], axis=1).tolist(),
            'simultaneous': np.stack([point - critical, point + critical], axis=1).tolist(),
            'method': ('95th percentile maximum absolute centered deviation under one shared '
                       'group resample; conditional on fitted out-of-fold predictions')}


def aggregate(rows: Sequence[dict], key: str, groups: Sequence[str]) -> np.ndarray:
    """Background within group, equal groups; one column per contrast key order.

    A group with no contributing row is refused rather than averaged to a
    not-a-number: a bootstrap draw that silently carries a missing cell would
    report an interval over a support that is not the declared one.
    """

    by_group: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_group[row['group']].append(float(row[key]))
    missing = [group for group in groups if not by_group.get(group)]
    if missing:
        raise ValueError(f'{len(missing)} groups contribute no {key} cell, first {missing[0]!r}')
    return np.asarray([float(np.mean(by_group[group])) for group in groups])


def fit_cohort(rows: Sequence[PhenotypeRow], blocks: dict[str, np.ndarray],
               scores: dict[str, dict[str, float]], *, unit_groups: dict[str, str],
               qualified: Sequence[str], endpoint: str,
               seeds: Sequence[int] = SPLIT_SEEDS) -> dict[str, Any]:
    """One cohort, one endpoint, every arm: the increment over the qualified set.

    ``endpoint`` is ``ranking`` or ``quantitative``. The ranking endpoint fits the
    within-background standardized-rank target and reports Spearman; the
    quantitative endpoint fits the endpoint's own units and reports group-equal
    squared error. The two are never combined into one score.
    """

    if endpoint not in ('ranking', 'quantitative'):
        raise ValueError('unknown endpoint')
    backgrounds = np.asarray([row.background for row in rows])
    groups = np.asarray([unit_groups[row.unit] for row in rows])
    labels = np.asarray([row.oriented_label() for row in rows], dtype=np.float64)
    target = rerank(labels, backgrounds) if endpoint == 'ranking' else labels
    metric = 'spearman' if endpoint == 'ranking' else 'error'
    model_column = 'M_rank' if endpoint == 'ranking' else 'M_nats'
    per_background: list[dict] = []
    fit_records: dict[str, Any] = {}
    for arm in sorted(scores):
        model = model_block(rows, scores[arm], backgrounds)
        arm_blocks = {**blocks, 'M': model[model_column]}
        for seed in seeds:
            signature = outer_signature(groups, seed)
            baseline, baseline_records = fold_predictions(
                design(arm_blocks, qualified), target, backgrounds, groups, signature)
            augmented, augmented_records = fold_predictions(
                design(arm_blocks, [*qualified, 'M']), target, backgrounds, groups, signature)
            predictions = {'control': baseline, 'control_plus_model': augmented}
            for row in background_metrics(predictions, target, backgrounds, groups, metric=metric):
                scores_row = row['scores']
                if any(value is None for value in scores_row.values()):
                    raise ValueError(f"{arm}: undefined {metric} in background {row['background']}")
                sign = 1.0 if endpoint == 'ranking' else -1.0
                per_background.append({
                    'arm': arm, 'seed': seed, 'background': row['background'], 'group': row['group'],
                    'rows': row['rows'], 'control': scores_row['control'],
                    'control_plus_model': scores_row['control_plus_model'],
                    'increment': sign * (scores_row['control_plus_model'] - scores_row['control'])})
            fit_records.setdefault(arm, {})[str(seed)] = {
                'alphas_control': [r['alpha'] for r in baseline_records],
                'alphas_augmented': [r['alpha'] for r in augmented_records],
                'design_sha256': array_digest(design(arm_blocks, [*qualified, 'M'])),
                'target_sha256': array_digest(target)}
    arms = sorted(scores)
    groups_ordered = sorted(set(groups.tolist()))
    averaged: list[dict] = []
    for arm in arms:
        for background in sorted({r['background'] for r in per_background if r['arm'] == arm}):
            cells = [r for r in per_background if r['arm'] == arm and r['background'] == background]
            averaged.append({'arm': arm, 'background': background, 'group': cells[0]['group'],
                             'increment': float(np.mean([c['increment'] for c in cells]))})
    matrix = np.column_stack([aggregate([r for r in averaged if r['arm'] == arm],
                                        'increment', groups_ordered) for arm in arms])
    band = simultaneous_band(matrix)
    resolved = []
    for index, arm in enumerate(arms):
        low, high = band['simultaneous'][index]
        point_low, point_high = band['pointwise'][index]
        resolved.append({'arm': arm, 'point': band['point'][index],
                         'pointwise': [point_low, point_high], 'simultaneous': [low, high],
                         'resolved_positive': low > 0, 'resolved_negative': high < 0,
                         'pointwise_positive': point_low > 0})
    return {'endpoint': endpoint, 'metric': metric, 'model_column': model_column,
            'qualified_controls': list(qualified), 'arms': arms,
            'groups': groups_ordered, 'split_seeds': list(seeds),
            'aggregation': 'seed mean within background, background mean within group, equal groups',
            'increment_sign': ('augmented minus control Spearman' if endpoint == 'ranking'
                               else 'control minus augmented group-equal squared error'),
            'per_background': per_background, 'fits': fit_records,
            'inference': band, 'arm_results': resolved,
            'resolved_positive': sum(1 for r in resolved if r['resolved_positive']),
            'resolved_negative': sum(1 for r in resolved if r['resolved_negative'])}
