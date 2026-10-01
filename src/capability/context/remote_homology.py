"""Measured stability on metagenome-assembled domains, banded by retrieval identity.

This module owns the remote-homology gate's endpoint, its label-blind support
rules, its identity strata and its self-calibrating comparison. It exists because
every cohort the capability map has measured so far is saturated with
high-identity units: the stability gate's 101 one-per-group wild types carry no
alignment-detectable residual family structure at all, and the retrieval gate
reaches 1, 2 and 12 groups in its three lower identity bands against an
eight-unit percentile floor. Cho and Tsuboyama 2026's MGnify-derived libraries
populate the bands nothing else does.

The target is

    e = deltaG(variant) - deltaG(background),

the variant's own effect on folding free energy in kcal/mol on the staged table's
combined ``deltaG`` column, inferred jointly from the trypsin and chymotrypsin
channels. Larger deltaG is more stable, so a destabilising substitution gives a
negative e. That is the same two-estimate difference and the same sign convention
the admitted folding-and-stability gate measures on Tsuboyama 2023, so the two
are commensurable quantities on two sources.

**What this endpoint does not license.** It shares the cDNA-display proteolysis
assay technology and an author with Tsuboyama 2023, so it is not external
confirmation of anything and may not be read as independent replication; the
frozen endpoint qualification records that boundary and this module does not
widen it. An identity band is a property of a search against one UniRef50
snapshot: no detected alignment is not absence from any model's pretraining
corpus.

**The design constraint that decides whether this gate is interpretable.** The
two protease channels of this table disagree far more than they do on the
admitted MegaScale endpoint. Over 1,413,334 admitted rows the trypsin-minus-
chymotrypsin deltaG has mean -0.478 and root-mean-square 0.824 kcal/mol against a
combined-deltaG spread of 1.251, with Pearson r = 0.857: a level-scale floor two
thirds of the endpoint's own variation, and a systematic offset rather than
symmetric noise. The MegaScale endpoint's per-channel discordance is 0.2337
[0.2209, 0.2458] kcal/mol against a shared component of 0.9013. Two consequences
are declared here rather than discovered.

* The mean offset is common to a variant and to its own background, so it
  cancels in the difference this endpoint takes. The floor that bounds this gate
  is therefore the *effect-scale* discordance, measured on this gate's own
  admitted support by :func:`channel_decomposition`, and never the level-scale
  number. Both are reported, and which one a statement rests on is named.
* Whatever that floor turns out to be, a remote null is uninterpretable on its
  own. The comparison is **self-calibrating**: the same fit, on the same folds,
  is read out separately on the groups whose every member retrieves a close
  relative and on the groups whose every member does not, and the two readouts
  together select one of three outcomes -- :data:`OUTCOME_SURVIVES`,
  :data:`OUTCOME_HOMOLOGY_DEPENDENT` or :data:`OUTCOME_UNRESOLVED`. A remote
  result is never reported without naming which.

Weighting, folds, the ridge recipe, the projection, the nonlinear response and
the group bootstrap are imported from the external-confirmation and admitted
Readout recipes rather than restated, and every feature block is imported from
the stability gate. The independent unit is the family group; inside a group each
background carries equal weight, inside a background each mutated site, and
inside a site each variant.
"""
from __future__ import annotations

import csv
import hashlib
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from ..mutation.external_confirmation import (
    ALIGNMENT_GAP_EXTEND, ALIGNMENT_GAP_OPEN, ARM_CANDIDATE_BLOCKS, BASE_BLOCKS,
    CANDIDATE_BLOCKS, G_FEATURE_ORDER, GROUPING_COVERAGE, GROUPING_IDENTITY,
    MODEL_BLOCKS, build_panel, declaration_digest, fold_predictions, group_errors,
    interval, kish_units, nested_weights, nuisance_response, paired_increment,
    qualify, raw_spearman, row_identity, spearman_increment)
from ..core.family_grouping import Union, batch_align, encode
from .homology import STRATUM_NAMES, assign_stratum
from ..interactions.pairwise_epistasis import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, FEATURE_BLOCKS, PROJECTION_DIM, PROJECTION_SEED,
    SPLIT_SEEDS)

# --------------------------------------------------------------------------- #
# The endpoint and the pinned source.
# --------------------------------------------------------------------------- #

#: What the target is, in one line, so an artefact cannot be read against an
#: abundance or a DMS-fitness endpoint by mistake.
ENDPOINT = (
    'e = deltaG(variant) - deltaG(background) on the staged combined `deltaG` scale in '
    'kcal/mol; larger deltaG is more stable, so a destabilising single substitution is '
    'negative')

#: The measured quantity, named so that this endpoint is never read as
#: independent replication of the development stability axis.
MEASURED_QUANTITY = ('folding free energy of a small protein domain inferred from '
                     'cDNA-display proteolysis against two proteases, the same assay '
                     'technology as Tsuboyama 2023')

SOURCE_FILE = '230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv'
SOURCE_SHA256 = '1607b8771108a040c23436088ae695385df11f5179c32517758ad426942e1e02'
SOURCE_DOI = '10.5281/zenodo.19411306'

#: Every column this module reads, in the order the positional reader takes them.
#: The table is 2.23 GB over 38 columns and a dictionary reader dominated the
#: pass, which is why the reader is positional.
SOURCE_COLUMNS = (
    'name', 'lib', 'aa_seq', 'mgnify', 'dm_design',
    'deltaG', 'deltaG_95CI', 'deltaG_t', 'deltaG_t_95CI', 'deltaG_c', 'deltaG_c_95CI',
)

#: The two protease channels, as two channels of one assay over one library and
#: one stability inference -- not two independent measurements of a truth.
CHANNELS = ('trypsin', 'chymotrypsin')

# --------------------------------------------------------------------------- #
# Declared admission and support rules.
# --------------------------------------------------------------------------- #

#: The tightened per-channel admission rule, which is the one the staged
#: stability cohort already applies rather than a new one: every one of the three
#: reported 95% interval widths -- trypsin, chymotrypsin and combined -- no wider
#: than this, in kcal/mol. Both the variant row and its background's wild-type
#: row must pass, because the endpoint is a difference of the two.
QC_WIDTH_KCAL_MOL = 0.5
QC_SOURCE = ('the table\'s own deltaG_95CI, deltaG_t_95CI and deltaG_c_95CI columns, '
             'applied to the variant row and to its background\'s wild-type row')

#: A single amino-acid substitution token. A name in these libraries is
#: ``<background>_<token>``; ``ins`` and ``del`` tokens change the construct's
#: length and are a separate class, and the libraries also carry scramble
#: controls, so the token vocabulary is not assumed -- it is matched.
SUBSTITUTION_TOKEN = re.compile(r'^([A-Z])(\d+)([A-Z])$')

#: The declared background draw. It is the seeded sample of the MGnify-flagged
#: backgrounds that carry a variant series and a wild-type row, and it is the
#: same draw -- same seed, same size, same sorted name universe -- whose wild
#: types the frozen endpoint qualification already searched and banded, so the
#: bands reported here are the canonical ones rather than a second search.
BACKGROUND_DRAW = 320
BACKGROUND_DRAW_SEED = 20260924

#: A retained background must carry at least this many admitted single
#: substitutions, so that it contributes a site-resolved panel rather than a
#: single row. The rule reads library coverage and interval widths, never a
#: measured effect. No variant cap is declared: the largest admitted count on
#: this support is 26, so a cap would never bind and is not invented.
MIN_VARIANTS = 8

#: Identity bands, imported from the frozen declaration rather than restated, so
#: a count here is commensurable with the retrieval gate's own tables.
BANDS = STRATUM_NAMES
REMOTE_BANDS = (STRATUM_NAMES[0], STRATUM_NAMES[1])
CLOSE_BANDS = (STRATUM_NAMES[2], STRATUM_NAMES[3])

#: The declared percentile-interval floor. A stratum below it carries no verdict
#: in either direction, which is why every count reported for a stratum is a
#: group count and never a variant count.
STRATUM_UNIT_FLOOR = 8

#: The three outcomes the self-calibrating readout selects between. Reporting a
#: remote result without naming one of these is what this gate exists to prevent.
OUTCOME_SURVIVES = 'resolves on close and on remote groups: survives remote homology'
OUTCOME_HOMOLOGY_DEPENDENT = 'resolves on close groups only: homology dependence'
OUTCOME_UNRESOLVED = ('resolves on neither stratum: the endpoint floor is too high on this '
                      'support and the remote case is unresolved, not null')

__all__ = [
    'ARM_CANDIDATE_BLOCKS', 'BACKGROUND_DRAW', 'BACKGROUND_DRAW_SEED', 'BANDS',
    'BASE_BLOCKS', 'BOOTSTRAP_DRAWS', 'BOOTSTRAP_SEED', 'CANDIDATE_BLOCKS',
    'CHANNELS', 'CLOSE_BANDS', 'ENDPOINT', 'FEATURE_BLOCKS', 'G_FEATURE_ORDER',
    'MEASURED_QUANTITY', 'MIN_VARIANTS', 'MODEL_BLOCKS', 'OUTCOME_HOMOLOGY_DEPENDENT',
    'OUTCOME_SURVIVES', 'OUTCOME_UNRESOLVED', 'PROJECTION_DIM', 'PROJECTION_SEED',
    'QC_SOURCE', 'QC_WIDTH_KCAL_MOL', 'REMOTE_BANDS', 'SOURCE_COLUMNS', 'SOURCE_DOI',
    'SOURCE_FILE', 'SOURCE_SHA256', 'SPLIT_SEEDS', 'STRATUM_UNIT_FLOOR',
    'CHANNEL_NAMES', 'CHANNEL_UNIT', 'SUBSTITUTION_TOKEN', 'admitted',
    'background_map', 'best_identity', 'build_panel',
    'channel_decomposition', 'channel_interval', 'channel_moments',
    'declaration_digest', 'declared_draw',
    'endpoint_digest', 'family_groups', 'fold_predictions', 'group_errors',
    'group_strata', 'interval', 'kish_units', 'nested_weights', 'nuisance_response',
    'outcome', 'outcome_record', 'paired_increment', 'qualify', 'query_bands',
    'raw_spearman',
    'read_source', 'row_identity', 'spearman_increment', 'stratum_keep',
    'substitution_at', 'unit_floor_cleared',
]


# --------------------------------------------------------------------------- #
# Reading the pinned source.
# --------------------------------------------------------------------------- #

def read_source(path, *, columns: tuple[str, ...] = SOURCE_COLUMNS) -> list[tuple[str, ...]]:
    """Read the pinned table's declared columns with a positional reader.

    A row whose field count differs from the header is dropped and is not
    repaired: a truncated line in a 2.23 GB table carries no resolvable
    measurement, and silently padding it would put an unmeasured row on the
    endpoint.
    """

    csv.field_size_limit(1 << 24)
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name: position for position, name in enumerate(header)}
        missing = [name for name in columns if name not in index]
        if missing:
            raise ValueError(f'{path} is missing declared columns: {missing}')
        order = [index[name] for name in columns]
        width = len(header)
        return [tuple(row[position] for position in order)
                for row in reader if len(row) == width]


def background_map(names: set[str]) -> dict[str, str]:
    """Each name's background: its longest proper underscore-prefix that is itself a name.

    The token vocabulary of these libraries is not fixed -- the staged table
    carries 31,412 distinct tokens, scramble controls beside substitutions,
    insertions and deletions -- so a declared grammar misparses most of the file.
    This rule reads only the name set the file actually carries. It is the rule
    the frozen endpoint qualification's census used, and the qualification entry
    point checks that it reproduces that census's own background count rather
    than assuming the two agree.
    """

    out = {}
    for name in names:
        parts = name.split('_')
        root = name
        for cut in range(len(parts) - 1, 0, -1):
            candidate = '_'.join(parts[:cut])
            if candidate in names:
                root = candidate
                break
        out[name] = root
    return out


def admitted(widths: tuple[str, ...]) -> bool:
    """Whether one row's three reported 95% interval widths all clear the rule.

    A width that does not parse as a number is not admitted. It is not treated as
    a wide interval and not treated as a missing one: either reading would put a
    row with no resolvable uncertainty on the endpoint.
    """

    try:
        values = [float(value) for value in widths]
    except (TypeError, ValueError):
        return False
    return all(np.isfinite(value) and 0.0 <= value <= QC_WIDTH_KCAL_MOL for value in values)


def substitution_at(background: str, variant: str, token: str) -> int:
    """The 0-based position a substitution token names, verified against both sequences.

    The token's own wild-type residue, mutant residue and position are all checked
    against the two sequences, and the two sequences must differ at that one
    position and nowhere else. A row whose name disagrees with its own sequence is
    refused rather than scored, and an insertion or deletion construct fails the
    length check rather than entering a substitution support length-matched.
    """

    match = SUBSTITUTION_TOKEN.match(token)
    if match is None:
        raise ValueError(f'{token!r} is not a single-substitution token')
    wild, position, mutant = match.group(1), int(match.group(2)), match.group(3)
    if len(background) != len(variant):
        raise ValueError(f'{token}: variant length {len(variant)} differs from the background')
    index = position - 1
    if not 0 <= index < len(background):
        raise ValueError(f'{token}: position {position} outside the background')
    if background[index] != wild or variant[index] != mutant:
        raise ValueError(f'{token}: disagrees with the sequences at position {position}')
    differing = [i for i in range(len(background)) if background[i] != variant[i]]
    if differing != [index]:
        raise ValueError(f'{token}: {len(differing)} substituted positions, not one')
    return index


def declared_draw(names, *, size: int = BACKGROUND_DRAW,
                  seed: int = BACKGROUND_DRAW_SEED) -> list[str]:
    """The seeded background draw, over the sorted name universe.

    The draw reads names only -- no measurement, no interval width and no model
    score -- and it is the draw the frozen endpoint qualification already banded,
    so reproducing it here is what makes those bands this cohort's own.
    """

    universe = sorted(names)
    if size > len(universe):
        raise ValueError(f'draw of {size} exceeds the {len(universe)} available backgrounds')
    return sorted(random.Random(seed).sample(universe, size))


def endpoint_digest(records) -> str:
    """Content digest over the endpoint's own values, in a stable order."""

    payload = '\n'.join(
        f"{row['background']}\t{row['position']}\t{row['mutant']}\t{row['target']:.10g}"
        for row in sorted(records, key=lambda r: (r['background'], r['position'], r['mutant'])))
    return hashlib.sha256(payload.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Between-channel agreement, on the level scale and on the effect scale.
# --------------------------------------------------------------------------- #

def channel_moments(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Sufficient statistics of one channel pair, so a resample is a mean of vectors.

    The seven moments are the two means, the two second moments, the cross moment,
    the mean squared difference and the mean absolute difference, in that order.
    """

    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape or first.size == 0:
        raise ValueError('channel moments need two aligned non-empty channels')
    difference = first - second
    return np.array([first.sum(), second.sum(), (first * first).sum(), (second * second).sum(),
                     (first * second).sum(), (difference * difference).sum(),
                     np.abs(difference).sum()]) / first.size


#: Default channel names and unit of the decomposition below: this gate's two
#: proteases, in kcal/mol. They are arguments and not literals inside the keys
#: because the same decomposition is read on other channel pairs of other
#: quantities -- the Domainome endpoint's two biological replicates are in log2
#: enrichment -- and a quantity published under another quantity's unit is a
#: defect, not a naming preference.
CHANNEL_NAMES = ('trypsin', 'chymotrypsin')
CHANNEL_UNIT = 'kcal_mol'


def channel_decomposition(moments: np.ndarray, *,
                          channels: tuple[str, str] = CHANNEL_NAMES,
                          unit: str = CHANNEL_UNIT) -> dict:
    """Shared component and per-channel discordance, from a moment vector.

    The definitions are the admitted stability gate's, used unchanged so the two
    endpoints' floors are the same quantity: the shared-component standard
    deviation is the square root of the between-channel covariance and is an
    **upper** bound on reproducible signal, because assay error common to both
    channels contributes to it; the per-channel discordance standard deviation is
    the standard deviation of the channel difference over the square root of two
    and is a **lower** bound on per-channel measurement noise, because common
    error cancels. No oracle ceiling is built by adding independent variances.

    The channel difference's mean is reported separately from its spread because
    on this source it is large and systematic. On a level scale that offset is
    part of the disagreement; in a within-background difference it cancels, and
    the discordance standard deviation is defined about the mean either way.
    """

    mean_first, mean_second, second_first, second_second, cross, mean_square, mean_abs = moments
    variance_first = max(float(second_first - mean_first * mean_first), 0.0)
    variance_second = max(float(second_second - mean_second * mean_second), 0.0)
    covariance = float(cross - mean_first * mean_second)
    offset = float(mean_first - mean_second)
    difference_variance = max(float(mean_square - offset * offset), 0.0)
    shared = float(np.sqrt(max(covariance, 0.0)))
    discordance = float(np.sqrt(difference_variance / 2.0))
    denominator = variance_first * variance_second
    first, second = channels
    return {
        f'mean_{first}_{unit}': float(mean_first),
        f'mean_{second}_{unit}': float(mean_second),
        f'sd_{first}_{unit}': float(np.sqrt(variance_first)),
        f'sd_{second}_{unit}': float(np.sqrt(variance_second)),
        'pearson_r': float(covariance / np.sqrt(denominator)) if denominator > 0 else None,
        f'shared_component_sd_{unit}': shared,
        f'channel_offset_mean_{unit}': offset,
        f'channel_difference_rms_{unit}': float(np.sqrt(mean_square)),
        f'mean_absolute_channel_difference_{unit}': float(mean_abs),
        f'per_channel_discordance_sd_{unit}': discordance,
        'shared_to_discordance_ratio': float(shared / discordance) if discordance > 0 else None,
    }


def channel_interval(vectors, *, channels: tuple[str, str] = CHANNEL_NAMES,
                     unit: str = CHANNEL_UNIT, draws: int = BOOTSTRAP_DRAWS,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """Group-bootstrap percentile intervals on every decomposed quantity.

    Each element of ``vectors`` is one resampling unit's moment vector, so a draw
    resamples whole family groups and keeps intact the reuse that defines this
    endpoint: every variant of a background is differenced against that
    background's single wild-type measurement.

    The shared unit floor governs it, applied to the family-group count through
    :func:`~.statistics.bootstrap_unit_floor` rather than through a literal of
    this module's own. Its refusal is **degenerate** rather than raised, for the
    same reason the higher-order gate's channel decomposition refuses that way:
    the unit count is a measured property of a stratum's support, the point
    estimates of a two-channel decomposition are still the finding when the
    stratum is too thin to bound, and the interval fields are then absent rather
    than null-valued, so a reader cannot quote a bound that was never computed.
    """

    from ..core.statistics import bootstrap_unit_floor

    stack = np.asarray(vectors, dtype=float)
    if stack.ndim != 2 or stack.shape[0] < 1:
        raise ValueError('channel intervals need at least one resampling unit')
    point = channel_decomposition(stack.mean(axis=0), channels=channels, unit=unit)
    floor = bootstrap_unit_floor(int(stack.shape[0]))
    report = {'units': int(stack.shape[0]), 'draws': int(draws), 'seed': int(seed),
              'channels': list(channels), 'unit': unit,
              'resampling_unit': 'family group of the wild-type background', **floor}
    if floor['degenerate']:
        for key, value in point.items():
            report[key] = {'point': value}
        return report
    generator = np.random.default_rng(seed)
    index = generator.integers(0, stack.shape[0], size=(draws, stack.shape[0]))
    samples = [channel_decomposition(stack[row].mean(axis=0), channels=channels, unit=unit)
               for row in index]
    for key, value in point.items():
        column = np.asarray([s[key] for s in samples if s[key] is not None], dtype=float)
        column = column[np.isfinite(column)]
        report[key] = {
            'point': value,
            'ci95': [float(np.percentile(column, 2.5)),
                     float(np.percentile(column, 97.5))] if column.size else None}
    return report


# --------------------------------------------------------------------------- #
# Family grouping and the identity strata.
# --------------------------------------------------------------------------- #

def family_groups(wildtypes: dict[str, str]) -> tuple[dict[str, str], dict]:
    """Single-linkage family groups under the frozen grouping contract's edge rule.

    Exact Smith-Waterman with BLOSUM62 and NCBI default affine gaps, an edge when
    percent identity is at least 30.0 and coverage is at least 80.0 for both
    sequences. Alignment is the only union source available here: these names
    carry no UniProt accession and no Pfam label, so unlike the Domainome support
    there is no accession or family field to union on, and unlike the MegaScale
    catalogue there is no design stratum or source cluster.
    """

    order = sorted(wildtypes)
    union = Union(order)
    sequences = [wildtypes[name] for name in order]
    by_sequence: dict[str, list[str]] = {}
    for name, sequence in zip(order, sequences):
        by_sequence.setdefault(sequence, []).append(name)
    for members in by_sequence.values():
        for other in members[1:]:
            union.join(members[0], other, 'exact_sequence')
    pairs = np.column_stack(np.triu_indices(len(order), k=1))
    statistics = batch_align(*encode(sequences), pairs, gap_open=ALIGNMENT_GAP_OPEN,
                             gap_extend=ALIGNMENT_GAP_EXTEND)
    edge = statistics.edges(identity_floor=GROUPING_IDENTITY, coverage_floor=GROUPING_COVERAGE)
    for row in np.flatnonzero(edge):
        union.join(order[int(pairs[row, 0])], order[int(pairs[row, 1])], 'alignment')
    components = union.components()
    labels = {}
    for number, root in enumerate(sorted(components), start=1):
        for member in components[root]:
            labels[member] = f'mgy-{number:03d}'
    sizes = Counter(labels.values())
    report = {
        'backgrounds': len(order),
        'groups': len(components),
        'group_size_max': int(max(sizes.values())),
        'group_size_median': float(np.median(sorted(sizes.values()))),
        'singleton_groups': int(sum(1 for value in sizes.values() if value == 1)),
        'alignment_edges': int(edge.sum()),
        'exact_sequence_edges': int(sum(len(m) - 1 for m in by_sequence.values())),
        'union_sources': ['exact_sequence', 'alignment'],
        'edge_rule': (f'percent identity >= {GROUPING_IDENTITY} and coverage >= '
                      f'{GROUPING_COVERAGE} for both sequences, Smith-Waterman with '
                      f'BLOSUM62 and gap {ALIGNMENT_GAP_OPEN:.0f} + '
                      f'{ALIGNMENT_GAP_EXTEND:.0f}k'),
    }
    return labels, report


def best_identity(hits, names) -> dict[str, float | None]:
    """Best identity over the query, per name, from a DIAMOND tabular hit table.

    Identity is ``100 * nident / qlen`` -- the percent of the query identically
    matched -- which is the definition the frozen band declaration uses. A name
    with no reported hit stays ``None``, and the distinction between that and a
    distant hit is kept rather than collapsed into a band.
    """

    wanted = set(names)
    best: dict[str, float | None] = {name: None for name in wanted}
    for line in Path(hits).read_text(encoding='utf-8').splitlines():
        fields = line.split('\t')
        if len(fields) < 4:
            continue
        query, identical, length = fields[0], int(fields[2]), int(fields[3])
        if query not in wanted or length == 0:
            continue
        identity = 100.0 * identical / length
        current = best[query]
        if current is None or identity > current:
            best[query] = identity
    return best


def query_bands(best: dict[str, float | None]) -> dict[str, str]:
    """Band each query, treating a query with no reported hit as identity 0.0."""

    return {name: assign_stratum(0.0 if value is None else value)
            for name, value in best.items()}


def group_strata(labels: dict[str, str], bands: dict[str, str]) -> tuple[dict, dict]:
    """Close, remote and mixed family groups, by the bands of all of their members.

    A group is close only if **every** member retrieves a relative at 70% identity
    or above, and remote only if **every** member falls below it. A group whose
    members straddle the boundary is reported as mixed and enters neither
    stratum: it is the unit a held-group readout uses, so a group that is partly
    close cannot certify either side.
    """

    missing = sorted(set(labels) - set(bands))
    if missing:
        raise ValueError(f'{len(missing)} retained backgrounds carry no identity band')
    members: dict[str, list[str]] = {}
    for name, label in labels.items():
        members.setdefault(label, []).append(name)
    assignment = {}
    for label, names in members.items():
        seen = {bands[name] for name in names}
        if seen <= set(REMOTE_BANDS):
            assignment[label] = 'remote'
        elif seen <= set(CLOSE_BANDS):
            assignment[label] = 'close'
        else:
            assignment[label] = 'mixed'
    counts = Counter(assignment.values())
    report = {
        'rule': ('a group is close when every member bands at 70% identity or above, '
                 'remote when every member bands below it, and mixed otherwise; a mixed '
                 'group enters neither stratum'),
        'groups': len(assignment),
        'close_groups': int(counts['close']),
        'remote_groups': int(counts['remote']),
        'mixed_groups': int(counts['mixed']),
        'unit_floor': STRATUM_UNIT_FLOOR,
        'close_clears_the_unit_floor': bool(counts['close'] >= STRATUM_UNIT_FLOOR),
        'remote_clears_the_unit_floor': bool(counts['remote'] >= STRATUM_UNIT_FLOOR),
        'backgrounds_per_band': dict(Counter(bands[name] for name in labels)),
        'backgrounds_with_no_reported_hit_band': STRATUM_NAMES[0],
    }
    return assignment, report


def stratum_keep(group: np.ndarray, assignment: dict[str, str], stratum: str) -> np.ndarray:
    """Row mask for one identity stratum, for a readout restriction rather than a refit.

    The fit is unchanged: the same folds, the same training groups and the same
    held-out predictions. Only the rows the paired per-group error is read over
    change, and :func:`group_errors` renormalises the nesting inside them.
    """

    if stratum not in ('close', 'remote', 'mixed'):
        raise ValueError(f'{stratum!r} is not a declared identity stratum')
    return np.asarray([assignment.get(label) == stratum for label in np.asarray(group)])


def unit_floor_cleared(count: int) -> bool:
    """Whether a stratum reaches the declared percentile-interval unit floor."""

    return int(count) >= STRATUM_UNIT_FLOOR


def outcome(close_resolved: bool, remote_resolved: bool) -> str:
    """Which of the three declared outcomes one arm's stratified readout selects.

    The close stratum is a **positive control**, so it gates the reading rather
    than merely accompanying it: with a floor this high, a stratum on which the
    pipeline resolves nothing at all cannot tell an absent quantity from an
    unmeasurable one. An arm that resolves on the remote groups while its own
    positive control did not fire is therefore **unresolved**, not survival --
    the remote increment is reported beside the verdict as an observation this
    design does not license, and :func:`outcome_record` carries the flag that
    says so.
    """

    if close_resolved and remote_resolved:
        return OUTCOME_SURVIVES
    if close_resolved:
        return OUTCOME_HOMOLOGY_DEPENDENT
    return OUTCOME_UNRESOLVED


def outcome_record(close_resolved: bool, remote_resolved: bool) -> dict:
    """One arm's stratified verdict, with what licensed it and what did not."""

    return {
        'close_resolved': bool(close_resolved),
        'remote_resolved': bool(remote_resolved),
        'positive_control_fired': bool(close_resolved),
        'outcome': outcome(close_resolved, remote_resolved),
        'remote_resolved_without_its_positive_control': bool(
            remote_resolved and not close_resolved),
    }
