"""External confirmation of the first-order likelihood finding on Domainome 1.0.

The capability map's last step confirms the strongest findings of the map on data
used in neither development panel. This module owns that step's endpoint, its
label-blind support rules, its declared controls and its nested held-group
comparison. It measures a **different measured quantity** from both development
sources and is not a replication of either estimand:

* the development mutation-ranking axis is ProteinGym's 217 heterogeneous deep
  mutational scans, 64 of whose readout anchor assays are Tsuboyama 2023;
* the development stability axis is cDNA-display proteolysis on the same
  Tsuboyama 2023 source;
* this endpoint is cellular abundance by protein-fragment complementation
  (aPCA) in *Saccharomyces cerevisiae*, from Beltran and Lehner's Domainome 1.0.

Agreement across two different measured quantities is stronger evidence than
agreement within one, and it is not a repeat measurement of either, so a
disagreement here is not by itself a failure to replicate.

The target is

    f = normalized_fitness of one single amino-acid substitution,

dimensionless, normalised by the source per domain so that the wild type reads
0 and the median nonsense variant reads about -1. Larger is greater abundance,
so a destabilising or abundance-lowering substitution gives a negative f. There
is no second measured state: the wild-type reference is identically zero by the
source's construction, which is what makes this endpoint a one-estimate level
rather than the two-estimate difference the stability gate targets.

Three support rules are declared here rather than discovered.

* **Development overlap is excluded by sequence, not by identifier.** No
  ProteinGym entry names Beltran or the domainome, yet 28 of the 522 domains
  match a development target exactly or by containment. Those 28 are excluded
  from the declared support by name, and every retained wild type is
  additionally screened by local alignment against all 217 ProteinGym targets
  and all 478 Tsuboyama backgrounds under a declared identity and coverage rule.
* **One accession, one Pfam family and one alignment component are one group.**
  Two domains cut from one protein are not independent units, so the family
  grouping unions shared UniProt accession, shared Pfam accession, exact
  sequence identity and the frozen grouping contract's own alignment edge rule.
* **At most a declared cap of variants per domain**, drawn by a stable hash of
  the draw seed, the domain identifier and the mutant sequence. The rule reads
  no measurement, no standard error and no model score.

Weighting, folds, the ridge recipe, the projection and the group bootstrap are
imported from the admitted Readout and pairwise recipes rather than restated,
and every feature block is imported from the stability gate rather than
reimplemented, so the control side of this confirmation is the same code the
development measurement fitted. The independent unit is the family group; inside
a group each domain carries equal weight, inside a domain each mutated site, and
inside a site each variant. That three-level nesting is the one the stability
gate's own endpoint qualification declares for a resampling unit holding several
backgrounds, and it is the one substitution this module makes to the
development fit's two-level weighting, which assumed one background per group.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from ..core.family_grouping import Union, batch_align, encode
from ..interactions.pairwise_epistasis import (
    ALPHAS, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, FEATURE_BLOCKS, INNER_SPLITS,
    OUTER_SPLITS, PROJECTION_DIM, PROJECTION_SEED, SPLIT_SEEDS, _select_alpha,
    group_spearman, interval, projection_matrices)
from ..readouts.readout_analysis import family_folds, ridge_predict
from ..stability.stability_gate import (
    TOKENISATION_FEATURE_ORDER, chemistry_block, chemistry_width,
    composition_block, geometry_block, identity_block, profile_block,
    profile_bounded_block, tokenisation_block)

# --------------------------------------------------------------------------- #
# The endpoint and the pinned source.
# --------------------------------------------------------------------------- #

#: What the target is, in one line, so an artefact cannot be read against a
#: stability or a DMS-fitness endpoint by mistake.
ENDPOINT = (
    'f = normalized_fitness of one single amino-acid substitution on the Domainome 1.0 '
    'abundance-PCA scale, dimensionless and normalised per domain so the wild type reads 0; '
    'larger is greater cellular abundance, so a destabilising substitution is negative')

#: The measured quantity, named so that it is never reported as a replication of
#: proteolysis-derived stability or of DMS fitness.
MEASURED_QUANTITY = ('cellular abundance of a protein domain fused into a '
                     'protein-fragment complementation assay in Saccharomyces cerevisiae')

#: The pinned source archive and the member inside it, with the digest the
#: dataset registry measured from bytes on disk.
SOURCE_ARCHIVE = 'Supplementary_Table_2_fitness_scores_normalized_domainranks.txt.zip'
SOURCE_MEMBER = 'Extended_data_Table_2_fitness_scores_normalized_domainranks.txt'
SOURCE_ARCHIVE_SHA256 = '071ce5fae6cd25d67cf15b3db73a8e6f476ec623a8152f36ea19712f0150a79d'
SOURCE_DOI = '10.5281/zenodo.14356805'

#: Every column this module reads.
SOURCE_COLUMNS = (
    'domain_ID', 'uniprot_ID', 'aa_seq', 'wt_aa', 'position', 'mut_aa', 'STOP',
    'input_count_rep1', 'input_count_rep2', 'input_count_rep3',
    'output_count_rep1', 'output_count_rep2', 'output_count_rep3',
    'mean_input_count', 'fitness', 'fitness_sigma',
    'normalized_fitness', 'normalized_fitness_sigma', 'quality_rank',
)

#: The source's ``position`` is the residue index in the parent protein, and the
#: last field of ``domain_ID`` is the domain's first position in that protein, so
#: ``aa_seq`` index = ``position`` - :func:`domain_start`. The derivation is
#: verified against every accepted row's own sequence rather than assumed.
POSITION_CONVENTION = ('source `position` counts residues of the parent protein; the '
                       '0-based index into the domain sequence is position minus the '
                       'domain start carried in the last field of `domain_ID`')

#: Replicate columns, as three channels of the same assay over the same library.
REPLICATES = (1, 2, 3)

# --------------------------------------------------------------------------- #
# Declared support rules.
# --------------------------------------------------------------------------- #

#: At most this many substitution variants per domain, drawn label-blind.
VARIANT_CAP = 256

#: Stable-hash draw seed. It keys the mutant sequence and the domain name only.
DRAW_SEED = 20260924

#: A retained domain must carry at least this many accepted variants, so that a
#: domain contributes a site-resolved panel rather than a handful of rows. The
#: rule reads library coverage, never a measured effect.
MIN_VARIANTS = 64

#: Declared exclusion screen against the two development sources. A retained
#: wild type must not reach this identity over this much of its own length
#: against any ProteinGym target or Tsuboyama background. 50% identity with 80%
#: coverage is the readout cohort's own wild-type-family rule, so the screen
#: refuses a domain that would fall in a development target's family under the
#: development panel's own definition.
DEVELOPMENT_IDENTITY = 50.0
DEVELOPMENT_COVERAGE = 80.0

#: Reported as a declared stratum beside the primary screen, at the frozen
#: grouping contract's own edge rule.
DEVELOPMENT_STRATUM_IDENTITY = 30.0

#: Family grouping: the frozen contract's alignment edge rule, applied between
#: retained wild types.
GROUPING_IDENTITY = 30.0
GROUPING_COVERAGE = 80.0
ALIGNMENT_GAP_OPEN = 11.0
ALIGNMENT_GAP_EXTEND = 1.0

#: Union sources, in the order they are applied. ``accession`` and ``pfam`` are
#: the two the frozen contract cannot express for this dataset and are declared
#: here: the source's ``domain_ID`` carries both, so every domain is labelled and
#: none is an incomplete annotation.
UNION_SOURCES = ('exact_sequence', 'accession', 'pfam', 'alignment')

# --------------------------------------------------------------------------- #
# Declared control blocks, in the order they are offered to the qualification.
# --------------------------------------------------------------------------- #

#: The base every candidate is qualified over: directed substitution identity
#: and the mutated position's geometry. No chemistry, no composition, no
#: evolutionary statistic. Identical to the stability gate's base.
BASE_BLOCKS = ('ident', 'geom')

#: Candidate controls in declared order. Each is a competing explanation for an
#: abundance change, never a mechanism. ``prof`` and ``prof2`` are the stability
#: gate's raw and bounded restatements of the same mutation-local profile
#: information; both are offered and both verdicts are reported.
CANDIDATE_BLOCKS = ('comp', 'chem', 'prof', 'prof2', 'G')

#: The tokenisation interface descriptors, offered to each arm's own matched
#: baseline under the same rule, because a representation difference can be
#: nonzero purely through segmentation.
ARM_CANDIDATE_BLOCKS = ('T',)

#: The nonlinear global response. The stability gate fits its first stage on
#: *absolute* state levels and applies the isotonic response to both states of a
#: pair, because its endpoint is a difference of two measured levels. This
#: endpoint has no second measured state -- the wild-type reference is
#: identically zero per domain -- so the response is applied to the predicted
#: effect itself. That is the declared substitution, and it is the only one.
G_FEATURE_ORDER = ('calibrated_response', 'first_stage_prediction')

#: Model additions. ``M`` is the native likelihood difference in nats, one
#: scalar column. ``R`` is the projected representation difference, four pooled
#: block summaries at 256 projected coordinates each.
MODEL_BLOCKS = ('M', 'R')

# --------------------------------------------------------------------------- #
# The numeric environment, declared once for every entry point of this gate and
# of the remote-homology gate that imports this machinery.
# --------------------------------------------------------------------------- #

#: Pinned BLAS thread count. A float32 matrix product's reduction order depends
#: on how many threads the library splits it over, so an unpinned thread count
#: makes a projected feature block and a ridge solution reproducible only on the
#: machine that produced them. The admitted projection ran at 4 threads and this
#: is that value, carried so a refit reproduces it rather than re-deriving it.
BLAS_THREADS = 4

#: Environment variables that must carry :data:`BLAS_THREADS` before NumPy is
#: imported. Setting them afterwards does not move an already-initialised
#: library's pool, which is why this is checked and refused rather than fixed.
BLAS_THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')


def require_blas_threads(expected: int = BLAS_THREADS) -> dict:
    """Refuse to run unless the BLAS thread count is pinned to ``expected``.

    At least one of the declared variables must be set to ``expected``, and none
    may be set to anything else. The check fails fast instead of setting the
    variables itself: by the time this module is imported NumPy has already
    initialised its thread pool, so a late assignment would record a pinned count
    while computing at an unpinned one.
    """

    import os

    observed = {name: os.environ.get(name) for name in BLAS_THREAD_VARIABLES}
    wrong = {name: value for name, value in observed.items()
             if value is not None and value.strip() != str(expected)}
    if wrong:
        raise SystemExit(f'BLAS thread count must be pinned to {expected}; found {wrong}')
    if not any(value is not None for value in observed.values()):
        raise SystemExit(
            f'BLAS thread count is not pinned: set one of {list(BLAS_THREAD_VARIABLES)} '
            f'to {expected} before invoking this entry point')
    return {'pinned_threads': int(expected), 'environment': observed,
            'reason': ('a float32 matmul reduction order depends on the thread count, so '
                       'a projected block and a ridge solution are reproducible only at a '
                       'pinned one; the admitted projection ran at 4 threads')}


#: The cohort schemas this gate's nested machinery fits, with the key each one
#: carries its independent measured units under. Two endpoints share the
#: weighting, the folds, the ridge recipe, the control ladder and the group
#: bootstrap, and differ in their unit's name and in whether they declare
#: identity strata, so the shared entry points read this rather than branching on
#: a file path.
COHORT_SCHEMAS = {
    'external_confirmation_cohort_v1': 'domains',
    'remote_homology_cohort_v1': 'backgrounds',
}


def load_cohort(path) -> dict:
    """Read a cohort of either declared schema, with its unit list and its strata.

    A cohort whose schema is not declared is refused rather than guessed at: the
    unit key decides what the independent unit is, and reading the wrong key
    would silently reweight every number downstream.
    """

    import json

    from ..core.io import sha256_file

    cohort = json.loads(Path(path).read_bytes())
    schema = cohort.get('schema')
    if schema not in COHORT_SCHEMAS:
        raise ValueError(f'{schema!r} is not a declared nested-gate cohort schema; '
                         f'expected one of {sorted(COHORT_SCHEMAS)}')
    key = COHORT_SCHEMAS[schema]
    units = cohort[key]
    strata = {row['group']: row['stratum'] for row in units if 'stratum' in row} or None
    return {'cohort': cohort, 'schema': schema, 'unit_key': key, 'units': units,
            'strata': strata, 'sha256': sha256_file(Path(path)),
            'endpoint': cohort['endpoint'], 'endpoint_sha256': cohort['endpoint_sha256'],
            'unit_name': 'domain' if key == 'domains' else 'background'}


def source_frame(path) -> 'object':
    """Read the pinned zip member into a pandas frame, columns as declared."""

    import pandas as pd

    with zipfile.ZipFile(Path(path)) as archive:
        names = archive.namelist()
        if SOURCE_MEMBER not in names:
            raise ValueError(f'{SOURCE_MEMBER} absent from {path}')
        with archive.open(SOURCE_MEMBER) as member:
            frame = pd.read_csv(io.TextIOWrapper(member, encoding='utf-8'),
                                sep='\t', low_memory=False)
    missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f'pinned source is missing declared columns: {missing}')
    return frame


def wildtype_rows(frame) -> 'object':
    """The one row per domain that carries the wild-type sequence.

    The source marks it by an absent ``position``, ``wt_aa`` and ``mut_aa``.
    """

    import pandas as pd

    mask = frame['position'].isna() & frame['wt_aa'].isna() & frame['mut_aa'].isna()
    rows = frame[mask]
    counts = rows.groupby('domain_ID').size()
    if not (counts == 1).all():
        raise ValueError('a domain carries more than one wild-type row')
    if isinstance(rows, pd.DataFrame) and rows['aa_seq'].isna().any():
        raise ValueError('a wild-type row carries no sequence')
    return rows


def accept_rows(frame, wildtypes: dict[str, str]) -> tuple[np.ndarray, dict]:
    """Admit substitution rows with a resolved endpoint, and account for the rest.

    The admission rule is declared: a detected single amino-acid substitution of
    a domain whose wild type is known, with a finite ``normalized_fitness`` and a
    finite ``normalized_fitness_sigma``. Nonsense rows are a separate class and
    are counted, never coerced onto the missense endpoint. Undetected rows carry
    no counts at all and are a library-coverage absence rather than a censored
    measurement: the assay reports a low abundance as a low fitness, so this
    exclusion does not select against destabilising substitutions the way a
    stability assay's resolvability floor does.
    """

    stop = frame['STOP']
    undetected = stop.isna().to_numpy()
    is_wildtype = (frame['position'].isna() & frame['wt_aa'].isna()
                   & frame['mut_aa'].isna()).to_numpy()
    nonsense = (stop.astype('object') == True).to_numpy()  # noqa: E712
    missense = (stop.astype('object') == False).to_numpy()  # noqa: E712
    finite = (np.isfinite(frame['normalized_fitness'].to_numpy(dtype=float))
              & np.isfinite(frame['normalized_fitness_sigma'].to_numpy(dtype=float)))
    known = frame['domain_ID'].isin(wildtypes).to_numpy()
    mask = missense & finite & known & ~is_wildtype
    accounting = {
        'rows': int(len(frame)),
        'wildtype_rows': int(is_wildtype.sum()),
        'undetected_rows_no_counts': int(undetected.sum()),
        'nonsense_rows': int(nonsense.sum()),
        'missense_rows': int(missense.sum()),
        'missense_rows_without_a_finite_endpoint': int((missense & ~finite).sum()),
        'missense_rows_of_a_domain_without_a_wildtype': int((missense & ~known).sum()),
        'accepted_rows': int(mask.sum()),
    }
    return mask, accounting


def substitution_records(frame, mask: np.ndarray, wildtypes: dict[str, str]) -> list[dict]:
    """One record per accepted row, with the declared position offset verified.

    Every record's position, wild-type residue, mutant residue and mutant
    sequence are re-derived against the domain's wild type, so a row whose
    coordinates disagree with its own sequence is refused rather than scored.
    """

    columns = frame.loc[mask, ['domain_ID', 'uniprot_ID', 'aa_seq', 'wt_aa', 'position',
                               'mut_aa', 'normalized_fitness', 'normalized_fitness_sigma',
                               'quality_rank']]
    records = []
    starts = {name: domain_start(name) for name in set(map(str, columns['domain_ID']))}
    for domain, accession, sequence, wild, position, mutant, value, sigma, rank in zip(
            columns['domain_ID'], columns['uniprot_ID'], columns['aa_seq'],
            columns['wt_aa'], columns['position'], columns['mut_aa'],
            columns['normalized_fitness'], columns['normalized_fitness_sigma'],
            columns['quality_rank']):
        wildtype = wildtypes[domain]
        index = int(position) - starts[str(domain)]
        if not 0 <= index < len(wildtype):
            raise ValueError(f'{domain}: position {position} outside the wild type')
        if len(sequence) != len(wildtype):
            raise ValueError(f'{domain}: variant length differs from the wild type')
        if wildtype[index] != wild or sequence[index] != mutant:
            raise ValueError(f'{domain}: row at position {position} disagrees with its sequence')
        differing = [i for i in range(len(wildtype)) if sequence[i] != wildtype[i]]
        if differing != [index]:
            raise ValueError(f'{domain}: row at position {position} is not a single substitution')
        records.append({'domain': str(domain), 'accession': str(accession),
                        'position': index + 1, 'wild': str(wild), 'mutant': str(mutant),
                        'sequence': str(sequence), 'target': float(value),
                        'sigma': float(sigma), 'quality_rank': int(rank)})
    return records


def endpoint_digest(records) -> str:
    """Content digest over the endpoint's own values, in a stable order."""

    payload = '\n'.join(
        f"{row['domain']}\t{row['position']}\t{row['mutant']}\t{row['target']:.10g}"
        for row in sorted(records, key=lambda r: (r['domain'], r['position'], r['mutant'])))
    return hashlib.sha256(payload.encode()).hexdigest()


def draw_order(domain: str, sequences, *, seed: int = DRAW_SEED) -> list[str]:
    """Stable-hash ordering of a domain's mutant sequences.

    The key is the seed, the domain name and the mutant sequence. It inspects no
    fitness, no standard error, no replicate count and no model score, so a draw
    cannot favour large or well-measured effects.
    """

    return sorted(sequences, key=lambda s: hashlib.sha256(
        f'{seed}:{domain}:{s}'.encode()).digest())


# --------------------------------------------------------------------------- #
# Family grouping.
# --------------------------------------------------------------------------- #

def domain_start(domain_id: str) -> int:
    """The domain's first residue position in its parent protein.

    ``A0PJY2_PF00096_289`` starts at residue 289, so its first mutated position
    is 289 and that position is ``aa_seq[0]``. A name whose last field is not an
    integer is refused rather than given a default offset.
    """

    field = domain_id.rsplit('_', 1)[-1]
    if not field.isdigit():
        raise ValueError(f'{domain_id}: last field {field!r} is not a domain start')
    return int(field)


def pfam_accession(domain_id: str) -> str:
    """The Pfam accession the source's ``domain_ID`` carries.

    ``A0A2R8Y422_PF00240_2`` is accession ``A0A2R8Y422``, Pfam ``PF00240``, start
    2. A name that carries no ``PF`` field is refused rather than silently
    treated as its own family.
    """

    fields = [field for field in domain_id.split('_') if field.startswith('PF')]
    if len(fields) != 1:
        raise ValueError(f'{domain_id}: no single Pfam field in the domain identifier')
    return fields[0]


def alignment_statistics(sequences: list[str], pairs: np.ndarray):
    """Exact Smith-Waterman statistics under the frozen contract's scoring."""

    codes, lengths = encode(sequences)
    return batch_align(codes, lengths, pairs, gap_open=ALIGNMENT_GAP_OPEN,
                       gap_extend=ALIGNMENT_GAP_EXTEND)


def family_groups(domains: list[str], wildtypes: dict[str, str],
                  accessions: dict[str, str]) -> tuple[dict[str, str], dict]:
    """Single-linkage family groups over the declared union sources.

    The alignment source is the frozen grouping contract's own edge rule --
    Smith-Waterman with BLOSUM62 and NCBI default affine gaps, an edge when
    percent identity is at least 30.0 and coverage is at least 80.0 for both
    sequences. Two sources the contract cannot express for this dataset are
    declared here: two domains cut from one UniProt accession are one group, and
    two domains the source library annotates with one Pfam accession are one
    group. Every domain carries exactly one Pfam label, so unlike the contract's
    hmmscan annotation there is no unlabelled stratum.
    """

    order = sorted(domains)
    union = Union(order)
    sequences = [wildtypes[name] for name in order]
    by_sequence: dict[str, list[str]] = {}
    for name, sequence in zip(order, sequences):
        by_sequence.setdefault(sequence, []).append(name)
    for members in by_sequence.values():
        for other in members[1:]:
            union.join(members[0], other, 'exact_sequence')
    for key, source in (('accession', 'accession'), ('pfam', 'pfam')):
        buckets: dict[str, list[str]] = {}
        for name in order:
            label = accessions[name] if key == 'accession' else pfam_accession(name)
            buckets.setdefault(label, []).append(name)
        for members in buckets.values():
            for other in members[1:]:
                union.join(members[0], other, source)
    index = np.triu_indices(len(order), k=1)
    pairs = np.stack(index, axis=1)
    statistics = alignment_statistics(sequences, pairs)
    edge = statistics.edges(identity_floor=GROUPING_IDENTITY,
                            coverage_floor=GROUPING_COVERAGE)
    for row in np.flatnonzero(edge):
        union.join(order[int(pairs[row, 0])], order[int(pairs[row, 1])], 'alignment')
    components = union.components()
    labels = {}
    for number, root in enumerate(sorted(components), start=1):
        for member in components[root]:
            labels[member] = f'dom-{number:03d}'
    sizes = Counter(labels.values())
    report = {
        'domains': len(order),
        'groups': len(components),
        'group_size_max': int(max(sizes.values())),
        'group_size_median': float(np.median(sorted(sizes.values()))),
        'singleton_groups': int(sum(1 for value in sizes.values() if value == 1)),
        'alignment_edges': int(edge.sum()),
        'exact_sequence_edges': int(sum(len(m) - 1 for m in by_sequence.values())),
        'union_sources': list(UNION_SOURCES),
        'edge_rule': (f'percent identity >= {GROUPING_IDENTITY} and coverage >= '
                      f'{GROUPING_COVERAGE} for both sequences, Smith-Waterman with '
                      f'BLOSUM62 and gap {ALIGNMENT_GAP_OPEN:.0f} + '
                      f'{ALIGNMENT_GAP_EXTEND:.0f}k'),
    }
    return labels, report


# --------------------------------------------------------------------------- #
# Nested weighting, group errors and effective unit counts.
# --------------------------------------------------------------------------- #

def nested_weights(group: np.ndarray, domain: np.ndarray, site: np.ndarray) -> np.ndarray:
    """Sum-one weights: groups equal, domains equal in a group, sites equal in a
    domain, variants equal in a site.

    The development fit's two-level weighting assumed one background per group.
    This support carries several domains in a group, so a two-level weighting
    would weight a domain by its site count and let a long domain dominate its
    family. Three-level nesting is the convention the stability gate's own
    endpoint qualification declares for exactly this case.
    """

    group, domain, site = np.asarray(group), np.asarray(domain), np.asarray(site)
    if not (len(group) == len(domain) == len(site)):
        raise ValueError('nested weighting needs aligned group, domain and site labels')
    domain_group, site_domain = {}, {}
    for g, d, s in zip(group, domain, site):
        if domain_group.setdefault(d, g) != g:
            raise ValueError(f'domain {d} assigned to more than one group')
        if site_domain.setdefault(s, d) != d:
            raise ValueError(f'site {s} assigned to more than one domain')
    groups = len(set(domain_group.values()))
    domains_per_group = Counter(domain_group.values())
    sites_per_domain = Counter(site_domain.values())
    rows_per_site = Counter(site.tolist())
    return np.asarray([
        1.0 / (groups * domains_per_group[g] * sites_per_domain[d] * rows_per_site[s])
        for g, d, s in zip(group, domain, site)], dtype=float)


def group_errors(target: np.ndarray, prediction: np.ndarray, group: np.ndarray,
                 domain: np.ndarray, site: np.ndarray,
                 keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-group mean squared error in squared normalised-fitness units.

    The nesting is the one :func:`nested_weights` realises, recomputed inside the
    retained rows so that a declared evaluation restriction renormalises rather
    than silently reweighting. A group with no retained row is dropped from the
    returned labels and counted by the caller.
    """

    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    group, domain, site = np.asarray(group), np.asarray(domain), np.asarray(site)
    rows = np.arange(len(target)) if keep is None else np.flatnonzero(keep)
    labels, values = [], []
    for label in sorted(set(group[rows].tolist())):
        in_group = rows[group[rows] == label]
        per_domain = []
        for name in sorted(set(domain[in_group].tolist())):
            in_domain = in_group[domain[in_group] == name]
            per_site = [float(np.mean((target[cell] - prediction[cell]) ** 2))
                        for cell in (in_domain[site[in_domain] == key]
                                     for key in sorted(set(site[in_domain].tolist())))]
            per_domain.append(float(np.mean(per_site)))
        labels.append(label)
        values.append(float(np.mean(per_domain)))
    return np.asarray(labels), np.asarray(values, dtype=float)


def kish_units(group: np.ndarray, domain: np.ndarray, site: np.ndarray) -> dict:
    """Effective unit counts under the weighting the estimator actually applies.

    Three counts are reported because three levels carry weight, and each is the
    Kish count of that level's own weights under this nesting. Quoting one of
    them under a different convention is what this programme had to correct in
    four places, so the convention travels with the number.
    """

    weights = nested_weights(group, domain, site)
    group, domain, site = np.asarray(group), np.asarray(domain), np.asarray(site)
    out = {'weighting': ('groups equal, domains equal inside a group, sites equal inside a '
                         'domain, variants equal inside a site')}
    for level, labels in (('group', group), ('domain', domain), ('site', site)):
        totals: dict[str, float] = {}
        for label, weight in zip(labels.tolist(), weights):
            totals[label] = totals.get(label, 0.0) + float(weight)
        values = np.asarray(list(totals.values()), dtype=float)
        out[f'{level}s'] = int(len(values))
        out[f'kish_effective_{level}s'] = float(values.sum() ** 2 / (values ** 2).sum())
    out['variants'] = int(len(group))
    return out


def row_identity(group: np.ndarray, site: np.ndarray, target: np.ndarray) -> str:
    """Digest of the evaluated rows: group, site and target, in order.

    The target is rendered through ``float`` before ``repr`` rather than left as
    a NumPy scalar, because ``repr(np.float64(0.5))`` differs between NumPy 1.26
    and NumPy 2 and a NumPy-scalar rendering would hash identical rows to
    different values across interpreters.
    """

    if not (len(group) == len(site) == len(target)):
        raise ValueError('row identity needs aligned group, site and target arrays')
    return hashlib.sha256('\n'.join(
        f'{g}|{s}|{float(t)!r}' for g, s, t in zip(group, site, target)).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Panel assembly, the nonlinear response and the nested held-group comparison.
# --------------------------------------------------------------------------- #

def build_panel(units: list[dict], profiles: dict) -> dict:
    """Rows, target, labels and every model-free control block.

    ``units`` is the cohort's list of independent measured units -- a Domainome
    domain here, a MGnify background in the remote-homology gate -- each carrying
    its name, wild type, family group and admitted variants. The argument is the
    unit list rather than the cohort so that one panel builder serves both
    endpoints without either cohort having to rename its own unit key.
    """

    target, group, domain, site, rank, uncertainty = [], [], [], [], [], []
    ident, geom, comp, chem, prof, prof2 = [], [], [], [], [], []
    for row in units:
        name, wildtype = row['name'], row['wildtype']
        profile = profiles[name]
        for variant in row['variants']:
            position = variant['position'] - 1
            mutant, sequence = variant['mutant'], variant['sequence']
            target.append(variant['target'])
            uncertainty.append(variant['uncertainty'])
            rank.append(row.get('quality_rank', 0))
            group.append(row['group'])
            domain.append(name)
            site.append(f"{name}:{variant['position']}")
            ident.append(identity_block(wildtype, position, mutant))
            geom.append(geometry_block(position, len(wildtype)))
            comp.append(composition_block(wildtype, sequence))
            chem.append(chemistry_block(wildtype, position, mutant))
            prof.append(profile_block(profile, wildtype, position, mutant))
            prof2.append(profile_bounded_block(profile, wildtype, position, mutant))
    panel = {
        'target': np.asarray(target, dtype=float),
        # The source's own reported uncertainty of each row, carried as reported
        # and named for what it is rather than as a standard error: the Domainome
        # cohort carries `normalized_fitness_sigma` and the remote-homology
        # cohort carries a 95% interval width, which are different quantities.
        # Nothing here weights, shrinks or thresholds on it.
        'uncertainty': np.asarray(uncertainty, dtype=float),
        'quality_rank': np.asarray(rank, dtype=int),
        'group': np.asarray(group), 'domain': np.asarray(domain), 'site': np.asarray(site),
        'blocks': {'ident': np.asarray(ident, dtype=float),
                   'geom': np.asarray(geom, dtype=float),
                   'comp': np.asarray(comp, dtype=float),
                   'chem': np.asarray(chem, dtype=float),
                   'prof': np.asarray(prof, dtype=float),
                   'prof2': np.asarray(prof2, dtype=float)},
    }
    panel['weights'] = nested_weights(panel['group'], panel['domain'], panel['site'])
    return panel


def nuisance_response(panel: dict, columns: tuple[str, ...], blocks: dict,
                      held: np.ndarray, inner_partition: list, *,
                      device='cpu') -> tuple[np.ndarray, dict]:
    """The nonlinear global response G for one outer fold.

    A first-stage ridge predictor of the endpoint is fitted on the outer training
    groups only, its penalty selected on the same inner held-group partition the
    outer tuning uses, and the training rows' predictions are cross-fitted on
    that partition. An isotonic response is fitted on those cross-fitted
    predictions against the training targets and applied to every row's
    first-stage prediction. The block is the calibrated response and the raw
    prediction, so a monotone nonlinearity between a linear sequence-statistical
    prediction and measured abundance can be accounted for before any model
    quantity is read. The raw prediction lies in the linear span of ``columns``
    and is carried only so the block's two coordinates are stated in full; what
    it adds to a design already holding ``columns`` is the monotone transform.

    No held-out group label reaches the first stage, its penalty choice or the
    response.
    """

    from sklearn.isotonic import IsotonicRegression

    x = np.column_stack([blocks[key] for key in columns])
    y, groups, weights = panel['target'], panel['group'], panel['weights']
    train = np.flatnonzero(~np.isin(groups, held))
    test = np.flatnonzero(np.isin(groups, held))
    if not len(train) or not len(test):
        raise ValueError('nuisance fold has an empty side')
    losses = np.zeros(len(ALPHAS))
    crossfit = np.full((len(train), len(ALPHAS)), np.nan)
    position = {index: order for order, index in enumerate(train)}
    for validation in inner_partition:
        fit = train[~np.isin(groups[train], validation)]
        valid = train[np.isin(groups[train], validation)]
        if not len(fit) or not len(valid):
            raise ValueError('nuisance inner fold has an empty side')
        prediction = ridge_predict(x[fit], y[fit], weights[fit], x[valid], ALPHAS, device)
        crossfit[[position[index] for index in valid]] = prediction
        share = weights[valid] / weights[valid].sum()
        losses += len(validation) * (share[:, None] * (prediction - y[valid][:, None]) ** 2).sum(0)
    if not np.isfinite(crossfit).all():
        raise ValueError('incomplete cross-fitted first-stage predictions')
    index = _select_alpha(losses / len(set(groups[train])))
    alpha = ALPHAS[index]
    fitted = np.empty(len(y))
    fitted[train] = crossfit[:, index]
    fitted[test] = ridge_predict(x[train], y[train], weights[train], x[test], [alpha], device)[:, 0]
    response = IsotonicRegression(increasing=True, out_of_bounds='clip')
    response.fit(crossfit[:, index], y[train], sample_weight=weights[train])
    block = np.column_stack([response.predict(fitted), fitted])
    share = weights[test] / weights[test].sum()
    centre = float((share * y[test]).sum())
    residual = float((share * (y[test] - fitted[test]) ** 2).sum())
    variance = float((share * (y[test] - centre) ** 2).sum())
    low, high = float(crossfit[:, index].min()), float(crossfit[:, index].max())
    diagnostics = {
        'selected_alpha': alpha,
        'first_stage_columns': list(columns),
        'first_stage_width': int(x.shape[1]),
        'held_out_rows': int(len(test)),
        'held_out_weighted_r2': None if variance <= 0 else 1.0 - residual / variance,
        'response_range': float(np.diff(response.predict(np.array([low, high])))[0]),
    }
    return block, diagnostics


def fold_predictions(panel: dict, blocks: dict, design: dict,
                     *, seed: int, first_stage: tuple[str, ...] | None = None,
                     device='cpu') -> dict:
    """Held-group predictions for every named design, on one seed.

    Every design sees identical rows, identical outer and inner group partitions
    and the identical label budget; only its declared column blocks differ.
    Centring and scaling are fitted on training rows inside the ridge recipe, so
    no held-out group label reaches feature scaling, penalty selection or the
    nonlinear response.
    """

    groups, weights = panel['group'], panel['weights']
    predictions = {name: np.full(len(groups), np.nan) for name in design}
    records, nuisance = [], []
    needs_g = any('G' in columns for columns in design.values())
    if needs_g and first_stage is None:
        raise ValueError('a design naming G requires a declared first-stage column set')
    for outer, held in enumerate(family_folds(groups, OUTER_SPLITS, seed)):
        held = list(held)
        train = np.flatnonzero(~np.isin(groups, held))
        test = np.flatnonzero(np.isin(groups, held))
        if not len(train) or not len(test):
            raise ValueError(f'fold {outer} has an empty side')
        inner_partition = family_folds(groups[train], INNER_SPLITS, seed + 100 + outer)
        fold_blocks = dict(blocks)
        if needs_g:
            block, diagnostics = nuisance_response(
                panel, first_stage, blocks, np.asarray(held), inner_partition, device=device)
            diagnostics.update(fold=outer, held_groups=held)
            nuisance.append(diagnostics)
            fold_blocks['G'] = block
        record = {'fold': outer, 'held_groups': held,
                  'training_groups': sorted(set(groups[train].tolist())),
                  'alpha': {}, 'dimensions': {}}
        for name, columns in design.items():
            x = np.column_stack([fold_blocks[key] for key in columns])
            if not np.isfinite(x).all():
                raise ValueError(f'{name}: nonfinite design matrix')
            losses = np.zeros(len(ALPHAS))
            for validation in inner_partition:
                fit = train[~np.isin(groups[train], validation)]
                valid = train[np.isin(groups[train], validation)]
                predicted = ridge_predict(x[fit], panel['target'][fit], weights[fit],
                                          x[valid], ALPHAS, device)
                losses += len(validation) * (
                    weights[valid][:, None]
                    * (predicted - panel['target'][valid][:, None]) ** 2).sum(0)
            alpha = ALPHAS[_select_alpha(losses / len(set(groups[train].tolist())))]
            predictions[name][test] = ridge_predict(
                x[train], panel['target'][train], weights[train], x[test], [alpha], device)[:, 0]
            record['alpha'][name] = alpha
            record['dimensions'][name] = int(x.shape[1])
        records.append(record)
    for name, value in predictions.items():
        if not np.isfinite(value).all():
            raise ValueError(f'{name}: incomplete held-out predictions')
    predictions['NO_EFFECT_NULL'] = np.zeros(len(groups))
    return {'predictions': predictions, 'folds': records, 'nuisance': nuisance}


def paired_increment(panel: dict, predictions: dict, augmented: str, baseline: str,
                     *, keep: np.ndarray | None = None, draws: int = BOOTSTRAP_DRAWS,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired per-group reduction in mean squared error, squared fitness units."""

    labels, base = group_errors(panel['target'], predictions[baseline], panel['group'],
                                panel['domain'], panel['site'], keep)
    _, better = group_errors(panel['target'], predictions[augmented], panel['group'],
                             panel['domain'], panel['site'], keep)
    record = interval(base - better, draws=draws, seed=seed)
    record.update(baseline=baseline, augmented=augmented, unit='squared normalised fitness',
                  baseline_mse=float(base.mean()), augmented_mse=float(better.mean()),
                  evaluated_groups=int(len(labels)))
    return record


def spearman_increment(panel: dict, predictions: dict, augmented: str, baseline: str,
                       *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired within-domain rank-correlation increment, averaged into groups."""

    record = {}
    for name, key in (('baseline', baseline), ('augmented', augmented)):
        record[name] = _domain_spearman(panel, predictions[key])
    paired = [None if a is None or b is None else b - a
              for a, b in zip(record['baseline'], record['augmented'])]
    out = interval(paired, draws=draws, seed=seed)
    out.update(baseline=baseline, augmented=augmented, unit='dimensionless Spearman')
    return out


def _domain_spearman(panel: dict, prediction: np.ndarray) -> list:
    """Within-domain Spearman averaged over the domains of each group."""

    labels, _ = group_spearman(panel['target'], prediction, panel['domain'])
    per_domain = dict(zip(labels, _))
    domain_group = dict(zip(panel['domain'].tolist(), panel['group'].tolist()))
    buckets: dict[str, list[float]] = {}
    for name, value in per_domain.items():
        buckets.setdefault(domain_group[name], []).append(value)
    out = []
    for label in sorted(buckets):
        defined = [v for v in buckets[label] if v is not None and np.isfinite(v)]
        out.append(float(np.mean(defined)) if defined else None)
    return out


def raw_spearman(panel: dict, values: np.ndarray, *, draws: int = BOOTSTRAP_DRAWS,
                 seed: int = BOOTSTRAP_SEED) -> dict:
    """Unfitted within-domain rank correlation of one scalar against the endpoint."""

    record = interval(_domain_spearman(panel, np.asarray(values, dtype=float)),
                      draws=draws, seed=seed)
    record.update(unit='dimensionless Spearman')
    return record


def qualify(increments: dict[int, float]) -> dict:
    """Apply the qualification rule to one candidate's per-seed increments.

    A candidate is kept only if its paired reduction in group-equal held-out mean
    squared error over the standing set is positive at every prespecified split
    seed. A candidate that lowers the standing set's held-out error at any seed
    is discarded and reported, never carried forward.
    """

    if set(increments) != set(SPLIT_SEEDS):
        raise ValueError('a candidate must be scored at every prespecified split seed')
    values = [float(increments[seed]) for seed in SPLIT_SEEDS]
    return {'per_seed_increment': {str(s): v for s, v in zip(SPLIT_SEEDS, values)},
            'min_increment': min(values), 'unit': 'squared normalised fitness',
            'qualified': bool(min(values) > 0.0)}


def secondary_control_set(controls: dict) -> tuple[tuple[str, ...], dict]:
    """The control set the correlation rule accepts, derived from the frozen ladder.

    The prespecified qualification rule is stated on group-equal held-out mean
    squared error, and the capability map's binding rule for a control is stated
    on held-out correlation. Where the two disagree about one block -- the
    mutation-local profile transfers rank and does not transfer level on the
    development endpoint -- the gate does not resolve the disagreement by
    choosing a metric after seeing it. Both sets are carried: the primary set is
    the one the mean-squared-error rule qualified, and this secondary set adds
    every discarded candidate whose correlation increment is positive at all
    three split seeds, in ladder order, with a superseded candidate replaced by
    its restatement.

    ``SUPERSEDED`` is imported from the stability gate rather than restated, so
    which block supersedes which is one declaration across the three endpoints
    that use these blocks. This function differs from that gate's own only in
    reading the unit-neutral ``per_seed_increment`` key, because these two
    endpoints are not in kcal²/mol².
    """

    from ..stability.stability_gate import SUPERSEDED

    kept = list(controls['qualified_control_set'])
    added, reasons = [], {}
    for row in controls['ladder']:
        candidate = row['candidate']
        if row['qualified'] or candidate in SUPERSEDED or candidate in kept:
            continue
        points = [row['spearman_increment'][str(seed)]['point'] for seed in SPLIT_SEEDS]
        if any(point is None or point <= 0 for point in points):
            continue
        added.append(candidate)
        reasons[candidate] = {'per_seed_spearman_increment': points,
                              'per_seed_squared_error_increment': list(
                                  row['per_seed_increment'].values())}
    wanted, ordered = set(kept) | set(added), []
    for candidate in controls['candidate_order']:
        block = SUPERSEDED.get(candidate, candidate)
        if block in wanted and block not in ordered:
            ordered.append(block)
    secondary = tuple([*controls['base_blocks'], *ordered])
    return secondary, {
        'added_over_primary': added,
        'rule': ('the primary set is the one the mean-squared-error rule qualified; this '
                 'secondary set adds every discarded candidate whose correlation increment '
                 'is positive at all three split seeds, with a superseded candidate replaced '
                 'by its restatement. A squared-error increment over the secondary set and a '
                 'correlation increment over the primary set are reported as sensitivities '
                 'beside the licensed pair, not as findings'),
        'evidence': reasons}


def declaration_digest(payload: dict) -> str:
    """Content digest over a declaration's canonical JSON form."""

    import json

    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


__all__ = [
    'ALIGNMENT_GAP_EXTEND', 'ALIGNMENT_GAP_OPEN', 'ARM_CANDIDATE_BLOCKS',
    'BASE_BLOCKS', 'BLAS_THREADS', 'BLAS_THREAD_VARIABLES', 'CANDIDATE_BLOCKS',
    'DEVELOPMENT_COVERAGE',
    'DEVELOPMENT_IDENTITY', 'DEVELOPMENT_STRATUM_IDENTITY', 'DRAW_SEED',
    'ENDPOINT', 'FEATURE_BLOCKS', 'G_FEATURE_ORDER', 'GROUPING_COVERAGE',
    'GROUPING_IDENTITY', 'MEASURED_QUANTITY', 'MIN_VARIANTS', 'MODEL_BLOCKS',
    'POSITION_CONVENTION', 'PROJECTION_DIM', 'PROJECTION_SEED', 'REPLICATES',
    'SOURCE_ARCHIVE', 'SOURCE_ARCHIVE_SHA256', 'SOURCE_COLUMNS', 'SOURCE_DOI',
    'SOURCE_MEMBER', 'SPLIT_SEEDS', 'TOKENISATION_FEATURE_ORDER',
    'UNION_SOURCES', 'VARIANT_CAP', 'accept_rows', 'alignment_statistics',
    'build_panel', 'chemistry_width', 'declaration_digest', 'domain_start', 'draw_order',
    'endpoint_digest', 'family_groups', 'fold_predictions', 'group_errors',
    'interval', 'kish_units', 'nested_weights', 'nuisance_response',
    'COHORT_SCHEMAS', 'load_cohort', 'secondary_control_set',
    'paired_increment', 'pfam_accession', 'projection_matrices', 'qualify',
    'raw_spearman', 'require_blas_threads', 'row_identity', 'source_frame',
    'spearman_increment',
    'substitution_records', 'tokenisation_block', 'wildtype_rows',
]
