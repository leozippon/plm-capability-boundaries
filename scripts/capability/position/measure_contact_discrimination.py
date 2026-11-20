#!/usr/bin/env python3
"""Whether explicit couplings or model propagation localise a residue contact.

E01 reports the mean response of contacting receivers minus matched non-contacting
ones. That contrast says whether contact shifts a mean; it does not say whether a
pair can be *identified* as a contact from either quantity. This stage asks the
identification question on the canonical endpoint -- is the pair ``(i, j)`` in
three-dimensional contact -- and puts the project's explicit evolutionary statistic
and its model readout on one axis, so the two are read against the same labels, the
same pair population and the same resampling unit.

**The endpoint.** The contact definition is
:func:`..position.contact_response.contact_pairs`'s own: C-beta, or C-alpha for
glycine, strictly below ``--contact-angstrom`` between two positions whose wild-type
sequence separation is at least ``--min-separation``. A position without an admitted
experimental coordinate enters neither class.

**The explicit rung.** The average-product-corrected Frobenius norm of the fitted
Potts coupling block, the standard form a coupling model is read in for this
endpoint. The block is first put into the project's one declared gauge
(:func:`...context.potts_gauge.to_zero_sum`), because a Frobenius norm is not
gauge-invariant and a norm taken in an unstated gauge is not a defined quantity.
The parameters are the ones already fitted for the interaction lane; nothing is
refitted here.

**The model rung.** Each arm's own per-pair propagation response, rebuilt from the
retained position-likelihood archives through the project's own validator: the
admitted variants and the mutated position of each are
:class:`...extensions.responses.RetainedResponses`'s, read against the cohort the
extraction was run on, and never parsed out of a mutation label. That matters because
the archives also carry multi-substitution states, for which a single sending
position does not exist. The per-receiver census is then
:func:`..position.contact_response.receiver_census`, the same one the frozen E01
reader uses, reduced to one value per pair by averaging the absolute response over
the admitted substitutions at the sending position. A causal arm contributes only the pairs it can carry: its
response upstream of the substitution is identically zero because the prefix that
predicts it is unchanged, so those pairs are excluded as undefined for that arm
rather than entered as zeros.

**The separation rung**, mandatory. Contact frequency falls with sequence separation,
so a discrimination that is really about sequence distance is the failure this design
exists to exclude. Negative separation is scored as its own column and every other
column is read beside it.

**The arm support floor.** A column enters the family only if it carries at least
:data:`...core.statistics.MINIMUM_BOOTSTRAP_UNITS` references of its own. An arm whose
byte-pair segmentation leaves almost no single-residue receiver cannot be compared
with a full-alphabet arm, and admitting it would cost every other column the
references it lacks, because the complete case is an intersection. Refused arms are
named with the coverage they have, the way E01 names the arms its own five-family
floor holds out, rather than being dropped silently or carried as a near-empty column.

**The statistic.** One reference's discrimination is the rank-sum area under the
curve of its pairs, minus one half, so zero is chance and the sign is readable. The
resampling unit is the biological family, which is E01's own unit, and the band is
the project's declared native one: the paired group bootstrap's maximum absolute
centered deviation over the whole family of columns, in the endpoint's own units
(:func:`...extensions.bidirectional.paired_bands`). A reference whose pairs carry one
class only has no AUC and is dropped by name.

This stage establishes a scope limit, not a capability. It is written to be read as
"on this cohort, neither rung localises contacts", and the support it rests on --
short domains with shallow alignments -- is recorded beside every number so the
limit is not read wider than the cohort that produced it.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.context.potts_gauge import to_zero_sum  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS  # noqa: E402
from src.capability.extensions import bidirectional as bidir  # noqa: E402
from src.capability.extensions.responses import RetainedResponses  # noqa: E402
from src.capability.position.contact_response import (  # noqa: E402
    CONTACT_ANGSTROM,
    MIN_SEQUENCE_SEPARATION,
    admitted_geometry,
    contact_pairs,
    rebuild_states,
    receiver_census,
    structural_rows,
)
from src.capability.position.position_likelihood import read_archive  # noqa: E402
from src.capability.stability.pairwise_stability import read_plmc  # noqa: E402

#: The two paradigm labels the retained archives carry, and which side of a
#: substitution each one can carry a response on. A causal arm's upstream terms are
#: identically zero because the prefix is unchanged, so an upstream pair is undefined
#: for that arm and is dropped rather than scored as a zero response.
PARADIGM_SIDES = {
    'causal_next_token': ('downstream',),
    'masked_marginal_pseudolikelihood': ('downstream', 'upstream'),
}

#: How a pair's many mutations become one value per pair, declared because the choice
#: is not forced: the sending position carries several substitutions and their
#: responses differ in sign. The absolute response is averaged, which is the quantity
#: E01's own contact contrast is formed on.
PAIR_REDUCTION = ('mean absolute per-position response in nats over the admitted '
                  'single substitutions at the sending position')

#: What an AUC of one half means here, stated so a null is not read as an absence.
CHANCE_READING = (
    'the statistic is a rank-sum area under the curve minus one half, so zero is '
    'chance discrimination on that reference\'s own pair population. An interval '
    'containing zero is a limit on resolution and never evidence that the quantity '
    'carries no contact information; this design declares no equivalence margin')

#: Minimum fraction of eligible residue pairs an arm must cover across admitted
#: references to enter the full-alphabet secondary summary. Full-alphabet arms
#: (bidirectional or single-residue causal) cover ~65-88% of pairs; thin BPE
#: tokenizers cover <13% and both thin the complete-case reference intersection
#: and inflate the simultaneous band critical value.
FULL_ALPHABET_MIN_PAIR_FRACTION = 0.50


def effective_depth_per_site(work: Path, length: int) -> float:
    """Neff over modelled columns, from the weight file the fit itself consumed.

    Taken from this fit's own alignment because that is the alignment whose couplings
    the comparison reads. Another campaign's fit of the same backbone can rest on a
    different search and a different depth, and quoting that one here would describe a
    model this stage never used.
    """

    weights = np.array([float(value) for value in (work / 'full.weights').read_text().split()],
                       dtype=np.float64)
    if not len(weights) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(f'{work}: the fit\'s weight file is empty, nonfinite or negative')
    return float(weights.sum()) / length


def paired_bands_with_marginal(values, *, draws: int = bidir.DRAWS, seed: int = bidir.SEED,
                               confidence: float = bidir.CONFIDENCE) -> dict:
    """Simultaneous native bands plus marginal percentile CIs on the same bootstrap draws."""

    bands = bidir.paired_bands(values, draws=draws, seed=seed, confidence=confidence)
    resampled = bidir.resample_means(values, draws=draws, seed=seed)
    alpha = 0.5 * (1.0 - confidence)
    low = np.quantile(resampled, alpha, axis=0)
    high = np.quantile(resampled, 1.0 - alpha, axis=0)
    marginal = np.column_stack([low, high]).tolist()
    return {**bands, 'marginal_interval': marginal}


def pair_coupling_scores(record) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """Aligned 0-based pair keys, contact labels, coupling norms and negative separation.

    ``record['pairs']['distance']`` keys come from :func:`structure_sites` and
    :func:`contact_pairs` and are already 0-based wild-type residue indices
    (``0 <= i < j < L``), matching the 0-based axes of :func:`apc_frobenius`.
    """

    keys = sorted(record['pairs']['distance'])
    length = int(record['length'])
    couplings = np.asarray(record['couplings'], dtype=float)
    if couplings.shape != (length, length):
        raise ValueError(
            f'{record.get("assay", "<unknown>")}: coupling matrix shape {couplings.shape} '
            f'disagrees with wild-type length {length}')
    for low, high in keys:
        if not (0 <= low < high < length):
            raise ValueError(
                f'{record.get("assay", "<unknown>")}: pair key ({low}, {high}) is outside '
                f'0 <= i < j < {length}')
    label = np.array([record['pairs']['distance'][key] < record['pairs']['cutoff_angstrom']
                      for key in keys], dtype=bool)
    potts = np.array([couplings[key[0], key[1]] for key in keys], dtype=float)
    gap = np.array([-(key[1] - key[0]) for key in keys], dtype=float)
    return keys, label, potts, gap


def auc_minus_half(score, label) -> float | None:
    """Rank-sum AUC of ``score`` against boolean ``label``, minus one half.

    Ties take mid-ranks, which is what makes the statistic well defined when a
    coupling norm repeats exactly. ``None`` where one class is absent, because a
    reference with no contact or no non-contact has no discrimination to report.
    """

    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    if score.shape != label.shape or score.ndim != 1:
        raise ValueError('a discrimination needs one score per labelled pair')
    if not np.isfinite(score).all():
        raise ValueError('a discrimination needs finite scores')
    positives = int(label.sum())
    negatives = int(len(label) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score, kind='stable')
    ranks = np.empty(len(score), dtype=float)
    sorted_score = score[order]
    start = 0
    for stop in range(1, len(score) + 1):
        if stop == len(score) or sorted_score[stop] != sorted_score[start]:
            ranks[order[start:stop]] = 0.5 * (start + stop + 1)
            start = stop
    return float(ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives) - 0.5


def apc_frobenius(params) -> np.ndarray:
    """Average-product-corrected Frobenius norms of a plmc fit, in the zero-sum gauge.

    The gauge is fixed first through the project's one convention, because the
    Frobenius norm of a coupling block changes under a gauge transformation and a
    norm reported without its gauge names no quantity. The average-product correction
    is the standard one, subtracting the outer product of the row means over their
    grand mean.
    """

    length = len(params.target)
    couplings = np.asarray(params.couplings, dtype=np.float64)
    states = couplings.shape[1]
    dense = np.zeros((length, length, states, states), dtype=np.float64)
    index = 0
    for low in range(length):
        for high in range(low + 1, length):
            dense[low, high] = couplings[index]
            dense[high, low] = couplings[index].T
            index += 1
    if index != couplings.shape[0]:
        raise ValueError('the plmc pair count disagrees with the modelled length')
    _, gauged = to_zero_sum(np.asarray(params.fields, dtype=np.float64), dense)
    norms = np.sqrt((gauged ** 2).sum(axis=(2, 3)))
    np.fill_diagonal(norms, 0.0)
    row = norms.sum(axis=1) / (length - 1)
    grand = norms.sum() / (length * (length - 1))
    corrected = norms - np.outer(row, row) / grand
    np.fill_diagonal(corrected, 0.0)
    return corrected


def arm_pair_responses(archive: Path, geometry, pairs, row, *, assay: str,
                       paradigm: str) -> dict:
    """One arm's mean absolute response per eligible pair of one reference.

    The admitted variants and each one's mutated position come from the frozen
    reader, so this stage inherits the per-state closure, the native mutation closure
    and the biological exclusions rather than re-deciding any of them.
    """

    sides = PARADIGM_SIDES[paradigm]
    payload = read_archive(archive)
    states = rebuild_states(payload, [row['wildtype'], *row['sequences']])
    identity = {'wildtype': row['wildtype'], 'mutants': row['mutants'],
                'sequences': row['sequences']}
    with np.load(archive, allow_pickle=False) as data:
        retained = RetainedResponses(data, states, identity)
        frozen = {index: retained.response(index) for index in retained.selected_indices}
    collected: dict = {}
    for index, response in frozen.items():
        census = receiver_census(payload, index, site=int(response['i']), paradigm=paradigm,
                                 upstream_tolerance=0.0)
        for entry in structural_rows(census, geometry, pairs, assay=assay,
                                     family=row['cluster'], mutation=row['mutants'][index]):
            if entry['direction'] not in sides:
                continue
            key = (min(entry['i'], entry['j']), max(entry['i'], entry['j']))
            collected.setdefault(key, []).append(float(entry['absolute_response']))
    return {'values': {key: float(np.mean(values)) for key, values in collected.items()},
            'admitted_variants': len(frozen)}


def reference_rows(sites_by_assay, coverage, params_roots, *, cutoff: float,
                   separation: int):
    """Geometry, contact labels and coupling norms for every reference that has both.

    ``params_roots`` are searched in the order given and the first hit wins. The
    directories are named rather than discovered, because the repository holds
    convergence and smoke fits of the same backbones beside the published ones and a
    recursive search would mix fits from different runs into one comparison.
    """

    admitted, refused = {}, []
    for assay, rows in sorted(sites_by_assay.items()):
        entry = assay.rsplit('_', 1)[-1]
        candidates = [root / f'{entry}.pdb' / 'selected.params' for root in params_roots]
        candidates = [path for path in candidates if path.exists()]
        if not candidates:
            continue
        geometry = admitted_geometry(rows, assay=assay, family=rows[0]['cluster'],
                                     wildtype=rows[0]['wildtype'], coverage=coverage)
        pairs = contact_pairs(geometry, cutoff=cutoff, min_separation=separation)
        params = read_plmc(candidates[0])
        if params.target != rows[0]['wildtype']:
            refused.append({'assay': assay, 'reason': (
                'the fitted coupling model\'s focus columns are not this assay\'s wild type, '
                'so a pair index in one is not the same pair in the other')})
            continue
        admitted[assay] = {
            'assay': assay, 'family': rows[0]['cluster'], 'entry': entry,
            'length': len(rows[0]['wildtype']), 'geometry': geometry, 'pairs': pairs,
            'couplings': apc_frobenius(params),
            'params': str(candidates[0]),
            'params_sha256': sha256_file(candidates[0]),
        }
    return admitted, refused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--structures', type=Path, required=True,
                        help='the sites.json.gz the structural annotation stage wrote')
    parser.add_argument('--coverage', type=Path, required=True,
                        help='the structural coverage receipt paired with --structures')
    parser.add_argument('--cohort', type=Path, required=True,
                        help='the cohort the extractions were run on; the frozen reader needs '
                             'its wild type, mutant labels and sequences to admit a variant')
    parser.add_argument('--params-root', type=Path, action='append', required=True,
                        help='a directory holding <entry>.pdb/selected.params plmc fits; '
                             'repeatable, searched in the order given, first hit wins')
    parser.add_argument('--extraction', type=Path, action='append', required=True,
                        help='a posresp_extract_* directory; repeatable, once per arm')
    parser.add_argument('--contact-angstrom', type=float, default=CONTACT_ANGSTROM)
    parser.add_argument('--min-separation', type=int, default=MIN_SEQUENCE_SEPARATION)
    parser.add_argument('--min-neff-per-site', type=float, default=0.,
                        help='keep only references whose own fitted alignment reaches this '
                             'effective depth per site (inclusive); 0 keeps every reference')
    parser.add_argument('--max-neff-per-site', type=float, default=None,
                        help='if set, keep only references strictly below this effective depth '
                             'per site (e.g. 5.0 for the shallow stratum)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.max_neff_per_site is not None and args.max_neff_per_site <= args.min_neff_per_site:
        raise SystemExit('--max-neff-per-site must exceed --min-neff-per-site')

    with gzip.open(args.structures, 'rt', encoding='utf-8') as handle:
        sites = json.load(handle)
    coverage = json.loads(args.coverage.read_text(encoding='utf-8'))
    cohort_rows = {row['assay']: row
                   for row in json.loads(args.cohort.read_text(encoding='utf-8'))['assays']}
    by_assay = collections.defaultdict(list)
    for row in sites:
        by_assay[row['assay_id']].append(row)

    admitted, refused = reference_rows(by_assay, coverage, args.params_root,
                                       cutoff=args.contact_angstrom,
                                       separation=args.min_separation)
    if not admitted:
        raise SystemExit('no reference carries both an admitted structure and a fitted '
                         'coupling model; there is nothing to compare')

    filtered_admitted = {}
    excluded_below_floor: dict[str, float] = {}
    excluded_at_or_above_ceiling: dict[str, float] = {}
    depth_by_assay: dict[str, float] = {}
    depth_filtering = args.min_neff_per_site > 0. or args.max_neff_per_site is not None
    for assay, record in sorted(admitted.items()):
        work = Path(record['params']).parent
        weights_file = work / 'full.weights'
        if depth_filtering or weights_file.exists():
            depth = effective_depth_per_site(work, record['length'])
            depth_by_assay[assay] = depth
            if depth < args.min_neff_per_site:
                excluded_below_floor[assay] = depth
                continue
            if args.max_neff_per_site is not None and depth >= args.max_neff_per_site:
                excluded_at_or_above_ceiling[assay] = depth
                continue
        filtered_admitted[assay] = record
    admitted = filtered_admitted
    if not admitted:
        raise SystemExit('no reference remains after the alignment-depth filter; '
                         'there is nothing to compare')

    arms = []
    for directory in args.extraction:
        index = json.loads((directory / 'position_likelihood.json').read_text(encoding='utf-8'))
        paradigm = index['identity']['paradigm']
        if paradigm not in PARADIGM_SIDES:
            raise SystemExit(f'{directory.name}: unknown paradigm {paradigm!r}; which side of a '
                             'substitution this arm can carry a response on is not declared')
        arms.append({'arm': directory.name, 'directory': directory, 'paradigm': paradigm,
                     'files': {entry['assay']: entry for entry in index['assays']},
                     'index_sha256': sha256_file(directory / 'position_likelihood.json')})

    columns = [{'readout': 'couplings_apc_frobenius', 'kind': 'explicit'},
               {'readout': 'negative_sequence_separation', 'kind': 'design'}]
    for arm in arms:
        columns.append({'readout': f'propagation:{arm["arm"]}', 'kind': 'model',
                        'paradigm': arm['paradigm']})

    table, support = {}, {}
    for assay, record in sorted(admitted.items()):
        keys, label, potts, gap = pair_coupling_scores(record)
        row = {'couplings_apc_frobenius': auc_minus_half(potts, label),
               'negative_sequence_separation': auc_minus_half(gap, label)}
        covered = {}
        for arm in arms:
            name = f'propagation:{arm["arm"]}'
            entry = arm['files'].get(assay)
            archive = None if entry is None else arm['directory'] / 'archives' / Path(entry['file']).name
            if archive is None or not archive.exists():
                row[name] = None
                continue
            measured = arm_pair_responses(archive, record['geometry'], record['pairs'],
                                          cohort_rows[assay], assay=assay,
                                          paradigm=arm['paradigm'])
            seen = [position for position, key in enumerate(keys) if key in measured['values']]
            covered[arm['arm']] = {'pairs': len(seen),
                                   'admitted_variants': measured['admitted_variants']}
            if len(seen) < 2:
                row[name] = None
                continue
            scores = np.array([measured['values'][keys[position]] for position in seen])
            row[name] = auc_minus_half(scores, label[seen])
        table[assay] = row
        support_entry = {'family': record['family'], 'entry': record['entry'],
                         'length': record['length'], 'pairs': len(keys),
                         'contacts': int(label.sum()),
                         'admitted_coordinates': len(record['geometry']),
                         'params': record['params'], 'params_sha256': record['params_sha256'],
                         'arm_coverage': covered}
        if assay in depth_by_assay:
            support_entry['effective_depth_per_site'] = depth_by_assay[assay]
        support[assay] = support_entry

    def summarise_columns(selected_columns: list[dict]) -> tuple[list[dict], dict, list[str]]:
        names = [column['readout'] for column in selected_columns]
        complete_refs = [assay for assay in sorted(table)
                         if all(table[assay][name] is not None for name in names)]
        out_columns = []
        if len(complete_refs) >= MINIMUM_BOOTSTRAP_UNITS:
            matrix = np.array([[table[assay][name] for name in names] for assay in complete_refs])
            bands = paired_bands_with_marginal(matrix)
            verdicts = bidir.verdict(bands['interval'])
            marginal_verdicts = bidir.verdict(bands['marginal_interval'])
            for position, column in enumerate(selected_columns):
                out_columns.append({
                    **column,
                    'point': bands['point'][position],
                    'interval': bands['interval'][position],
                    'verdict': verdicts[position],
                    'marginal_interval': bands['marginal_interval'][position],
                    'marginal_verdict': marginal_verdicts[position],
                })
            record_band = {key: value for key, value in bands.items()
                           if key not in ('point', 'interval', 'marginal_interval')}
            record_band.update({
                'units': 'rank-sum AUC minus one half, dimensionless',
                'resampling_unit': 'biological family; one reference per family here',
                'references': complete_refs,
            })
        else:
            record_band = {'refused': (
                f'{len(complete_refs)} references carry every column against the '
                f'{MINIMUM_BOOTSTRAP_UNITS}-unit resampling floor; a nominal interval over so few '
                'units does not realise its stated coverage and is not formed')}
            for column in selected_columns:
                out_columns.append({
                    **column, 'point': None, 'interval': None, 'verdict': 'unresolved',
                    'marginal_interval': None, 'marginal_verdict': 'unresolved',
                })
        return out_columns, record_band, complete_refs

    coverage_by_column = {column['readout']: sum(
        1 for assay in table if table[assay][column['readout']] is not None)
        for column in columns}
    admitted_columns, refused_columns = [], []
    for column in columns:
        covered = coverage_by_column[column['readout']]
        if covered >= MINIMUM_BOOTSTRAP_UNITS:
            admitted_columns.append(column)
        else:
            refused_columns.append({**column, 'references_covered': covered, 'reason': (
                f'{covered} references carry this readout against the '
                f'{MINIMUM_BOOTSTRAP_UNITS}-unit resampling floor; the column is named rather '
                'than estimated, and it is not allowed to thin the complete case of the others')})
    columns = admitted_columns
    payload_columns, band_record, complete = summarise_columns(columns)

    total_eligible_pairs = sum(entry['pairs'] for entry in support.values())
    pair_fraction_by_arm = {
        arm['arm']: (
            sum(support[assay]['arm_coverage'].get(arm['arm'], {}).get('pairs', 0)
                for assay in support) / total_eligible_pairs
            if total_eligible_pairs else 0.0
        )
        for arm in arms
    }
    full_alpha_columns, thin_bpe_excluded = [], []
    for column in columns:
        if column['kind'] != 'model':
            full_alpha_columns.append(column)
            continue
        arm_name = column['readout'].split('propagation:', 1)[1]
        fraction = pair_fraction_by_arm.get(arm_name, 0.0)
        if fraction >= FULL_ALPHABET_MIN_PAIR_FRACTION:
            full_alpha_columns.append(column)
        else:
            thin_bpe_excluded.append({'arm': arm_name, 'pair_coverage_fraction': fraction})
    fa_columns, fa_band, fa_complete = summarise_columns(full_alpha_columns)
    full_alphabet_secondary = {
        'rule': (f'admitted non-model columns plus model arms covering at least '
                 f'{FULL_ALPHABET_MIN_PAIR_FRACTION:.2f} of eligible residue pairs across '
                 'admitted references, banded on their own complete-case intersection'),
        'min_pair_fraction': FULL_ALPHABET_MIN_PAIR_FRACTION,
        'pair_coverage_fraction_by_arm': pair_fraction_by_arm,
        'thin_bpe_arms_excluded': thin_bpe_excluded,
        'references_complete': fa_complete,
        'columns': fa_columns,
        'native': fa_band,
    }

    write_json(args.out, {
        'schema': 'contact_discrimination_v1',
        'experiment': 'R2/R3 contact localisation by explicit couplings and by propagation',
        'question': ('whether a residue pair can be identified as a three-dimensional contact '
                     'from the fitted Potts coupling norm, from a model\'s per-pair propagation '
                     'response, or from sequence separation alone'),
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'contact_definition': {'cutoff_angstrom': float(args.contact_angstrom),
                               'min_separation': int(args.min_separation),
                               'rule': ('C-beta, or C-alpha for glycine, strictly below the '
                                        'cutoff; a position without an admitted coordinate is '
                                        'in neither class')},
        'statistic': {'per_reference': 'rank-sum AUC minus one half, mid-ranks on ties',
                      'pair_reduction': PAIR_REDUCTION,
                      'chance_reading': CHANCE_READING,
                      'gauge': ('zero-sum, through src.capability.context.potts_gauge.to_zero_sum, '
                                'because a Frobenius norm is not gauge-invariant'),
                      'causal_sides': PARADIGM_SIDES},
        'columns': payload_columns,
        'columns_refused': refused_columns,
        'column_coverage': coverage_by_column,
        'native': band_record,
        'full_alphabet_secondary': full_alphabet_secondary,
        'depth': {
            'min_neff_per_site': float(args.min_neff_per_site),
            'max_neff_per_site': None if args.max_neff_per_site is None else float(args.max_neff_per_site),
            'source': 'the full.weights file each selected.params fit consumed',
            'references_below_floor': excluded_below_floor,
            'references_at_or_above_ceiling': excluded_at_or_above_ceiling,
            'realised': {assay: depth_by_assay[assay] for assay in complete if assay in depth_by_assay},
        },
        'families': sorted({record['family'] for record in admitted.values()}),
        'references_admitted': len(admitted),
        'references_complete': complete,
        'references_refused': refused,
        'support': support,
        'per_reference': table,
        'sources': {
            'cohort': {'path': str(args.cohort), 'sha256': sha256_file(args.cohort)},
            'params_roots': [str(root) for root in args.params_root],
            'structures': {'path': str(args.structures), 'sha256': sha256_file(args.structures)},
            'coverage': {'path': str(args.coverage), 'sha256': sha256_file(args.coverage)},
            'extractions': [{'arm': arm['arm'], 'directory': str(arm['directory']),
                             'index_sha256': arm['index_sha256'],
                             'paradigm': arm['paradigm']} for arm in arms],
        },
        'bounds': (
            'every reference is a 37-to-72-residue domain, so no contact here has sequence '
            'separation above 72 and nothing is said about long-range contacts in large '
            'proteins. Realised alignment depth per site (Neff/L) for each fitted Potts '
            'model is recorded under depth and support, and all pair lookups use 0-based '
            'wild-type residue indices matching structure_sites() and contact_pairs()'),
    })


if __name__ == '__main__':
    main()
