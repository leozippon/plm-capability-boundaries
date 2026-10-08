"""Matched multi-phenotype support: the same substitution, two phenotypes.

A cross-phenotype question cannot be answered by comparing two cohorts that
differ in their proteins as well as their phenotypes. This module builds the
support where the confound is absent: one exact wild-type sequence, one
substitution, and two experimental endpoints of *different* adjudicated
phenotype classes measured on it.

Why that support exists at all. The anchor's own 201 assays include thirteen
proteins whose authors published two or three assays of different phenotype
classes on the identical construct -- abundance beside activity, surface
expression beside ion conduction, abundance beside binding. Matching those
tables gives genuinely matched rows without acquiring anything new.

What the matching does and does not buy. It removes the sequence and support
confound, so a difference between two phenotypes is a difference between
phenotypes. It does not create independent proteins: the thirteen proteins are
anchor proteins, so this is matched multi-phenotype support (evidence level 2 of
the phenotype programme), never an independent task cohort. Counts are reported
per phenotype pair precisely because no single pair reaches the
independent-group floor; the pooled reading over all thirteen proteins is the
only one the floor permits.

The scientific content is sharing. The model's likelihood difference for a given
substitution is one number, identical for every phenotype measured on it, so the
model cannot be "better at abundance" through a different score -- only through a
different alignment with each label. Three quantities separate those cases on
identical rows: how correlated the two labels are, how the likelihood increment
over a qualified control differs between the two phenotypes, and whether adding
the likelihood reduces the *shared* part of the baseline's residual.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata

from ..context.profile_increment import correlation
from ..core.statistics import bootstrap_unit_floor
from . import phenotype_breadth as breadth
from .overlap import rerank

SCHEMA = 'matched_phenotype_v1'

#: Columns this module reads from each processed ProteinGym substitution table
#: and from the release reference. Nothing else is read.
ASSAY_COLUMNS = ('mutant', 'mutated_sequence', 'DMS_score')
REFERENCE_COLUMNS = ('DMS_id', 'UniProt_ID', 'target_seq', 'selection_assay', 'selection_type',
                     'raw_DMS_phenotype_name', 'raw_DMS_directionality')

#: Orientation convention of the processed tables, recorded rather than assumed
#: silently: the release applies ``raw_DMS_directionality`` to the raw phenotype,
#: so a larger processed score is more of the measured function in every assay.
#: The raw directionality of each assay travels in the registry so a reader can
#: check the orientation of the quantity that was actually measured.
ORIENTATION = ('processed DMS_score with the release directionality already applied; larger is '
               'more of the measured function')

LIMITATIONS = [
    'The phenotype class of an assay is the coarse adjudicated stratum of the existing strata '
    'analysis, not an independently verified mechanism; two assays in one class may measure '
    'different quantities and one assay may report a mixed quantity, so the per-assay selection '
    'description travels with every pair.',
    'The thirteen matched proteins are anchor proteins. This is matched multi-phenotype support, '
    'not an independent confirmation cohort, and the model was never trained on these labels but '
    'the anchor panel was developed on assays of these same proteins.',
    'No phenotype pair reaches the independent-group floor on its own. The pooled protein-unit '
    'reading is the only inference; every per-pair number carries its own degenerate-floor record '
    'and is descriptive.',
    'Labels of two phenotypes are in different units. Their shared structure is reported as rank '
    'correlation only, and no squared-error metric is pooled across phenotypes.',
    'Residual sharing is computed from fitted out-of-fold predictions and conditions on them; it '
    'carries no training or tuning uncertainty.',
]


@dataclass(frozen=True)
class MatchedPair:
    """Two assays of different phenotype classes on one exact wild-type sequence."""

    protein: str
    group: str
    wildtype: str
    assay_a: str
    assay_b: str
    class_a: str
    class_b: str
    description_a: str
    description_b: str
    directionality_a: int
    directionality_b: int
    mutations: tuple[str, ...]

    @property
    def classes(self) -> tuple[str, str]:
        return tuple(sorted((self.class_a, self.class_b)))

    def record(self) -> dict[str, Any]:
        return {'protein': self.protein, 'group': self.group, 'wildtype_length': len(self.wildtype),
                'wildtype_sha256': breadth.sha_text(self.wildtype),
                'assay_a': self.assay_a, 'assay_b': self.assay_b,
                'class_a': self.class_a, 'class_b': self.class_b,
                'phenotype_pair': list(self.classes),
                'selection_a': self.description_a, 'selection_b': self.description_b,
                'raw_directionality_a': self.directionality_a,
                'raw_directionality_b': self.directionality_b,
                'matched_substitutions': len(self.mutations)}


def read_reference(path: Path) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(Path(path).open(encoding='utf-8')))
    missing = [c for c in REFERENCE_COLUMNS if rows and c not in rows[0]]
    if missing:
        raise ValueError(f'release reference is missing declared columns: {missing}')
    return {row['DMS_id']: row for row in rows}


def read_strata_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Assay to adjudicated phenotype class and frozen family cluster."""
    rows = list(csv.DictReader(Path(path).open(encoding='utf-8')))
    for column in ('assay', 'category', 'cluster'):
        if rows and column not in rows[0]:
            raise ValueError(f'strata metadata is missing the {column} column')
    return {row['assay']: row for row in rows}


def read_assay(path: Path) -> dict[str, tuple[float, str]]:
    """Single substitutions of one processed table: mutation to (score, sequence)."""
    out: dict[str, tuple[float, str]] = {}
    with Path(path).open(encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ASSAY_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f'{path} is missing declared columns: {missing}')
        for row in reader:
            mutation = row['mutant']
            if ':' in mutation:
                continue
            score = float(row['DMS_score'])
            if not np.isfinite(score):
                continue
            if mutation in out:
                raise ValueError(f'{path}: duplicate single substitution {mutation}')
            out[mutation] = (score, row['mutated_sequence'])
    return out


def discover_pairs(assay_dir: Path, reference: dict[str, dict[str, str]],
                   strata: dict[str, dict[str, str]]) -> tuple[list[MatchedPair], dict[str, Any]]:
    """Every cross-class assay pair that shares one exact wild-type sequence.

    Grouping is by the release's exact ``target_seq``, so two assays on different
    constructs of the same gene are not matched. Same-class pairs are excluded by
    construction: they answer a replication question, not a cross-phenotype one.
    """

    by_sequence: dict[str, list[str]] = defaultdict(list)
    for assay in sorted(strata):
        row = reference.get(assay)
        if row is None:
            raise ValueError(f'{assay} is not in the release reference')
        by_sequence[row['target_seq']].append(assay)
    pairs: list[MatchedPair] = []
    census: Counter[str] = Counter()
    for sequence, assays in sorted(by_sequence.items(), key=lambda item: item[1][0]):
        classes = {strata[a]['category'] for a in assays}
        if len(assays) < 2 or len(classes) < 2:
            census['sequence_without_cross_class_pair'] += 1
            continue
        tables = {a: read_assay(assay_dir / f'{a}.csv') for a in assays}
        for index, left in enumerate(assays):
            for right in assays[index + 1:]:
                if strata[left]['category'] == strata[right]['category']:
                    census['same_class_pair_excluded'] += 1
                    continue
                shared = sorted(set(tables[left]) & set(tables[right]))
                checked = []
                for mutation in shared:
                    if tables[left][mutation][1] != tables[right][mutation][1]:
                        raise ValueError(f'{left}/{right}: {mutation} has two mutant sequences')
                    checked.append(mutation)
                if len(checked) < breadth.MIN_BACKGROUND_ROWS:
                    census['pair_below_minimum_matched_rows'] += 1
                    continue
                clusters = {strata[left]['cluster'], strata[right]['cluster']}
                if len(clusters) != 1:
                    raise ValueError(f'{left}/{right}: one construct spans two frozen clusters')
                pairs.append(MatchedPair(
                    protein=reference[left]['UniProt_ID'], group=clusters.pop(), wildtype=sequence,
                    assay_a=left, assay_b=right,
                    class_a=strata[left]['category'], class_b=strata[right]['category'],
                    description_a=reference[left]['selection_assay'],
                    description_b=reference[right]['selection_assay'],
                    directionality_a=int(reference[left]['raw_DMS_directionality']),
                    directionality_b=int(reference[right]['raw_DMS_directionality']),
                    mutations=tuple(checked)))
                census['cross_class_pair_retained'] += 1
    return pairs, {k: int(v) for k, v in census.items()}


def pair_counts(pairs: Sequence[MatchedPair]) -> dict[str, Any]:
    """Matched-pair counts per phenotype pair, with each pair's floor record."""
    by_classes: dict[tuple[str, str], list[MatchedPair]] = defaultdict(list)
    for pair in pairs:
        by_classes[pair.classes].append(pair)
    records = []
    for classes in sorted(by_classes):
        members = by_classes[classes]
        groups = sorted({p.group for p in members})
        records.append({
            'phenotype_pair': list(classes), 'assay_pairs': len(members),
            'proteins': len({p.protein for p in members}),
            'independent_groups': len(groups), 'groups': groups,
            'matched_substitution_rows': sum(len(p.mutations) for p in members),
            'independence_floor': bootstrap_unit_floor(len(groups)),
            'inference': ('descriptive: the pair is below the independent-group floor'
                          if len(groups) < breadth.INDEPENDENCE_FLOOR else
                          'the pair reaches the independent-group floor')})
    pooled_groups = sorted({p.group for p in pairs})
    return {'by_phenotype_pair': records,
            'pooled': {'assay_pairs': len(pairs), 'proteins': len({p.protein for p in pairs}),
                       'independent_groups': len(pooled_groups), 'groups': pooled_groups,
                       'matched_substitution_rows': sum(len(p.mutations) for p in pairs),
                       'independence_floor': bootstrap_unit_floor(len(pooled_groups))}}


def matched_rows(pair: MatchedPair, assay_dir: Path, *, cohort: str = 'matched_proteingym',
                 cap: int = breadth.VARIANT_CAP) -> tuple[list[breadth.PhenotypeRow],
                                                          list[breadth.PhenotypeRow], dict]:
    """Two aligned row lists for one matched pair: identical identities, two labels.

    The retained substitutions are capped label-blind by the same stable-hash
    draw the breadth cohorts use, so the matched support is fixed before any
    label or model output is read, and both phenotypes get exactly the same rows.
    """

    left = read_assay(assay_dir / f'{pair.assay_a}.csv')
    right = read_assay(assay_dir / f'{pair.assay_b}.csv')
    ordered = sorted(pair.mutations,
                     key=lambda m: (breadth.stable_draw_key(cohort, pair.protein, m), m))
    retained = sorted(ordered[:cap])
    rows_a, rows_b = [], []
    for mutation in retained:
        wt_aa, position, mt_aa = mutation[0], int(mutation[1:-1]), mutation[-1]
        sequence = left[mutation][1]
        for assay, table, bucket in ((pair.assay_a, left, rows_a), (pair.assay_b, right, rows_b)):
            bucket.append(breadth.PhenotypeRow(
                cohort=cohort, row_id=f'{assay}:{mutation}', background=assay,
                unit=breadth.sha_text(pair.wildtype), wildtype=pair.wildtype,
                mutant_sequence=sequence, position=position, wt_aa=wt_aa, mt_aa=mt_aa,
                label=table[mutation][0], direction=1, label_unit=None,
                source_sha256=breadth.sha_text(f'{assay}|{mutation}')))
    return rows_a, rows_b, {'matched_substitutions': len(pair.mutations),
                            'retained_substitutions': len(retained), 'cap': cap,
                            'draw_seed': breadth.DRAW_SEED}


def label_sharing(pairs: Sequence[MatchedPair], assay_dir: Path) -> list[dict[str, Any]]:
    """Per matched pair, the rank correlation of the two phenotype labels."""
    out = []
    for pair in pairs:
        rows_a, rows_b, _ = matched_rows(pair, assay_dir)
        left = np.asarray([r.label for r in rows_a], dtype=np.float64)
        right = np.asarray([r.label for r in rows_b], dtype=np.float64)
        out.append({**pair.record(), 'retained_substitutions': len(left),
                    'label_spearman': correlation(rankdata(left), rankdata(right))})
    return out


def residual(prediction: np.ndarray, target: np.ndarray, backgrounds: np.ndarray) -> np.ndarray:
    """Within-background rank residual of a fitted out-of-fold prediction."""
    return rerank(np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64),
                  backgrounds)


def pair_increments(pair: MatchedPair, assay_dir: Path, blocks_for,
                    scores: dict[str, dict[str, float]], qualified: Sequence[str],
                    *, seeds: Sequence[int] = breadth.SPLIT_SEEDS) -> list[dict[str, Any]]:
    """Per arm, the two phenotypes' ranking increments on identical matched rows.

    Both phenotypes use the same rows, the same control design and the same
    held-out partitions, so the difference between their increments is a
    phenotype difference and nothing else.

    The partition is over the *mutated positions* of this one protein, because one
    protein is one independent group and a protein-held-out partition inside a
    single protein would be empty. A held-out substitution's own position never
    appears in training, so the increment generalises across sites of a known
    protein and must not be read as the cross-protein increment the anchor panel
    reports. Rows are weighted so that each mutated position counts equally,
    which keeps a few saturated positions from carrying a protein. The protein is
    restored as the unit of inference by the pooled bootstrap over proteins.
    """

    rows_a, rows_b, draw = matched_rows(pair, assay_dir)
    blocks = blocks_for(rows_a)
    sites = np.asarray([f'site{row.position}' for row in rows_a])
    backgrounds = np.asarray([pair.protein] * len(rows_a))
    out = []
    for arm in sorted(scores):
        model = breadth.model_block(rows_a, scores[arm], backgrounds)
        arm_blocks = {**blocks, 'M': model['M_rank']}
        cells: dict[str, dict[str, Any]] = {}
        residuals: dict[str, dict[str, np.ndarray]] = {}
        for label, rows in (('a', rows_a), ('b', rows_b)):
            labels = np.asarray([r.oriented_label() for r in rows], dtype=np.float64)
            target = rerank(labels, backgrounds)
            increments, control_scores, augmented_scores = [], [], []
            control_residuals, augmented_residuals = [], []
            for seed in seeds:
                signature = breadth.outer_signature(sites, seed)
                control, _ = breadth.fold_predictions(
                    breadth.design(arm_blocks, qualified), target, sites, sites, signature)
                augmented, _ = breadth.fold_predictions(
                    breadth.design(arm_blocks, [*qualified, 'M']), target, sites,
                    sites, signature)
                control_score = correlation(rankdata(control), target)
                augmented_score = correlation(rankdata(augmented), target)
                if control_score is None or augmented_score is None:
                    raise ValueError(f'{pair.protein}/{arm}: undefined matched rank correlation')
                control_scores.append(control_score)
                augmented_scores.append(augmented_score)
                increments.append(augmented_score - control_score)
                control_residuals.append(residual(control, target, backgrounds))
                augmented_residuals.append(residual(augmented, target, backgrounds))
            cells[label] = {'control': float(np.mean(control_scores)),
                            'control_plus_model': float(np.mean(augmented_scores)),
                            'increment': float(np.mean(increments))}
            residuals[label] = {'control': np.mean(control_residuals, axis=0),
                                'control_plus_model': np.mean(augmented_residuals, axis=0)}
        shared = {}
        for design_name in ('control', 'control_plus_model'):
            value = correlation(rankdata(residuals['a'][design_name]),
                                rankdata(residuals['b'][design_name]))
            if value is None:
                raise ValueError(f'{pair.protein}/{arm}: undefined residual correlation')
            shared[design_name] = value
        out.append({**pair.record(), 'arm': arm, 'draw': draw,
                    'retained_substitutions': len(rows_a),
                    'increment_a': cells['a']['increment'], 'increment_b': cells['b']['increment'],
                    'control_a': cells['a']['control'], 'control_b': cells['b']['control'],
                    'augmented_a': cells['a']['control_plus_model'],
                    'augmented_b': cells['b']['control_plus_model'],
                    'increment_mean': float(np.mean([cells['a']['increment'], cells['b']['increment']])),
                    'increment_difference': cells['a']['increment'] - cells['b']['increment'],
                    'shared_residual_control': shared['control'],
                    'shared_residual_control_plus_model': shared['control_plus_model'],
                    'shared_residual_change': (shared['control_plus_model'] - shared['control'])})
    return out


def pooled_inference(cells: Sequence[dict[str, Any]], arms: Sequence[str]) -> dict[str, Any]:
    """Protein-unit simultaneous inference over the declared matched contrasts.

    One protein is one unit however many assay pairs it contributes, so a
    protein's pairs are averaged before the protein enters the resample. The
    contrast family is declared here and spans every arm: the mean increment
    across the two phenotypes, the signed difference between them, and the change
    in shared residual when the likelihood is added.
    """

    contrasts = ('increment_mean', 'increment_difference', 'shared_residual_change')
    groups = sorted({cell['group'] for cell in cells})
    keys = [(arm, contrast) for arm in arms for contrast in contrasts]
    matrix = np.empty((len(groups), len(keys)), dtype=np.float64)
    for row, group in enumerate(groups):
        for column, (arm, contrast) in enumerate(keys):
            values = [cell[contrast] for cell in cells
                      if cell['group'] == group and cell['arm'] == arm]
            if not values:
                raise ValueError(f'{group}/{arm}: no matched cell for the pooled contrast')
            matrix[row, column] = float(np.mean(values))
    band = breadth.simultaneous_band(matrix)
    results = []
    for index, (arm, contrast) in enumerate(keys):
        low, high = band['simultaneous'][index]
        point_low, point_high = band['pointwise'][index]
        results.append({'arm': arm, 'contrast': contrast, 'point': band['point'][index],
                        'pointwise': [point_low, point_high], 'simultaneous': [low, high],
                        'resolved_positive': low > 0, 'resolved_negative': high < 0})
    return {'contrast_family': list(contrasts), 'arms': list(arms), 'groups': groups,
            'aggregation': 'assay pairs averaged within protein, equal proteins',
            'contrasts': ({'increment_mean': 'mean ranking increment over the two phenotypes',
                           'increment_difference': ('ranking increment of the first-listed assay '
                                                    'minus that of the second, on identical rows'),
                           'shared_residual_change': ('rank correlation of the two phenotypes\' '
                                                      'residuals with the likelihood minus without '
                                                      'it; negative means the likelihood explained '
                                                      'part of the shared component')}),
            'inference': band, 'results': results}
