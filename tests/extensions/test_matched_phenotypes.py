"""Matching logic, matched-pair counting, and the cross-phenotype quantities.

The conditions asserted here are the ones that decide whether a cross-phenotype
claim is about phenotypes at all: a pair is matched only when one exact wild-type
sequence and one substitution carry two labels of *different* adjudicated
phenotype classes; a same-class pair is a replication question and is excluded;
two constructs of one gene are not matched; and every per-pair count carries its
own independent-group floor record, because the pooled protein-unit reading is
the only one the floor permits.
"""
import csv
import random
from pathlib import Path

import numpy as np
import pytest

from src.capability.core.amino_acids import AA20
from src.capability.extensions import matched_phenotypes as matched
from src.capability.extensions import phenotype_breadth as breadth

SEED = 20261008
SUBSTITUTIONS = 40
NARROW_QUALIFIED = ('geom',)


def wildtype(name: str, length: int = 120) -> str:
    rng = random.Random(f'{SEED}:{name}')
    return ''.join(rng.choices(AA20, k=length))


def mutations(wt: str, count: int = SUBSTITUTIONS) -> list[tuple[str, str]]:
    rng = random.Random(f'{SEED}:mut')
    out = []
    for index in range(count):
        position = index + 1
        residue = rng.choice([a for a in AA20 if a != wt[position - 1]])
        out.append((f'{wt[position - 1]}{position}{residue}',
                    wt[:position - 1] + residue + wt[position:]))
    return out


def write_assay(path: Path, pairs, labels) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['mutant', 'mutated_sequence', 'DMS_score'])
        for (mutant, sequence), value in zip(pairs, labels):
            writer.writerow([mutant, sequence, f'{value:.8f}'])


def write_reference(path: Path, rows) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matched.REFERENCE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_strata(path: Path, rows) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['assay', 'cluster', 'category'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def reference_row(assay: str, protein: str, sequence: str) -> dict:
    return {'DMS_id': assay, 'UniProt_ID': protein, 'target_seq': sequence,
            'selection_assay': f'{assay} selection', 'selection_type': 'FACS',
            'raw_DMS_phenotype_name': 'phenotype', 'raw_DMS_directionality': '1'}


@pytest.fixture
def world(tmp_path):
    """Four proteins: two simple cross-class, one with a same-class pair, one with
    two different constructs of the same gene."""
    assays = tmp_path / 'assays'
    assays.mkdir()
    reference, strata = [], []
    rng = random.Random(f'{SEED}:labels')
    plan = [
        ('P1', [('P1_act', 'Activity'), ('P1_abd', 'Expression')]),
        ('P2', [('P2_act', 'Activity'), ('P2_abd', 'Expression')]),
        # Three assays, two of them the same class: the same-class pair must be
        # excluded while both cross-class pairs are retained.
        ('P3', [('P3_a', 'Activity'), ('P3_b', 'Activity'), ('P3_c', 'Expression')]),
    ]
    for protein, members in plan:
        sequence = wildtype(protein)
        pairs = mutations(sequence)
        for index, (assay, category) in enumerate(members):
            labels = [rng.gauss(0.0, 1.0) for _ in pairs]
            write_assay(assays / f'{assay}.csv', pairs, labels)
            reference.append(reference_row(assay, protein, sequence))
            strata.append({'assay': assay, 'cluster': f'c{protein}', 'category': category})
    # One gene measured on two different constructs: not one matched support.
    for suffix, category in (('long', 'Activity'), ('short', 'Expression')):
        sequence = wildtype(f'P4-{suffix}', 120 if suffix == 'long' else 110)
        pairs = mutations(sequence)
        write_assay(assays / f'P4_{suffix}.csv', pairs, [rng.gauss(0, 1) for _ in pairs])
        reference.append(reference_row(f'P4_{suffix}', 'P4', sequence))
        strata.append({'assay': f'P4_{suffix}', 'cluster': 'cP4', 'category': category})
    write_reference(tmp_path / 'reference.csv', reference)
    write_strata(tmp_path / 'strata.csv', strata)
    return {'assays': assays,
            'reference': matched.read_reference(tmp_path / 'reference.csv'),
            'strata': matched.read_strata_metadata(tmp_path / 'strata.csv'),
            'root': tmp_path}


# --------------------------------------------------------------------------- #
# What counts as matched
# --------------------------------------------------------------------------- #

def test_only_cross_class_pairs_on_one_exact_wildtype_are_matched(world):
    pairs, census = matched.discover_pairs(world['assays'], world['reference'], world['strata'])
    assert sorted(pair.protein for pair in pairs) == ['P1', 'P2', 'P3', 'P3']
    assert census['same_class_pair_excluded'] == 1
    # Two constructs of one gene never form a matched pair, however similar.
    assert census['sequence_without_cross_class_pair'] == 2
    for pair in pairs:
        assert pair.class_a != pair.class_b
        assert pair.classes == ('Activity', 'Expression')
        assert len(pair.mutations) == SUBSTITUTIONS
        assert pair.record()['wildtype_sha256'] == breadth.sha_text(pair.wildtype)


def test_a_mutation_with_two_different_mutant_sequences_raises(world):
    rows = list(csv.DictReader((world['assays'] / 'P1_abd.csv').open()))
    rows[0]['mutated_sequence'] = 'A' + rows[0]['mutated_sequence'][1:]
    with (world['assays'] / 'P1_abd.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['mutant', 'mutated_sequence', 'DMS_score'])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='two mutant sequences'):
        matched.discover_pairs(world['assays'], world['reference'], world['strata'])


def test_a_pair_below_the_minimum_matched_rows_is_not_a_pair(world):
    short = list(csv.DictReader((world['assays'] / 'P2_abd.csv').open()))[:2]
    with (world['assays'] / 'P2_abd.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['mutant', 'mutated_sequence', 'DMS_score'])
        writer.writeheader()
        writer.writerows(short)
    pairs, census = matched.discover_pairs(world['assays'], world['reference'], world['strata'])
    assert [pair.protein for pair in pairs] == ['P1', 'P3', 'P3']
    assert census['pair_below_minimum_matched_rows'] == 1


def test_multiple_substitutions_and_nonfinite_scores_never_enter(world):
    path = world['assays'] / 'P1_act.csv'
    rows = list(csv.DictReader(path.open()))
    wt = world['reference']['P1_act']['target_seq']
    rows.append({'mutant': 'A1C:A2D', 'mutated_sequence': wt, 'DMS_score': '0.5'})
    rows.append({'mutant': f'{wt[99]}100{"W" if wt[99] != "W" else "Y"}',
                 'mutated_sequence': wt, 'DMS_score': 'nan'})
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['mutant', 'mutated_sequence', 'DMS_score'])
        writer.writeheader()
        writer.writerows(rows)
    table = matched.read_assay(path)
    assert ':' not in ''.join(table)
    assert len(table) == SUBSTITUTIONS
    assert all(np.isfinite(value) for value, _ in table.values())


# --------------------------------------------------------------------------- #
# Counting, and the floor
# --------------------------------------------------------------------------- #

def test_every_phenotype_pair_carries_its_own_floor_record(world):
    pairs, _ = matched.discover_pairs(world['assays'], world['reference'], world['strata'])
    counts = matched.pair_counts(pairs)
    assert len(counts['by_phenotype_pair']) == 1
    record = counts['by_phenotype_pair'][0]
    assert record['phenotype_pair'] == ['Activity', 'Expression']
    assert record['assay_pairs'] == 4
    assert record['proteins'] == 3 and record['independent_groups'] == 3
    assert record['matched_substitution_rows'] == 4 * SUBSTITUTIONS
    assert record['independence_floor']['degenerate'] is True
    assert 'below the independent-group floor' in record['inference']
    assert counts['pooled']['independent_groups'] == 3
    assert counts['pooled']['independence_floor']['degenerate'] is True


def test_matched_rows_are_aligned_identities_with_two_labels(world):
    pairs, _ = matched.discover_pairs(world['assays'], world['reference'], world['strata'])
    pair = pairs[0]
    rows_a, rows_b, draw = matched.matched_rows(pair, world['assays'])
    assert len(rows_a) == len(rows_b) == SUBSTITUTIONS
    for left, right in zip(rows_a, rows_b):
        assert left.mutant_sequence == right.mutant_sequence
        assert (left.position, left.wt_aa, left.mt_aa) == (right.position, right.wt_aa, right.mt_aa)
        assert left.background != right.background
        assert left.label != right.label
        breadth.validate_row(left)
        breadth.validate_row(right)
    assert draw['retained_substitutions'] == SUBSTITUTIONS


def test_the_matched_cap_is_label_blind_and_identical_for_both_phenotypes(world):
    pairs, _ = matched.discover_pairs(world['assays'], world['reference'], world['strata'])
    pair = pairs[0]
    rows_a, rows_b, draw = matched.matched_rows(pair, world['assays'], cap=9)
    assert draw['cap'] == 9 and len(rows_a) == len(rows_b) == 9
    assert [row.row_id.split(':', 1)[1] for row in rows_a] == \
           [row.row_id.split(':', 1)[1] for row in rows_b]
    again, _, _ = matched.matched_rows(pair, world['assays'], cap=9)
    assert [row.row_id for row in rows_a] == [row.row_id for row in again]


def test_label_sharing_is_computed_without_any_model_quantity(world):
    pairs, _ = matched.discover_pairs(world['assays'], world['reference'], world['strata'])
    sharing = matched.label_sharing(pairs, world['assays'])
    assert len(sharing) == 4
    for record in sharing:
        assert -1.0 <= record['label_spearman'] <= 1.0
        assert record['retained_substitutions'] == SUBSTITUTIONS
        assert 'arm' not in record


# --------------------------------------------------------------------------- #
# The cross-phenotype quantities
# --------------------------------------------------------------------------- #

def scores_for(pair, assay_dir, *, carries: str) -> dict[str, float]:
    """A likelihood table whose difference reproduces one phenotype's label."""
    rows_a, rows_b, _ = matched.matched_rows(pair, assay_dir)
    chosen = rows_a if carries == 'a' else rows_b
    table = {breadth.sha_text(pair.wildtype): 0.0}
    for row in chosen:
        table[breadth.sha_text(row.mutant_sequence)] = row.oriented_label()
    return table


def test_a_likelihood_aligned_with_one_phenotype_shows_a_signed_difference(world):
    pairs, _ = matched.discover_pairs(world['assays'], world['reference'], world['strata'])
    pair = pairs[0]
    scores = {'carries_a': scores_for(pair, world['assays'], carries='a'),
              'carries_b': scores_for(pair, world['assays'], carries='b')}
    cells = matched.pair_increments(
        pair, world['assays'], lambda rows: breadth.control_blocks(rows, None),
        scores, NARROW_QUALIFIED)
    by_arm = {cell['arm']: cell for cell in cells}
    assert set(by_arm) == {'carries_a', 'carries_b'}
    # The same score cannot favour both phenotypes: the signed difference of the
    # two increments has opposite signs for the two arms.
    assert by_arm['carries_a']['increment_difference'] > 0
    assert by_arm['carries_b']['increment_difference'] < 0
    for cell in cells:
        assert cell['increment_a'] == pytest.approx(cell['augmented_a'] - cell['control_a'])
        assert cell['increment_b'] == pytest.approx(cell['augmented_b'] - cell['control_b'])
        assert cell['increment_mean'] == pytest.approx(
            0.5 * (cell['increment_a'] + cell['increment_b']))
        assert -2.0 <= cell['shared_residual_change'] <= 2.0
        assert cell['retained_substitutions'] == SUBSTITUTIONS


def test_pooled_inference_treats_a_protein_as_one_unit(world):
    pairs, _ = matched.discover_pairs(world['assays'], world['reference'], world['strata'])
    cells = []
    for pair in pairs:
        scores = {'arm': scores_for(pair, world['assays'], carries='a')}
        cells.extend(matched.pair_increments(
            pair, world['assays'], lambda rows: breadth.control_blocks(rows, None),
            scores, NARROW_QUALIFIED))
    # P3 contributes two assay pairs and must still be one bootstrap unit.
    assert sum(1 for cell in cells if cell['group'] == 'cP3') == 2
    pooled = matched.pooled_inference(cells, ['arm'])
    assert pooled['groups'] == ['cP1', 'cP2', 'cP3']
    assert pooled['contrast_family'] == ['increment_mean', 'increment_difference',
                                         'shared_residual_change']
    assert len(pooled['inference']['point']) == 3
    assert pooled['inference']['groups'] == 3
    assert pooled['inference']['independence_floor']['degenerate'] is True
    # The point estimate is the mean over proteins of each protein's own mean.
    for index, contrast in enumerate(pooled['contrast_family']):
        expected = np.mean([np.mean([cell[contrast] for cell in cells if cell['group'] == group])
                            for group in pooled['groups']])
        assert pooled['inference']['point'][index] == pytest.approx(expected)


def test_a_pooled_cell_missing_an_arm_is_refused_not_imputed(world):
    pairs, _ = matched.discover_pairs(world['assays'], world['reference'], world['strata'])
    pair = pairs[0]
    scores = {'arm': scores_for(pair, world['assays'], carries='a')}
    cells = matched.pair_increments(
        pair, world['assays'], lambda rows: breadth.control_blocks(rows, None),
        scores, NARROW_QUALIFIED)
    with pytest.raises(ValueError, match='no matched cell'):
        matched.pooled_inference(cells, ['arm', 'absent_arm'])
