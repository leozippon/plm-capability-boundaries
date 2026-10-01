"""Conditions the structure-contact gate must always satisfy.

Four properties are enforced here, each with the failing side as well as the
passing one: the contact definition reads the geometry it declares, the matching
scheme actually balances the confounders it names, a structure whose numbering
does not carry the assayed wild type exactly is refused rather than annotated,
and no measurement reaches the annotation or the matching.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.interactions.contact_enrichment import (  # noqa: E402
    CB_CONTACT_ANGSTROM, HEAVY_ATOM_CONTACT_ANGSTROM, PROBE_RADIUS, VDW_RADII,
    annotate_background, balance_table, cem_design, load_structure, match_chain,
    read_cif, relative_accessibility, residue_accessibility, site_pair_plan,
    resampled_arm_difference, two_stage_bootstrap)
from src.capability.stability.pairwise_stability import COHORT_SCHEMA  # noqa: E402

ATOM_FIELDS = ('group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_alt_id',
               'label_comp_id', 'label_asym_id', 'label_entity_id', 'label_seq_id',
               'pdbx_PDB_ins_code', 'Cartn_x', 'Cartn_y', 'Cartn_z', 'occupancy',
               'B_iso_or_equiv', 'pdbx_formal_charge', 'auth_seq_id', 'auth_comp_id',
               'auth_asym_id', 'auth_atom_id', 'pdbx_PDB_model_num')

THREE = {'A': 'ALA', 'G': 'GLY', 'V': 'VAL', 'L': 'LEU', 'K': 'LYS', 'E': 'GLU'}


def synthetic_cif(sequence: str, atoms: list[dict], *, method: str = 'X-RAY DIFFRACTION',
                  chain: str = 'A') -> str:
    """One mmCIF data block with exactly the geometry a test declares.

    ``atoms`` carries dicts of ``position`` (1-based in ``sequence``), ``name``,
    ``element``, ``xyz`` and optional ``model``, so a test states coordinates
    rather than deriving them.
    """

    lines = ['data_TEST', '#', "_entry.id  TEST", '#', f"_exptl.method  '{method}'", '#',
             '_entity_poly.entity_id  1', "_entity_poly.type  'polypeptide(L)'",
             f'_entity_poly.pdbx_seq_one_letter_code_can  {sequence}',
             f'_entity_poly.pdbx_strand_id  {chain}', '#', 'loop_']
    lines += [f'_atom_site.{field}' for field in ATOM_FIELDS]
    for index, atom in enumerate(atoms, start=1):
        comp = THREE[sequence[atom['position'] - 1]]
        x, y, z = atom['xyz']
        lines.append(' '.join(str(value) for value in (
            'ATOM', index, atom['element'], atom['name'], '.', comp, chain, 1,
            atom['position'], '?', f'{x:.3f}', f'{y:.3f}', f'{z:.3f}', '1.00', '0.00',
            '?', atom['position'], comp, chain, atom['name'], atom.get('model', 1))))
    lines.append('#')
    return '\n'.join(lines) + '\n'


def write_cif(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def backbone(sequence: str, *, spacing: float = 3.8) -> list[dict]:
    return [{'position': index + 1, 'name': 'CA', 'element': 'C',
             'xyz': (spacing * index, 0.0, 0.0)} for index in range(len(sequence))]


class TheContactDefinitionReadsTheDeclaredGeometry(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parent / '_contact_enrichment_tmp'
        self.tmp.mkdir(exist_ok=True)

    def tearDown(self):
        for path in self.tmp.glob('*'):
            path.unlink()
        self.tmp.rmdir()

    def test_the_threshold_is_inclusive_and_is_measured_over_heavy_atoms(self):
        # Five alanines on a line 3.8 A apart, so 1-2 is 3.8 A and 1-3 is 7.6 A.
        sequence = 'AAAAA'
        atoms = backbone(sequence)
        path = write_cif(self.tmp / 'line.cif', synthetic_cif(sequence, atoms))
        entry = {'name': 'line.pdb', 'group': 'g', 'kind': 'natural', 'length': 5,
                 'wildtype': sequence, 'site_pairs': [[1, 2], [1, 3], [1, 5]]}
        pairs = {row['site_pair']: row for row in annotate_background(entry, path)['site_pairs']}
        self.assertAlmostEqual(pairs['line.pdb:1-2']['min_heavy_atom_angstrom'], 3.8, places=6)
        self.assertTrue(pairs['line.pdb:1-2']['contact'])
        self.assertAlmostEqual(pairs['line.pdb:1-3']['min_heavy_atom_angstrom'], 7.6, places=6)
        self.assertFalse(pairs['line.pdb:1-3']['contact'])
        self.assertEqual(pairs['line.pdb:1-5']['separation'], 4)

        # Exactly at the threshold is a contact; one thousandth beyond it is not.
        for distance, expected in ((HEAVY_ATOM_CONTACT_ANGSTROM, True),
                                   (HEAVY_ATOM_CONTACT_ANGSTROM + 0.001, False)):
            moved = [dict(atom) for atom in atoms]
            moved[1]['xyz'] = (distance, 0.0, 0.0)
            path = write_cif(self.tmp / 'edge.cif', synthetic_cif(sequence, moved))
            row = annotate_background(
                {**entry, 'name': 'edge.pdb', 'site_pairs': [[1, 2]]}, path)['site_pairs'][0]
            self.assertEqual(row['contact'], expected, distance)

    def test_a_side_chain_contact_is_found_where_the_cb_definition_misses_it(self):
        # Two long side chains pointing at each other: CB-CB is 13 A, so the CB
        # definition calls no contact, while the closest heavy atoms are 4 A apart.
        sequence = 'LAAAL'
        atoms = backbone(sequence) + [
            {'position': 1, 'name': 'CB', 'element': 'C', 'xyz': (0.0, 0.0, 0.0)},
            {'position': 1, 'name': 'CG', 'element': 'C', 'xyz': (0.0, 0.0, 4.0)},
            {'position': 5, 'name': 'CB', 'element': 'C', 'xyz': (0.0, 0.0, 13.0)},
            {'position': 5, 'name': 'CG', 'element': 'C', 'xyz': (0.0, 0.0, 8.0)},
        ]
        path = write_cif(self.tmp / 'sidechain.cif', synthetic_cif(sequence, atoms))
        row = annotate_background({'name': 's.pdb', 'group': 'g', 'kind': 'natural',
                                   'length': 5, 'wildtype': sequence,
                                   'site_pairs': [[1, 5]]}, path)['site_pairs'][0]
        self.assertAlmostEqual(row['min_heavy_atom_angstrom'], 4.0, places=6)
        self.assertTrue(row['contact'])
        self.assertAlmostEqual(row['cb_angstrom'], 13.0, places=6)
        self.assertFalse(row['contact_cb'])
        self.assertGreater(row['cb_angstrom'], CB_CONTACT_ANGSTROM)

    def test_the_ensemble_fraction_counts_the_models_that_hold_the_contact(self):
        sequence = 'AAA'
        atoms = [{'position': index + 1, 'name': 'CA', 'element': 'C',
                  'xyz': (3.8 * index, 0.0, 0.0), 'model': 1} for index in range(3)]
        atoms += [{'position': index + 1, 'name': 'CA', 'element': 'C',
                   'xyz': (20.0 * index, 0.0, 0.0), 'model': 2} for index in range(3)]
        path = write_cif(self.tmp / 'ensemble.cif',
                         synthetic_cif(sequence, atoms, method='SOLUTION NMR'))
        row = annotate_background({'name': 'e.pdb', 'group': 'g', 'kind': 'natural',
                                   'length': 3, 'wildtype': sequence,
                                   'site_pairs': [[1, 2]]}, path)['site_pairs'][0]
        self.assertTrue(row['contact'])
        self.assertEqual(row['models_measured'], 2)
        self.assertAlmostEqual(row['contact_model_fraction'], 0.5, places=6)

    def test_an_isolated_atom_carries_its_whole_sphere_of_accessible_area(self):
        sequence = 'A'
        path = write_cif(self.tmp / 'atom.cif', synthetic_cif(
            sequence, [{'position': 1, 'name': 'CB', 'element': 'C', 'xyz': (0.0, 0.0, 0.0)}]))
        structure = load_structure(path)
        chain = structure['chains'][(1, 'A')]
        area = residue_accessibility(chain)[1]
        expected = 4 * math.pi * (VDW_RADII['C'] + PROBE_RADIUS) ** 2
        self.assertAlmostEqual(area / expected, 1.0, places=2)
        self.assertAlmostEqual(relative_accessibility(area, 'A'), area / 129.0, places=9)

    def test_burial_lowers_the_accessible_area_of_the_same_atom(self):
        # One carbon surrounded by six others at 3 A: the same atom, now enclosed.
        sequence = 'AAAAAAA'
        atoms = [{'position': 1, 'name': 'CB', 'element': 'C', 'xyz': (0.0, 0.0, 0.0)}]
        for index, offset in enumerate(((3, 0, 0), (-3, 0, 0), (0, 3, 0),
                                        (0, -3, 0), (0, 0, 3), (0, 0, -3)), start=2):
            atoms.append({'position': index, 'name': 'CB', 'element': 'C',
                          'xyz': tuple(float(value) for value in offset)})
        path = write_cif(self.tmp / 'buried.cif', synthetic_cif(sequence, atoms))
        chain = load_structure(path)['chains'][(1, 'A')]
        areas = residue_accessibility(chain)
        isolated = 4 * math.pi * (VDW_RADII['C'] + PROBE_RADIUS) ** 2
        self.assertLess(areas[1], 0.25 * isolated)
        self.assertGreater(areas[2], areas[1])


class TheNumberingMustCarryTheAssayedWildType(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parent / '_contact_numbering_tmp'
        self.tmp.mkdir(exist_ok=True)
        self.sequence = 'AAAAA'
        self.path = write_cif(self.tmp / 'line.cif', synthetic_cif(
            self.sequence, backbone(self.sequence)))

    def tearDown(self):
        for path in self.tmp.glob('*'):
            path.unlink()
        self.tmp.rmdir()

    def entry(self, wildtype: str, pairs=((1, 2),)) -> dict:
        return {'name': 'line.pdb', 'group': 'g', 'kind': 'natural',
                'length': len(wildtype), 'wildtype': wildtype,
                'site_pairs': [list(pair) for pair in pairs]}

    def test_one_substituted_residue_is_refused_with_its_identity_named(self):
        with self.assertRaises(ValueError) as caught:
            annotate_background(self.entry('AAVAA'), self.path)
        self.assertIn('identity', str(caught.exception))
        self.assertIn('%', str(caught.exception))

    def test_an_offset_wild_type_maps_to_the_structure_and_is_admitted(self):
        # A trimmed wild type sitting inside a longer entity keeps its own
        # 1-based numbering and must map onto the structure's later residues.
        sequence = 'GGAAAAA'
        atoms = backbone(sequence)
        path = write_cif(self.tmp / 'offset.cif', synthetic_cif(sequence, atoms))
        record = annotate_background(
            {**self.entry('AAAAA', pairs=((1, 2),)), 'name': 'offset.pdb'}, path)
        self.assertEqual(record['structure']['wildtype_to_entity_offset'], 2)
        self.assertAlmostEqual(record['site_pairs'][0]['min_heavy_atom_angstrom'], 3.8,
                               places=6)

    def test_an_unobserved_site_pair_residue_is_refused(self):
        sequence = 'AAAAA'
        atoms = [atom for atom in backbone(sequence) if atom['position'] != 3]
        path = write_cif(self.tmp / 'gap.cif', synthetic_cif(sequence, atoms))
        with self.assertRaises(ValueError) as caught:
            annotate_background({**self.entry('AAAAA', pairs=((1, 3),)),
                                 'name': 'gap.pdb'}, path)
        self.assertIn('unobserved', str(caught.exception))

    def test_the_matched_chain_is_the_one_that_carries_the_wild_type(self):
        structure = load_structure(self.path)
        matched = match_chain('AAAAA', structure)
        self.assertEqual(matched['label_asym_id'], 'A')
        self.assertEqual(matched['identity_over_wildtype'], 1.0)
        self.assertEqual(sorted(matched['mapping']), [1, 2, 3, 4, 5])

    def test_a_loop_row_count_that_does_not_divide_is_refused(self):
        with self.assertRaises(ValueError):
            read_cif('data_T\nloop_\n_atom_site.id\n_atom_site.type_symbol\n1 C\n2\n')


class TheMatchingBalancesTheDeclaredConfounders(unittest.TestCase):
    @staticmethod
    def row(group: str, separation: int, rsa: float, hydrophobic: int,
            contact: bool) -> dict:
        return {'group': group, 'background': group,
                'site_pair': f'{group}:{separation}-{rsa}-{int(contact)}',
                'separation': separation, 'rsa': [rsa, rsa],
                'hydrophobic_count': hydrophobic, 'charged_count': 0,
                'hydropathy_mean': 1.0, 'length': 60, 'contact': contact}

    def sample(self) -> tuple[list[dict], np.ndarray]:
        # Contacts sit mostly near in sequence, buried and hydrophobic; controls
        # mostly distant, exposed and polar, which is the imbalance this cohort
        # actually has. Both cells hold both arms, in reciprocal proportions, so
        # the unmatched difference is large and the matching can remove it.
        rows = []
        for index in range(12):
            group = f'g{index:02d}'
            for copies, (separation, rsa, hydrophobic, contact) in (
                    (3, (5, 0.10, 1, True)), (1, (30, 0.45, 0, True)),
                    (1, (5, 0.10, 1, False)), (3, (30, 0.45, 0, False))):
                for copy_index in range(copies):
                    row = self.row(group, separation, rsa, hydrophobic, contact)
                    row['site_pair'] = f"{row['site_pair']}#{copy_index}"
                    rows.append(row)
        return rows, np.asarray([row['contact'] for row in rows])

    def test_weighting_reduces_every_declared_imbalance(self):
        rows, treated = self.sample()
        design = cem_design(rows, treated, rsa_boundary=0.28)
        table = balance_table(rows, design['weight'], treated,
                              np.asarray([cell is not None for cell in design['cells']]))
        for name in ('separation', 'rsa_mean', 'hydrophobic_count'):
            with self.subTest(name):
                self.assertLess(abs(table[name]['standardized_after']),
                                abs(table[name]['standardized_before']))
                self.assertLessEqual(abs(table[name]['standardized_after']), 0.1)

    def test_a_cell_holding_one_arm_is_pruned_and_counted(self):
        rows, treated = self.sample()
        # One extra contact in a cell no control occupies: separation 3-9,
        # exposed, no hydrophobic residue.
        rows = rows + [self.row('g99', 5, 0.45, 0, True)]
        treated = np.asarray([row['contact'] for row in rows])
        design = cem_design(rows, treated, rsa_boundary=0.28)
        self.assertEqual(design['pruned_contacts_without_control'], 1)
        self.assertEqual(design['pruned_controls_without_contact'], 0)
        self.assertFalse(design['retained'][-1])

    def test_a_separation_outside_the_declared_cells_is_ineligible(self):
        rows, treated = self.sample()
        rows = rows + [self.row('g98', 2, 0.10, 1, True)]
        treated = np.asarray([row['contact'] for row in rows])
        design = cem_design(rows, treated, rsa_boundary=0.28)
        self.assertEqual(design['pruned_ineligible_separation'], 1)
        self.assertIsNone(design['cells'][-1])
        self.assertFalse(design['retained'][-1])

    def test_the_control_arm_carries_the_contact_arm_cell_shares(self):
        rows, treated = self.sample()
        design = cem_design(rows, treated, rsa_boundary=0.28)
        weight, cells = design['weight'], design['cells']
        for cell in design['retained_cells']:
            inside = [index for index, value in enumerate(cells) if value == cell]
            contacts = sum(weight[index] for index in inside if treated[index])
            controls = sum(weight[index] for index in inside if not treated[index])
            self.assertAlmostEqual(contacts, controls, places=9)

    def test_the_interval_refuses_a_group_count_below_the_shared_floor(self):
        rows, treated = self.sample()
        few = [row for row in rows if row['group'] < 'g07']
        record = two_stage_bootstrap(
            few, np.linspace(0.0, 1.0, len(few)),
            np.asarray([row['contact'] for row in few]), rsa_boundary=0.28, draws=100)
        self.assertTrue(record['degenerate'])
        self.assertIsNone(record['interval'])
        full = two_stage_bootstrap(rows, np.linspace(0.0, 1.0, len(rows)), treated,
                                   rsa_boundary=0.28, draws=200)
        self.assertFalse(full['degenerate'])
        self.assertEqual(len(full['interval']), 2)


class BootstrapGroupMultiplicity(unittest.TestCase):
    def test_repeated_groups_retain_separate_arm_weight(self):
        # Three source families, unequal numbers of pairs. Draw A, A, B;
        # the two independent inner draws of A have different arm counts.
        # Their arm means are (2, 0), (6, 0), (12, 0): the group-equal
        # draw is 20/3, not the 25/3 obtained by pooling both A occurrences.
        values = np.array([2., 6., 0., 12., 0., 0., 30., 0.])
        groups = np.array(['a'] * 3 + ['b'] * 3 + ['c'] * 2)
        arm = np.array([True, True, False, True, False, False, True, False])
        rows = [TheMatchingBalancesTheDeclaredConfounders.row(g, 5, .1, 1, a)
                for g, a in zip(groups, arm)]
        original = copy.deepcopy(rows)
        choices = [np.array([0, 0, 1]), np.array([0, 2, 2]),
                   np.array([1, 1, 2]), np.array([3, 4, 5])]
        for matched in (False, True):
            for equal, expected in ((False, 6.5), (True, 20 / 3)):
                with self.subTest(matched=matched, group_equal=equal):
                    with patch('src.capability.interactions.contact_enrichment.np.random.default_rng') as rng:
                        rng.return_value.choice.side_effect = choices
                        if matched:
                            result = two_stage_bootstrap(
                                rows, values, arm, rsa_boundary=.28,
                                group_equal=equal, draws=1, minimum_units=3)
                        else:
                            result = resampled_arm_difference(
                                values, groups, arm, group_equal=equal,
                                draws=1, minimum_units=3)
                    np.testing.assert_allclose(result['interval'], [expected, expected])
                    self.assertEqual(result['draws'], 1)
                    self.assertEqual(result['skipped_draws'], 0)
        self.assertEqual(rows, original)


class NoMeasurementReachesTheAnnotationOrTheMatching(unittest.TestCase):
    @staticmethod
    def cohort(epsilon: float, value: float) -> dict:
        wild, first, second, double = 'AAAAA', 'AVAAA', 'AAAVA', 'AVAVA'
        return {'schema': COHORT_SCHEMA,
                'backgrounds': [{
                    'name': 'line.pdb', 'group': 'nat-001', 'kind': 'natural',
                    'length': 5, 'site_pairs': [[2, 4]],
                    'sequences': sorted({wild, first, second, double}),
                    'cycles': [{'sequences': [wild, first, second, double],
                                'positions': [2, 4], 'separation': '3-9',
                                'epsilon': epsilon}],
                    'measurements': {sequence: {'value': value, 'rows': []}
                                     for sequence in (wild, first, second, double)}}]}

    def test_the_label_free_projection_is_invariant_to_every_measurement(self):
        first = site_pair_plan(self.cohort(-0.86, 1.5))
        second = site_pair_plan(self.cohort(+12.0, -7.25))
        self.assertEqual(json.dumps(first, sort_keys=True),
                         json.dumps(second, sort_keys=True))
        for plan in (first, second):
            serialised = json.dumps(plan)
            self.assertNotIn('epsilon', serialised)
            self.assertNotIn('measurements', serialised)

    def test_the_annotation_of_a_real_structure_is_invariant_to_the_endpoint(self):
        tmp = Path(__file__).resolve().parent / '_contact_label_tmp'
        tmp.mkdir(exist_ok=True)
        try:
            sequence = 'AAAAA'
            path = write_cif(tmp / 'line.cif', synthetic_cif(sequence, backbone(sequence)))
            digests = []
            for epsilon, value in ((-0.86, 1.5), (+12.0, -7.25)):
                plan = site_pair_plan(self.cohort(epsilon, value))
                record = annotate_background(plan['backgrounds'][0], path)
                digests.append(hashlib.sha256(
                    json.dumps(record, sort_keys=True).encode()).hexdigest())
            self.assertEqual(digests[0], digests[1])
        finally:
            for item in tmp.glob('*'):
                item.unlink()
            tmp.rmdir()

    def test_the_matching_cells_do_not_read_the_endpoint(self):
        rows = [TheMatchingBalancesTheDeclaredConfounders.row(
            f'g{index:02d}', 5 + index, 0.1 + 0.02 * index, index % 3, index % 2 == 0)
            for index in range(20)]
        treated = np.asarray([row['contact'] for row in rows])
        reference = cem_design(rows, treated, rsa_boundary=0.28)
        polluted = copy.deepcopy(rows)
        for index, row in enumerate(polluted):
            row['epsilon'] = 3.0 * index
            row['mean_abs_epsilon'] = -index
        after = cem_design(polluted, treated, rsa_boundary=0.28)
        np.testing.assert_array_equal(reference['weight'], after['weight'])
        self.assertEqual(reference['retained_cells'], after['retained_cells'])

    def test_a_cohort_whose_cycles_disagree_on_the_wild_type_is_refused(self):
        cohort = self.cohort(-0.5, 1.0)
        cohort['backgrounds'][0]['cycles'].append(
            {'sequences': ['AAAAV', 'AVAAV', 'AAAVV', 'AVAVV'], 'positions': [2, 4],
             'separation': '3-9', 'epsilon': 0.1})
        with self.assertRaises(ValueError) as caught:
            site_pair_plan(cohort)
        self.assertIn('wild-type', str(caught.exception))

    def test_a_declared_site_pair_that_no_cycle_carries_is_refused(self):
        cohort = self.cohort(-0.5, 1.0)
        cohort['backgrounds'][0]['site_pairs'] = [[2, 4], [1, 5]]
        with self.assertRaises(ValueError) as caught:
            site_pair_plan(cohort)
        self.assertIn('site pairs', str(caught.exception))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
