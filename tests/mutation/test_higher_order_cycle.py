"""Conditions that must hold for the higher-order gate's cycle construction.

The gate's whole claim rests on one identity — a mutation pair measured in two
backgrounds that differ at one untouched site is an eight-corner third-order
measurement — and on its sign convention. These tests pin both on synthetic input
with a known interaction, check that a strictly additive generator produces
exactly zero third-order signal, and check that the support enumeration reads only
the measured keys so that no held-out label can reach it.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.mutation.higher_order_cycle import (
    BackgroundContrast, background_difference, complete_pairs, cycle_contrast,
    enumerate_contrasts, kish_effective_units)

#: One reference background and two variants: the first differs at site 5 only,
#: the second at sites 5 and 9, so the first gives an exact third-order cube and
#: the second only a background contrast of order four.
REFERENCE = 'ACDEFGHIKL'
VARIANT_ONE_SITE = 'ACDEWGHIKL'
VARIANT_TWO_SITES = 'ACDEWGHIWL'
FIRST = (1, 'M')
SECOND = (7, 'P')
WILDTYPES = {'bg-a.pdb': REFERENCE, 'bg-a.pdb_G5W': VARIANT_ONE_SITE,
             'bg-b.pdb': VARIANT_TWO_SITES}
CORNERS = tuple((a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1))
#: An eight-corner contrast of values of order 1 kcal/mol cancels to within a few
#: multiples of double-precision epsilon, far below the 0.4 kcal/mol channel
#: discordance the endpoint is measured against.
CONSTRUCTION_ZERO_KCAL_MOL = 1e-12


def cube_states(pair=(FIRST, SECOND)) -> set[tuple]:
    """Every state a complete pairwise cycle needs, in one background."""
    first, second = pair
    return {(), (first,), (second,), (first, second)}


class CycleContrastConvention(unittest.TestCase):
    def test_order_two_is_the_pairwise_cycle_the_instrument_declared(self):
        values = {(1, 1): 4.0, (1, 0): 1.5, (0, 1): 0.5, (0, 0): 0.25}
        self.assertAlmostEqual(cycle_contrast(values), 4.0 - 1.5 - 0.5 + 0.25)

    def test_order_three_signs_place_the_all_mutant_corner_positive(self):
        for corner in CORNERS:
            values = {c: 0.0 for c in CORNERS}
            values[corner] = 1.0
            expected = (-1) ** (3 - sum(corner))
            self.assertEqual(cycle_contrast(values), float(expected), corner)

    def test_a_planted_third_order_term_is_recovered_with_its_sign(self):
        """A generator carrying one known omega must return exactly that omega."""
        planted = -0.75
        values = {c: 0.31 + 1.1 * c[0] - 0.4 * c[1] + 2.0 * c[2]
                  + 0.6 * c[0] * c[1] - 0.2 * c[0] * c[2] + 0.9 * c[1] * c[2]
                  + planted * c[0] * c[1] * c[2] for c in CORNERS}
        self.assertAlmostEqual(cycle_contrast(values), planted, places=12)

    def test_a_strictly_additive_generator_gives_its_construction_zero(self):
        """Not measurably nonzero: the residue is floating-point cancellation.

        The pairwise experiment reports its own construction zeros the same way,
        at 6.66e-16 for the additive cycle and 2.84e-14 for the independent-site
        contrast, rather than replacing them with a manufactured comparator.
        """
        values = {c: 0.31 + 1.1 * c[0] - 0.4 * c[1] + 2.0 * c[2] for c in CORNERS}
        self.assertLess(abs(cycle_contrast(values)), CONSTRUCTION_ZERO_KCAL_MOL)

    def test_a_pairwise_generator_with_no_triple_term_gives_its_construction_zero(self):
        """Position-independent couplings of any size leave omega at zero."""
        values = {c: 0.31 + 1.1 * c[0] + 0.6 * c[0] * c[1] - 3.0 * c[0] * c[2]
                  + 7.5 * c[1] * c[2] for c in CORNERS}
        self.assertLess(abs(cycle_contrast(values)), CONSTRUCTION_ZERO_KCAL_MOL)

    def test_a_partial_cube_has_no_contrast(self):
        values = {c: 1.0 for c in CORNERS if c != (1, 1, 1)}
        with self.assertRaises(ValueError):
            cycle_contrast(values)
        with self.assertRaises(ValueError):
            cycle_contrast({})


class BackgroundContrastConstruction(unittest.TestCase):
    def setUp(self):
        self.contrast = BackgroundContrast('bg-a.pdb', 'bg-a.pdb_G5W', (4,), FIRST, SECOND)

    def test_background_difference_refuses_unequal_length_and_identity(self):
        with self.assertRaises(ValueError):
            background_difference(REFERENCE, REFERENCE + 'A')
        with self.assertRaises(ValueError):
            background_difference(REFERENCE, REFERENCE)
        self.assertEqual(background_difference(REFERENCE, VARIANT_TWO_SITES), (4, 8))

    def test_the_eight_corners_are_the_declared_sequences(self):
        sequences = self.contrast.sequences(WILDTYPES)
        self.assertEqual(len(set(sequences.values())), 8)
        self.assertEqual(sequences[(0, 0, 0)], REFERENCE)
        self.assertEqual(sequences[(0, 0, 1)], VARIANT_ONE_SITE)
        self.assertEqual(sequences[(1, 0, 0)], 'AMDEFGHIKL')
        self.assertEqual(sequences[(0, 1, 0)], 'ACDEFGHPKL')
        self.assertEqual(sequences[(1, 1, 0)], 'AMDEFGHPKL')
        self.assertEqual(sequences[(1, 1, 1)], 'AMDEWGHPKL')
        for corner, sequence in sequences.items():
            self.assertEqual(sequence[4], 'W' if corner[2] else 'F')

    def test_omega_equals_the_variant_minus_reference_epsilon(self):
        """The identity that makes a recurring pair a third-order measurement."""
        values = {name: {state: 0.41 * index + 1.7 * offset
                         for index, state in enumerate(sorted(cube_states()))}
                  for (name, offset) in (('bg-a.pdb', 0), ('bg-a.pdb_G5W', 1))}
        values['bg-a.pdb_G5W'][(FIRST, SECOND)] += 0.83
        cube = self.contrast.value_cube(values)
        epsilon = {}
        for tag, bit, name in (('reference', 0, 'bg-a.pdb'), ('variant', 1, 'bg-a.pdb_G5W')):
            epsilon[tag] = cycle_contrast(
                {(a, b): cube[(a, b, bit)] for a in (0, 1) for b in (0, 1)})
            self.assertAlmostEqual(
                epsilon[tag],
                values[name][(FIRST, SECOND)] - values[name][(FIRST,)]
                - values[name][(SECOND,)] + values[name][()], places=12)
        self.assertAlmostEqual(self.contrast.omega(values),
                               epsilon['variant'] - epsilon['reference'], places=12)

    def test_a_per_background_additive_offset_cancels(self):
        base = {name: {state: 0.1 * index for index, state in enumerate(sorted(cube_states()))}
                for name in ('bg-a.pdb', 'bg-a.pdb_G5W')}
        shifted = {name: {state: value + offset for state, value in states.items()}
                   for (name, states), offset in zip(base.items(), (3.5, -1.25))}
        self.assertAlmostEqual(self.contrast.omega(base), self.contrast.omega(shifted), places=12)

    def test_swapping_reference_and_variant_flips_the_sign(self):
        values = {name: {state: 0.37 * index + (0.9 if name.endswith('W') else 0.0)
                         * (len(state) == 2)
                         for index, state in enumerate(sorted(cube_states()))}
                  for name in ('bg-a.pdb', 'bg-a.pdb_G5W')}
        swapped = BackgroundContrast('bg-a.pdb_G5W', 'bg-a.pdb', (4,), FIRST, SECOND)
        self.assertAlmostEqual(self.contrast.omega(values), -swapped.omega(values), places=12)
        self.assertNotAlmostEqual(self.contrast.omega(values), 0.0)

    def test_construction_refuses_a_pair_that_touches_the_background_difference(self):
        overlapping = BackgroundContrast('bg-a.pdb', 'bg-a.pdb_G5W', (4,), (4, 'M'), SECOND)
        with self.assertRaises(ValueError):
            overlapping.sequences(WILDTYPES)

    def test_construction_refuses_a_substitution_that_changes_nothing(self):
        idle = BackgroundContrast('bg-a.pdb', 'bg-a.pdb_G5W', (4,), (1, REFERENCE[1]), SECOND)
        with self.assertRaises(ValueError):
            idle.sequences(WILDTYPES)

    def test_construction_refuses_a_mismatched_background_difference(self):
        wrong = BackgroundContrast('bg-a.pdb', 'bg-b.pdb', (4,), FIRST, SECOND)
        with self.assertRaises(ValueError):
            wrong.sequences(WILDTYPES)
        self.assertEqual(
            BackgroundContrast('bg-a.pdb', 'bg-b.pdb', (4, 8), FIRST, SECOND).order, 4)


class SupportEnumeration(unittest.TestCase):
    def test_complete_pairs_needs_the_wild_type_and_both_singles(self):
        self.assertEqual(complete_pairs(REFERENCE, cube_states()), {(FIRST, SECOND)})
        for missing in ((), (FIRST,), (SECOND,)):
            partial = cube_states() - {missing}
            self.assertEqual(complete_pairs(REFERENCE, partial), set())

    def test_complete_pairs_refuses_states_from_another_numbering(self):
        alien = cube_states(((1, REFERENCE[1]), SECOND))
        with self.assertRaises(ValueError):
            complete_pairs(REFERENCE, alien)

    def test_enumeration_reads_keys_only_and_honours_the_distance_cut(self):
        states = {name: cube_states() for name in WILDTYPES}
        one = enumerate_contrasts(WILDTYPES, states, max_background_distance=1)
        self.assertEqual([(c.reference, c.variant, c.background_sites) for c in one],
                         [('bg-a.pdb', 'bg-a.pdb_G5W', (4,)),
                          ('bg-a.pdb_G5W', 'bg-b.pdb', (8,))])
        self.assertEqual({c.order for c in one}, {3})
        two = enumerate_contrasts(WILDTYPES, states, max_background_distance=2)
        self.assertEqual(len(two), 3)
        self.assertEqual({c.order for c in two}, {3, 4})
        with self.assertRaises(ValueError):
            enumerate_contrasts(WILDTYPES, states, max_background_distance=0)

    def test_enumeration_drops_a_pair_touching_the_background_difference(self):
        pair = ((4, 'M'), SECOND)
        states = {name: cube_states(pair) for name in ('bg-a.pdb', 'bg-a.pdb_G5W')}
        self.assertEqual(
            enumerate_contrasts(WILDTYPES, states, max_background_distance=1), [])

    def test_enumeration_drops_unequal_length_backgrounds(self):
        wildtypes = {'bg-a.pdb': REFERENCE, 'bg-c.pdb': REFERENCE + 'A'}
        states = {name: cube_states() for name in wildtypes}
        self.assertEqual(
            enumerate_contrasts(wildtypes, states, max_background_distance=10), [])

    def test_kish_count_matches_its_definition(self):
        self.assertAlmostEqual(kish_effective_units([4, 4, 4, 4]), 4.0)
        self.assertAlmostEqual(kish_effective_units([100, 1]), 101 ** 2 / (100 ** 2 + 1))
        self.assertEqual(kish_effective_units([]), 0.0)


class InventoryStageContract(unittest.TestCase):
    """The declaration must bind the noise-floor stage to the support it declared."""

    SCRIPT = REPO_ROOT / 'scripts/capability/mutation/inventory_higher_order_support.py'

    def test_noise_floor_refuses_to_run_without_a_declaration(self):
        self.assertTrue(self.SCRIPT.exists())
        run = subprocess.run([sys.executable, str(self.SCRIPT), 'noise-floor',
                              '--out', '/dev/null'], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('--inventory', run.stderr)

    def test_the_inventory_stage_reads_no_measurement_column(self):
        """The support-deciding call site must be handed keys, never values."""
        source = self.SCRIPT.read_text(encoding='utf-8')
        self.assertIn("enumerate_contrasts(keys['wildtypes'], keys['states']", source)
        self.assertNotIn("enumerate_contrasts(keys['measured']", source)

    def test_the_committed_inventory_and_noise_floor_agree_on_support(self):
        inventory = REPO_ROOT / 'archive/logs/shared/gate_higher_order_20260923/inventory.json'
        floor = REPO_ROOT / 'archive/logs/shared/gate_higher_order_20260923/noise_floor.json'
        if not inventory.exists() or not floor.exists():
            self.skipTest('gate outputs are retained under ignored logs/ only')
        declared = json.loads(inventory.read_bytes())['background_dependence_endpoint']
        measured = json.loads(floor.read_bytes())['cuts']
        self.assertEqual(set(declared), set(measured))
        for key, entry in measured.items():
            self.assertEqual(entry['declared_support'], declared[key])


if __name__ == '__main__':
    unittest.main()
