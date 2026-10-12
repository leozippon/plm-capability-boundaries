"""The declared state-level repair of the frozen measured cohort.

The frozen cohort's cycle epsilon is the raw four-state combination of state
medians that still contain insertion- and deletion-construct rows. The measured
side of the cycle interaction must therefore apply the declaration at the state
level and rebuild the epsilon, not read the frozen one. These tests pin the
propagation of a corrected state into every cycle it appears in, the removal of a
cycle that indexes an absent state, the digest binding, the exact-raw path with no
declaration, and the authoritative accounting on the retained cohort bytes.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.capability.interactions import analyse_measured_cycle_interaction as amci
from src.capability.core.io import sha256_file
from src.capability.interactions import generated_mutation as gm

WILD = 'AAAAAAAAAA'
LOW = 'CAAAAAAAAA'
HIGH = 'AAAAADAAAA'
DOUBLE = 'CAAAADAAAA'
ALT_LOW = 'AAAAAAAAAG'
ALT_DOUBLE = 'CAAAAAAAAG'
ALT_HIGH_DOUBLE = 'AAAAADAAAG'
VALUES = {WILD: 0.0, LOW: 1.0, HIGH: 2.0, DOUBLE: 1.5,
          ALT_LOW: 0.5, ALT_DOUBLE: 1.2, ALT_HIGH_DOUBLE: 0.4}
CYCLES = [
    ((WILD, LOW, HIGH, DOUBLE), (1, 2)),
    ((WILD, LOW, ALT_LOW, ALT_DOUBLE), (1, 3)),
    ((WILD, HIGH, ALT_LOW, ALT_HIGH_DOUBLE), (2, 3)),
]


def frozen_cohort(*, name: str = 'bg', group: str = 'g1') -> dict:
    rows = []
    for (wild, low, high, double), positions in CYCLES:
        rows.append({
            'sequences': [wild, low, high, double],
            'positions': list(positions),
            'separation': '10+',
            'epsilon': (VALUES[double] - VALUES[low] - VALUES[high] + VALUES[wild]),
        })
    return {'backgrounds': [{
        'name': name, 'group': group,
        'measurements': {sequence: {'value': value} for sequence, value in VALUES.items()},
        'cycles': rows,
    }]}


def declaration(states) -> dict:
    return {
        'schema': 'pairwise_indel_exclusion_v1',
        'cohort_sha256': 'abc',
        'states': [
            {'background': background, 'sequence': sequence,
             'value_kcal_mol': VALUES[sequence],
             'value_without_indel_rows_kcal_mol': value}
            for background, sequence, value in states
        ],
    }


def raw_map(frozen) -> dict:
    background = frozen['backgrounds'][0]
    return {
        (background['name'], gm.state_label(cycle['sequences'][0], cycle['sequences'][3])):
            cycle['epsilon']
        for cycle in background['cycles']
    }


class StateLevelRepair(unittest.TestCase):
    def test_a_corrected_wildtype_moves_every_cycle_it_is_in_by_the_signed_change(self):
        frozen = frozen_cohort()
        repaired, accounting, _ = amci.corrected_measured(
            frozen, declaration([('bg', WILD, 1.0)]), cohort_sha256='abc')
        raw = raw_map(frozen)
        self.assertEqual(set(repaired), set(raw))
        for key, value in repaired.items():
            self.assertAlmostEqual(value - raw[key], 1.0, places=12)
        self.assertEqual(accounting['changed_cycles_count'], 3)
        self.assertAlmostEqual(accounting['largest_change_kcal_mol'], 1.0, places=12)

    def test_a_corrected_single_state_moves_its_cycles_by_the_opposite_sign(self):
        frozen = frozen_cohort()
        raw = raw_map(frozen)
        repaired, accounting, _ = amci.corrected_measured(
            frozen, declaration([('bg', LOW, 1.5)]), cohort_sha256='abc')
        moved = {key: value - raw[key] for key, value in repaired.items()
                 if value != raw[key]}
        # LOW is a single state, so the four-state combination carries it negatively
        # and only the two cycles that index it move.
        self.assertEqual(len(moved), 2)
        for delta in moved.values():
            self.assertAlmostEqual(delta, -0.5, places=12)
        self.assertEqual(accounting['changed_cycles_count'], 2)

    def test_a_null_state_removes_only_the_cycles_that_index_it(self):
        frozen = frozen_cohort()
        repaired, accounting, absent = amci.corrected_measured(
            frozen, declaration([('bg', ALT_DOUBLE, None)]), cohort_sha256='abc')
        self.assertEqual(accounting['removed_cycles_count'], 1)
        self.assertEqual(accounting['removed_cycles'][0]['site_pair'], 'bg:1-3')
        self.assertEqual(accounting['retained'],
                         {'cycles': 2, 'site_pairs': 2, 'groups': 1})
        self.assertNotIn(('bg', gm.state_label(WILD, ALT_DOUBLE)), repaired)
        self.assertEqual(absent, {('bg', ALT_DOUBLE)})

    def test_a_declaration_over_a_different_cohort_is_refused(self):
        frozen = frozen_cohort()
        with self.assertRaises(SystemExit):
            amci.corrected_measured(frozen, declaration([('bg', WILD, 1.0)]),
                                    cohort_sha256='a different cohort')

    def test_without_a_declaration_the_raw_epsilon_is_kept_exactly(self):
        frozen = frozen_cohort()
        repaired, accounting, absent = amci.corrected_measured(
            frozen, None, cohort_sha256='abc')
        self.assertEqual(repaired, raw_map(frozen))
        self.assertFalse(accounting['repair_applied'])
        self.assertEqual(accounting['removed_cycles_count'], 0)
        self.assertEqual(accounting['changed_cycles_count'], 0)
        self.assertEqual(accounting['retained'],
                         {'cycles': 3, 'site_pairs': 3, 'groups': 1})
        self.assertEqual(absent, set())

    def test_the_frozen_epsilon_is_kept_exactly_and_the_declaration_recomputes(self):
        frozen = frozen_cohort()
        cycle = frozen['backgrounds'][0]['cycles'][0]
        recomputed = (VALUES[cycle['sequences'][3]] - VALUES[cycle['sequences'][1]]
                      - VALUES[cycle['sequences'][2]] + VALUES[cycle['sequences'][0]])
        perturbed = recomputed + 5e-10
        self.assertLess(abs(perturbed - recomputed), amci.CYCLE_IDENTITY_TOLERANCE)
        cycle['epsilon'] = perturbed
        key = ('bg', gm.state_label(cycle['sequences'][0], cycle['sequences'][3]))
        raw, accounting, _ = amci.corrected_measured(frozen, None, cohort_sha256='abc')
        # The no-declaration identity is the frozen value, not the recomputed combination,
        # even where the two differ only by roundoff inside the checked tolerance.
        self.assertEqual(raw[key], perturbed)
        self.assertNotEqual(raw[key], recomputed)
        self.assertEqual(accounting['policy'], amci.RAW_MEASURED_POLICY)
        self.assertEqual(accounting['changed_cycles_count'], 0)
        declared, declared_accounting, _ = amci.corrected_measured(
            frozen, declaration([('bg', WILD, 1.0)]), cohort_sha256='abc')
        # A declaration recomputes from the corrected states and does not carry the
        # roundoff-shifted frozen value.
        self.assertNotEqual(declared[key], perturbed)
        self.assertAlmostEqual(declared[key], recomputed + 1.0, places=12)
        self.assertEqual(declared_accounting['policy'], amci.DECLARED_REPAIR_POLICY)

    def test_a_cohort_whose_epsilon_disagrees_with_its_four_states_is_refused(self):
        frozen = frozen_cohort()
        frozen['backgrounds'][0]['cycles'][0]['epsilon'] += 1.0
        with self.assertRaises(SystemExit):
            amci.corrected_measured(frozen, None, cohort_sha256='abc')

    def test_measured_contrast_refuses_a_support_pair_without_a_measurement(self):
        support = [{'site_pair': 'bg:1-2', 'contact': True, 'contact_cb': True,
                    'group': 'g1'}]
        pairs = {'bg:1-2': {'mean_abs_measured_epsilon': None, 'cycles': 1}}
        with self.assertRaises(SystemExit):
            amci.measured_contrast(support, pairs, 1.0,
                                   correction={'repair_applied': False})


class FrozenCohortAccounting(unittest.TestCase):
    cohort = Path('archive/logs/R3/pairwise_cohort_20260924/cohort.json')
    declaration = Path('results/R3/pairwise_indel_exclusion_20260928/pairwise_indel_exclusion.json')

    @unittest.skipUnless(cohort.exists() and declaration.exists(),
                         'the frozen cohort and its declaration are retained locally')
    def test_the_declared_repair_reproduces_the_authoritative_accounting(self):
        frozen = json.loads(self.cohort.read_text())
        record = json.loads(self.declaration.read_text())
        digest = sha256_file(self.cohort)
        self.assertEqual(record['cohort_sha256'], digest)
        measured, accounting, absent = amci.corrected_measured(
            frozen, record, cohort_sha256=digest)
        self.assertEqual(accounting['removed_cycles_count'], 3)
        self.assertEqual({row['site_pair'] for row in accounting['removed_cycles']},
                         {'1O6X.pdb:1-2', '2AMI.pdb:1-2', '2D1U.pdb:1-2'})
        self.assertEqual(accounting['changed_cycles_count'], 127)
        self.assertEqual(accounting['changed_site_pairs'],
                         {'2AMI.pdb:10-70': 73, '2AMI.pdb:14-63': 39, '2AMI.pdb:14-66': 15})
        self.assertEqual(len(accounting['changed_epsilon_kcal_mol']), 1)
        self.assertAlmostEqual(accounting['changed_epsilon_kcal_mol'][0], 0.03153236940292947,
                               places=12)
        self.assertEqual(accounting['retained'],
                         {'cycles': 8189, 'site_pairs': 214, 'groups': 64})
        self.assertEqual(len(accounting['absent_states']), 3)
        self.assertEqual(len(accounting['corrected_states']), 2)
        self.assertEqual(len(measured), 8189)
        self.assertEqual(len(absent), 3)
        # The conservative cycle-level exclusion is a different, narrower support and
        # must never be reported as this state-level repair.
        self.assertNotEqual(
            (accounting['retained']['cycles'], accounting['retained']['site_pairs'],
             accounting['retained']['groups']),
            (8062, 211, 63))


if __name__ == '__main__':
    unittest.main()
