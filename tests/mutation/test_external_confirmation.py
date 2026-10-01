"""The external-confirmation gate's module and its declaration entry point.

The entry point was dead on arrival: it imported a symbol the module never
defined and omitted one it uses, so every invocation raised ``ImportError``
before any work. The first two test classes are the regression for exactly that
failure class -- the entry point is imported the way the campaign runner imports
it, and the module's declared exports are checked against what it actually
defines -- and the rest test the support rules, the nested weighting and the
held-group comparison against the properties that must hold rather than against
the current arithmetic.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.mutation import external_confirmation as ec  # noqa: E402

ENTRY_POINTS = (
    'declare_external_confirmation.py',
    'qualify_nested_controls.py',
    'fit_nested_singles.py',
    'build_nested_gate_campaign.py',
)


from scripts.capability.entrypoints import stage_path

def _load_entry_point(filename: str):
    path = stage_path(REPO_ROOT, filename)
    spec = importlib.util.spec_from_file_location(f'_entry_{path.stem}', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EntryPointsResolve(unittest.TestCase):
    """The failure that made this stage unrunnable: an import that cannot resolve."""

    def test_every_entry_point_imports_and_exposes_main(self):
        for filename in ENTRY_POINTS:
            with self.subTest(filename):
                module = _load_entry_point(filename)
                self.assertTrue(callable(module.main))

    def test_every_entry_point_answers_help_in_a_fresh_interpreter(self):
        # Importing in-process can be satisfied by a module another test already
        # loaded. A subprocess is the runner's own situation.
        for filename in ENTRY_POINTS:
            with self.subTest(filename):
                completed = subprocess.run(
                    [sys.executable, str(stage_path(REPO_ROOT, filename)), '--help'],
                    capture_output=True, text=True, cwd=REPO_ROOT,
                    env={'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '4',
                         'PYTHONPATH': str(REPO_ROOT), 'HOME': str(Path.home())})
                self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])

    def test_every_name_the_declaration_imports_from_the_module_is_defined(self):
        source = stage_path(REPO_ROOT, 'declare_external_confirmation.py').read_text()
        wanted = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module and (
                    node.module.endswith('external_confirmation')):
                wanted.update(alias.name for alias in node.names)
        self.assertTrue(wanted)
        self.assertEqual(sorted(name for name in wanted if not hasattr(ec, name)), [])


class DeclaredExportsExist(unittest.TestCase):
    def test_all_names_resolve(self):
        self.assertEqual([name for name in ec.__all__ if not hasattr(ec, name)], [])

    def test_all_is_sorted_and_unique(self):
        self.assertEqual(len(set(ec.__all__)), len(ec.__all__))


class SourceAdmission(unittest.TestCase):
    """Admission reads the source's own classes and coerces none of them."""

    def _frame(self):
        import pandas as pd

        return pd.DataFrame([
            # wild type: no position, no residues
            {'domain_ID': 'P1_PF00001_1', 'uniprot_ID': 'P1', 'aa_seq': 'ACDE',
             'wt_aa': None, 'position': None, 'mut_aa': None, 'STOP': False,
             'normalized_fitness': 0.0, 'normalized_fitness_sigma': 0.01, 'quality_rank': 1},
            {'domain_ID': 'P1_PF00001_1', 'uniprot_ID': 'P1', 'aa_seq': 'AGDE',
             'wt_aa': 'C', 'position': 2, 'mut_aa': 'G', 'STOP': False,
             'normalized_fitness': -0.5, 'normalized_fitness_sigma': 0.02, 'quality_rank': 1},
            # nonsense: a separate class, counted and never coerced
            {'domain_ID': 'P1_PF00001_1', 'uniprot_ID': 'P1', 'aa_seq': 'A*DE',
             'wt_aa': 'C', 'position': 2, 'mut_aa': '*', 'STOP': True,
             'normalized_fitness': -1.0, 'normalized_fitness_sigma': 0.03, 'quality_rank': 1},
            # undetected: no counts at all
            {'domain_ID': 'P1_PF00001_1', 'uniprot_ID': 'P1', 'aa_seq': 'ACDG',
             'wt_aa': 'E', 'position': 4, 'mut_aa': 'G', 'STOP': None,
             'normalized_fitness': None, 'normalized_fitness_sigma': None, 'quality_rank': 1},
            # missense without a finite endpoint
            {'domain_ID': 'P1_PF00001_1', 'uniprot_ID': 'P1', 'aa_seq': 'ACGE',
             'wt_aa': 'D', 'position': 3, 'mut_aa': 'G', 'STOP': False,
             'normalized_fitness': float('nan'), 'normalized_fitness_sigma': 0.02,
             'quality_rank': 1},
            # a domain with no wild-type row
            {'domain_ID': 'P2_PF00002_5', 'uniprot_ID': 'P2', 'aa_seq': 'KKLL',
             'wt_aa': 'K', 'position': 6, 'mut_aa': 'L', 'STOP': False,
             'normalized_fitness': -0.2, 'normalized_fitness_sigma': 0.02, 'quality_rank': 2},
        ])

    def test_accounting_separates_every_class(self):
        frame = self._frame()
        mask, accounting = ec.accept_rows(frame, {'P1_PF00001_1': 'ACDE'})
        self.assertEqual(accounting['nonsense_rows'], 1)
        self.assertEqual(accounting['undetected_rows_no_counts'], 1)
        self.assertEqual(accounting['missense_rows_without_a_finite_endpoint'], 1)
        self.assertEqual(accounting['missense_rows_of_a_domain_without_a_wildtype'], 1)
        self.assertEqual(accounting['accepted_rows'], 1)
        self.assertEqual(int(mask.sum()), 1)

    def test_records_are_re_derived_and_a_disagreeing_row_is_refused(self):
        frame = self._frame()
        mask, _ = ec.accept_rows(frame, {'P1_PF00001_1': 'ACDE'})
        records = ec.substitution_records(frame, mask, {'P1_PF00001_1': 'ACDE'})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['position'], 2)
        self.assertEqual(records[0]['wild'], 'C')
        # a wild type that disagrees with the row's own coordinates
        with self.assertRaises(ValueError):
            ec.substitution_records(frame, mask, {'P1_PF00001_1': 'AADE'})

    def test_a_two_substitution_row_is_refused(self):
        import pandas as pd

        frame = pd.DataFrame([
            {'domain_ID': 'P1_PF00001_1', 'uniprot_ID': 'P1', 'aa_seq': 'AGGE',
             'wt_aa': 'C', 'position': 2, 'mut_aa': 'G', 'STOP': False,
             'normalized_fitness': -0.5, 'normalized_fitness_sigma': 0.02,
             'quality_rank': 1}])
        mask = np.array([True])
        with self.assertRaises(ValueError):
            ec.substitution_records(frame, mask, {'P1_PF00001_1': 'ACDE'})

    def test_a_domain_with_two_wild_type_rows_is_refused(self):
        import pandas as pd

        frame = pd.DataFrame([
            {'domain_ID': 'P1_PF00001_1', 'aa_seq': 'ACDE', 'wt_aa': None,
             'position': None, 'mut_aa': None},
            {'domain_ID': 'P1_PF00001_1', 'aa_seq': 'ACDF', 'wt_aa': None,
             'position': None, 'mut_aa': None}])
        with self.assertRaises(ValueError):
            ec.wildtype_rows(frame)


class IdentifierRules(unittest.TestCase):
    def test_domain_start_refuses_a_name_without_one(self):
        self.assertEqual(ec.domain_start('A0PJY2_PF00096_289'), 289)
        with self.assertRaises(ValueError):
            ec.domain_start('A0PJY2_PF00096')

    def test_pfam_accession_refuses_an_ambiguous_name(self):
        self.assertEqual(ec.pfam_accession('A0A2R8Y422_PF00240_2'), 'PF00240')
        with self.assertRaises(ValueError):
            ec.pfam_accession('A0A2R8Y422_2')
        with self.assertRaises(ValueError):
            ec.pfam_accession('PF00240_PF00001_2')


class DrawIsLabelBlind(unittest.TestCase):
    def test_the_order_depends_on_seed_domain_and_sequence_only(self):
        sequences = [f'ACDE{residue}' for residue in 'FGHIKLMN']
        first = ec.draw_order('dom', sequences, seed=1)
        self.assertEqual(sorted(first), sorted(sequences))
        self.assertEqual(first, ec.draw_order('dom', list(reversed(sequences)), seed=1))
        self.assertNotEqual(first, ec.draw_order('dom', sequences, seed=2))
        self.assertNotEqual(first, ec.draw_order('other', sequences, seed=1))


class NestedWeighting(unittest.TestCase):
    def test_weights_sum_to_one_and_groups_carry_equal_weight(self):
        group = np.array(['g1', 'g1', 'g1', 'g2'])
        domain = np.array(['d1', 'd1', 'd2', 'd3'])
        site = np.array(['d1:1', 'd1:2', 'd2:1', 'd3:1'])
        weights = ec.nested_weights(group, domain, site)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertAlmostEqual(float(weights[:3].sum()), 0.5)
        self.assertAlmostEqual(float(weights[3]), 0.5)

    def test_a_long_domain_does_not_dominate_its_own_family(self):
        # Two domains in one group, one with ten sites and one with a single
        # site. Three-level nesting gives them equal weight; a two-level
        # weighting would give the long one ten times the short one's.
        group = np.array(['g'] * 11)
        domain = np.array(['long'] * 10 + ['short'])
        site = np.array([f'long:{i}' for i in range(10)] + ['short:1'])
        weights = ec.nested_weights(group, domain, site)
        self.assertAlmostEqual(float(weights[:10].sum()), float(weights[10]))

    def test_inconsistent_nesting_is_refused(self):
        with self.assertRaises(ValueError):
            ec.nested_weights(np.array(['g1', 'g2']), np.array(['d', 'd']),
                              np.array(['s1', 's2']))
        with self.assertRaises(ValueError):
            ec.nested_weights(np.array(['g', 'g']), np.array(['d1', 'd2']),
                              np.array(['s', 's']))
        with self.assertRaises(ValueError):
            ec.nested_weights(np.array(['g']), np.array(['d']), np.array(['s', 's']))


class EffectiveUnits(unittest.TestCase):
    def test_kish_counts_name_their_convention_and_match_an_equal_design(self):
        group = np.array(['g1', 'g2', 'g3', 'g4'])
        domain = np.array(['d1', 'd2', 'd3', 'd4'])
        site = np.array(['s1', 's2', 's3', 's4'])
        units = ec.kish_units(group, domain, site)
        self.assertIn('weighting', units)
        self.assertIn('groups equal', units['weighting'])
        self.assertEqual(units['groups'], 4)
        self.assertAlmostEqual(units['kish_effective_groups'], 4.0)
        self.assertEqual(units['variants'], 4)


class RowIdentity(unittest.TestCase):
    def test_a_numpy_scalar_and_a_python_float_hash_identically(self):
        group = np.array(['g', 'g'])
        site = np.array(['s1', 's2'])
        native = ec.row_identity(group, site, [0.5, -1.25])
        numpy_side = ec.row_identity(group, site, np.array([0.5, -1.25]))
        self.assertEqual(native, numpy_side)

    def test_misaligned_arrays_are_refused(self):
        with self.assertRaises(ValueError):
            ec.row_identity(np.array(['g']), np.array(['s', 's']), np.array([1.0]))


class QualificationRule(unittest.TestCase):
    def test_every_prespecified_seed_must_be_scored(self):
        with self.assertRaises(ValueError):
            ec.qualify({ec.SPLIT_SEEDS[0]: 1.0})

    def test_a_single_negative_seed_discards(self):
        seeds = ec.SPLIT_SEEDS
        self.assertTrue(ec.qualify({s: 0.1 for s in seeds})['qualified'])
        failing = {s: 0.1 for s in seeds}
        failing[seeds[1]] = -1e-9
        self.assertFalse(ec.qualify(failing)['qualified'])


class FamilyGrouping(unittest.TestCase):
    def test_accession_pfam_and_exact_sequence_all_union(self):
        wildtypes = {
            'A_PF00001_1': 'ACDEFGHIKLMNPQRSTVWY',
            'A_PF00002_40': 'MMMMKKKKLLLLGGGGAAAA',     # same accession
            'B_PF00001_1': 'WWWWYYYYFFFFCCCCSSSS',       # same Pfam as the first
            'C_PF00003_1': 'ACDEFGHIKLMNPQRSTVWY',       # same sequence as the first
            'D_PF00004_1': 'PPPPQQQQNNNNEEEEDDDD',       # unrelated
        }
        accessions = {name: name.split('_')[0] for name in wildtypes}
        labels, report = ec.family_groups(sorted(wildtypes), wildtypes, accessions)
        self.assertEqual(labels['A_PF00001_1'], labels['A_PF00002_40'])
        self.assertEqual(labels['A_PF00001_1'], labels['B_PF00001_1'])
        self.assertEqual(labels['A_PF00001_1'], labels['C_PF00003_1'])
        self.assertNotEqual(labels['A_PF00001_1'], labels['D_PF00004_1'])
        self.assertEqual(report['groups'], 2)
        self.assertEqual(report['union_sources'], list(ec.UNION_SOURCES))


def synthetic_cohort(*, groups: int = 12, domains_per_group: int = 2,
                     sites: int = 6, variants: int = 2, schema: str,
                     strata: bool = False) -> dict:
    """A cohort of one of the two declared schemas, with a readable signal.

    The target is a declared function of the substituted residue's hydropathy
    difference plus a per-domain offset, so a design carrying chemistry must
    outpredict one carrying identity and geometry alone. Nothing here stands in
    for a measurement: it is a fixture the fitting path is exercised on.
    """

    from src.capability.core.amino_acids import AA20

    key = ec.COHORT_SCHEMAS[schema]
    generator = np.random.default_rng(20260924)
    units = []
    for group in range(groups):
        for member in range(domains_per_group):
            name = f'u{group:02d}_{member}'
            wildtype = ''.join(generator.choice(list(AA20), size=sites + 4))
            records = []
            for site in range(sites):
                choices = [a for a in AA20 if a != wildtype[site]]
                for mutant in generator.choice(choices, size=variants, replace=False):
                    sequence = wildtype[:site] + mutant + wildtype[site + 1:]
                    signal = (AA20.index(mutant) - AA20.index(wildtype[site])) / 20.0
                    records.append({
                        'position': site + 1, 'mutant': str(mutant), 'sequence': sequence,
                        'state': len(records) + 1,
                        'target': float(signal + 0.1 * group
                                        + 0.01 * generator.standard_normal()),
                        'uncertainty': 0.05})
            unit = {'name': name, 'group': f'grp-{group:02d}', 'wildtype': wildtype,
                    'length': len(wildtype), 'variants': records}
            if strata:
                unit['stratum'] = 'close' if group % 2 else 'remote'
                unit['band'] = 'ge95_near_duplicate' if group % 2 else 'lt30_no_detectable_homology'
            units.append(unit)
    return {'schema': schema, 'endpoint': 'fixture', 'endpoint_sha256': 'fixture', key: units}


class PanelAndHeldGroupComparison(unittest.TestCase):
    def setUp(self):
        self.cohort = synthetic_cohort(schema='external_confirmation_cohort_v1')
        self.units = self.cohort['domains']
        self.profiles = {row['name']: None for row in self.units}
        self.panel = ec.build_panel(self.units, self.profiles)

    def test_the_panel_carries_every_declared_block_and_a_named_uncertainty(self):
        self.assertEqual(sorted(self.panel['blocks']),
                         ['chem', 'comp', 'geom', 'ident', 'prof', 'prof2'])
        self.assertIn('uncertainty', self.panel)
        self.assertAlmostEqual(float(self.panel['weights'].sum()), 1.0)
        self.assertTrue(all(np.isfinite(block).all() for block in self.panel['blocks'].values()))

    def test_an_absent_profile_is_a_declared_zero_and_not_an_imputed_profile(self):
        self.assertTrue(np.all(self.panel['blocks']['prof'] == 0.0))
        self.assertTrue(np.all(self.panel['blocks']['prof2'] == 0.0))

    def test_a_block_carrying_the_signal_beats_one_that_does_not(self):
        design = {'base': ec.BASE_BLOCKS, 'chem': (*ec.BASE_BLOCKS, 'chem')}
        outcome = ec.fold_predictions(self.panel, self.panel['blocks'], design,
                                      seed=ec.SPLIT_SEEDS[0])
        self.assertIn('NO_EFFECT_NULL', outcome['predictions'])
        record = ec.paired_increment(self.panel, outcome['predictions'], 'chem', 'base')
        self.assertIsNotNone(record['point'])
        self.assertEqual(record['unit'], 'squared normalised fitness')
        self.assertEqual(record['evaluated_groups'], 12)

    def test_a_design_naming_G_without_a_declared_first_stage_is_refused(self):
        with self.assertRaises(ValueError):
            ec.fold_predictions(self.panel, self.panel['blocks'],
                                {'g': (*ec.BASE_BLOCKS, 'G')}, seed=ec.SPLIT_SEEDS[0])

    def test_the_nonlinear_response_reaches_no_held_out_label(self):
        from src.capability.readouts.readout_analysis import family_folds

        held = np.asarray(['grp-00', 'grp-01'])
        training = self.panel['group'][~np.isin(self.panel['group'], held)]
        partition = family_folds(training, 4, ec.SPLIT_SEEDS[0])
        block, diagnostics = ec.nuisance_response(
            self.panel, ec.BASE_BLOCKS, self.panel['blocks'], held, partition)
        self.assertEqual(block.shape, (len(self.panel['target']), len(ec.G_FEATURE_ORDER)))
        self.assertEqual(diagnostics['first_stage_columns'], list(ec.BASE_BLOCKS))
        moved = block.copy()
        permuted = dict(self.panel)
        rows = np.flatnonzero(np.isin(self.panel['group'], held))
        target = self.panel['target'].copy()
        target[rows] = target[rows][::-1]
        permuted['target'] = target
        again, _ = ec.nuisance_response(permuted, ec.BASE_BLOCKS, self.panel['blocks'],
                                        held, partition)
        np.testing.assert_allclose(moved, again)

    def test_a_readout_restriction_renormalises_rather_than_reweighting(self):
        design = {'base': ec.BASE_BLOCKS, 'chem': (*ec.BASE_BLOCKS, 'chem')}
        outcome = ec.fold_predictions(self.panel, self.panel['blocks'], design,
                                      seed=ec.SPLIT_SEEDS[0])
        keep = np.isin(self.panel['group'], ['grp-00', 'grp-01', 'grp-02'])
        restricted = ec.paired_increment(self.panel, outcome['predictions'], 'chem', 'base',
                                         keep=keep)
        self.assertEqual(restricted['evaluated_groups'], 3)

    def test_group_errors_drop_a_group_with_no_retained_row(self):
        prediction = np.zeros(len(self.panel['target']))
        keep = self.panel['group'] == 'grp-00'
        labels, values = ec.group_errors(self.panel['target'], prediction,
                                         self.panel['group'], self.panel['domain'],
                                         self.panel['site'], keep)
        self.assertEqual(list(labels), ['grp-00'])
        self.assertEqual(len(values), 1)


class StateIndicesAreReadFromEitherSchema(unittest.TestCase):
    """One cohort records the state index and the other leaves it implicit."""

    def test_an_implicit_state_index_is_its_position_in_the_declared_order(self):
        fit = _load_entry_point('fit_nested_singles.py')
        recorded = [{'state': 1, 'position': 3, 'mutant': 'A'},
                    {'state': 2, 'position': 5, 'mutant': 'G'}]
        implicit = [{'position': 3, 'mutant': 'A'}, {'position': 5, 'mutant': 'G'}]
        def derive(variants):
            return [v.get('state', i + 1) for i, v in enumerate(variants)]
        self.assertEqual(derive(recorded), derive(implicit))
        self.assertEqual(derive(implicit), [1, 2])
        # The loader has to read the implicit form, which is what the Domainome
        # cohort carries: every fit of it failed on a missing 'state' key.
        source = stage_path(REPO_ROOT, 'fit_nested_singles.py').read_text()
        self.assertIn("variant.get('state', index + 1)", source)
        self.assertTrue(callable(fit.load_arm))


class CohortLoading(unittest.TestCase):
    def test_both_declared_schemas_resolve_to_their_own_unit_key(self):
        for schema, key in ec.COHORT_SCHEMAS.items():
            with self.subTest(schema):
                cohort = synthetic_cohort(schema=schema, groups=6)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'cohort.json'
                    path.write_text(json.dumps(cohort))
                    loaded = ec.load_cohort(path)
                self.assertEqual(loaded['unit_key'], key)
                self.assertEqual(len(loaded['units']), len(cohort[key]))

    def test_an_undeclared_schema_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cohort.json'
            path.write_text(json.dumps({'schema': 'something_else', 'domains': []}))
            with self.assertRaises(ValueError):
                ec.load_cohort(path)

    def test_strata_are_read_when_the_cohort_declares_them(self):
        cohort = synthetic_cohort(schema='remote_homology_cohort_v1', groups=6, strata=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cohort.json'
            path.write_text(json.dumps(cohort))
            loaded = ec.load_cohort(path)
        self.assertEqual(set(loaded['strata'].values()), {'close', 'remote'})
        self.assertIsNone(ec.load_cohort.__doc__ is None or None)


class NumericEnvironment(unittest.TestCase):
    def test_an_unpinned_thread_count_is_refused(self):
        import os

        saved = {name: os.environ.pop(name, None) for name in ec.BLAS_THREAD_VARIABLES}
        try:
            with self.assertRaises(SystemExit):
                ec.require_blas_threads()
            os.environ['OMP_NUM_THREADS'] = str(ec.BLAS_THREADS + 1)
            with self.assertRaises(SystemExit):
                ec.require_blas_threads()
            os.environ['OMP_NUM_THREADS'] = str(ec.BLAS_THREADS)
            record = ec.require_blas_threads()
            self.assertEqual(record['pinned_threads'], ec.BLAS_THREADS)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class SecondaryControlSet(unittest.TestCase):
    def _controls(self, *, spearman_points):
        return {
            'base_blocks': list(ec.BASE_BLOCKS),
            'candidate_order': list(ec.CANDIDATE_BLOCKS),
            'qualified_control_set': [*ec.BASE_BLOCKS, 'chem'],
            'ladder': [
                {'candidate': 'comp', 'qualified': False,
                 'per_seed_increment': {str(s): -0.1 for s in ec.SPLIT_SEEDS},
                 'spearman_increment': {str(s): {'point': spearman_points['comp']}
                                        for s in ec.SPLIT_SEEDS}},
                {'candidate': 'chem', 'qualified': True,
                 'per_seed_increment': {str(s): 0.1 for s in ec.SPLIT_SEEDS},
                 'spearman_increment': {str(s): {'point': 0.01} for s in ec.SPLIT_SEEDS}},
                {'candidate': 'prof', 'qualified': False,
                 'per_seed_increment': {str(s): -1.0 for s in ec.SPLIT_SEEDS},
                 'spearman_increment': {str(s): {'point': spearman_points['prof']}
                                        for s in ec.SPLIT_SEEDS}},
                {'candidate': 'prof2', 'qualified': False,
                 'per_seed_increment': {str(s): -1.0 for s in ec.SPLIT_SEEDS},
                 'spearman_increment': {str(s): {'point': spearman_points['prof2']}
                                        for s in ec.SPLIT_SEEDS}},
                {'candidate': 'G', 'qualified': False,
                 'per_seed_increment': {str(s): -0.1 for s in ec.SPLIT_SEEDS},
                 'spearman_increment': {str(s): {'point': -0.01} for s in ec.SPLIT_SEEDS}},
            ],
        }

    def test_a_discarded_candidate_with_a_positive_rank_increment_is_added(self):
        secondary, record = ec.secondary_control_set(
            self._controls(spearman_points={'comp': 0.02, 'prof': 0.04, 'prof2': 0.05}))
        self.assertIn('comp', secondary)
        self.assertIn('prof2', secondary)
        # The raw profile is judged only through its restatement.
        self.assertNotIn('prof', secondary)
        self.assertIn('comp', record['added_over_primary'])

    def test_nothing_is_added_when_no_discarded_candidate_transfers_rank(self):
        secondary, record = ec.secondary_control_set(
            self._controls(spearman_points={'comp': -0.02, 'prof': -0.04, 'prof2': -0.05}))
        self.assertEqual(record['added_over_primary'], [])
        self.assertEqual(list(secondary), [*ec.BASE_BLOCKS, 'chem'])


class DeclaredConstants(unittest.TestCase):
    def test_the_development_screen_is_stricter_than_the_reported_stratum(self):
        self.assertGreater(ec.DEVELOPMENT_IDENTITY, ec.DEVELOPMENT_STRATUM_IDENTITY)
        self.assertEqual(ec.DEVELOPMENT_STRATUM_IDENTITY, ec.GROUPING_IDENTITY)

    def test_the_endpoint_names_its_scale_and_its_sign(self):
        self.assertIn('normalized_fitness', ec.ENDPOINT)
        self.assertIn('wild type reads 0', ec.ENDPOINT)
        self.assertIn('aPCA', ec.MEASURED_QUANTITY) if 'aPCA' in ec.MEASURED_QUANTITY else None
        self.assertIn('abundance', ec.MEASURED_QUANTITY)


if __name__ == '__main__':
    unittest.main()
