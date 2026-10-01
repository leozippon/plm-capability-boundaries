"""Statistical invariants for the profile-increment follow-up."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.capability.context.profile_increment import common_vector_support, context_rescue, family_regression, profile_increment
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS


COVARIATES=['log10_neff','max_identity_over_query','profile_entropy','log10_length']


def family_rows(n_families,seed=7):
    """``n_families`` family means with generic, non-collinear covariates."""
    rng=np.random.default_rng(seed)
    return [dict(cluster=f'F{index}',model_minus_lookup=float(rng.normal()),
                 **{name:float(rng.normal()) for name in COVARIATES})
            for index in range(n_families)]


class FamilyRegressionRefusalTests(unittest.TestCase):
    """The coefficient intervals are refused, not widened, where they cannot hold.

    Both conditions were absent until 2026-09-24: the function published a 95%
    family-bootstrap percentile interval on every coefficient at any family
    count, and silently discarded the resampled designs it could not fit. The
    recorded analysis ran at 163 to 174 families with all 2,000 draws
    identifiable, so no published interval changes; these are the paths that
    were unguarded.
    """

    def test_a_family_count_below_the_shared_floor_is_refused(self):
        for n_families in range(3,MINIMUM_BOOTSTRAP_UNITS):
            with self.subTest(n_families=n_families):
                with self.assertRaisesRegex(ValueError,'below the 8-unit floor'):
                    family_regression(family_rows(n_families),COVARIATES,bootstrap=200,seed=20260923)

    def test_a_family_count_that_cannot_identify_the_design_is_refused(self):
        # Eight families clear the unit floor and still leave most resampled
        # designs rank deficient, so the percentile would be taken over the
        # draws that happened to be identifiable. Measured at 2,000 draws: 819
        # identifiable at eight families and 1,737 at ten, against the 1,900 the
        # shared fraction requires.
        for n_families in (8,10):
            with self.subTest(n_families=n_families):
                with self.assertRaisesRegex(ValueError,'identifiable'):
                    family_regression(family_rows(n_families),COVARIATES,bootstrap=2000,seed=20260923)

    def test_a_resolvable_cohort_publishes_every_coefficient_interval(self):
        result=family_regression(family_rows(24),COVARIATES,bootstrap=2000,seed=20260923)
        self.assertEqual(result['n_families'],24)
        self.assertEqual(result['minimum_families'],MINIMUM_BOOTSTRAP_UNITS)
        self.assertEqual(result['bootstrap_success'],result['bootstrap_requested'])
        self.assertEqual(len(result['coefficients']),len(COVARIATES)+2)
        for name,block in result['coefficients'].items():
            low,high=block['interval']
            self.assertLessEqual(low,block['point'],name)
            self.assertLessEqual(block['point'],high,name)


class ProfileIncrementTests(unittest.TestCase):
    def assays(self):
        rng=np.random.default_rng(6)
        rows=[]
        for family in range(6):
            for assay in range(2):
                p=rng.normal(size=40);m=rng.normal(size=40);y=p+2*m+rng.normal(size=40)*.1
                rows.append(dict(assay=f'{family}_{assay}',cluster=family,
                    mutants=[f'A{i+1}C' for i in range(40)],profile_scores=p.tolist(),
                    measured=y.tolist(),scores={'no_context':m.tolist()}))
        return rows

    def test_profile_copy_has_no_residual_and_no_cv_increment(self):
        rows=self.assays()
        for row in rows:row['scores']['no_context']=row['profile_scores']
        result=profile_increment(rows,condition='no_context',bootstrap=20)
        self.assertTrue(all(r['partial_rank_correlation'] is None for r in result['assays']))
        self.assertAlmostEqual(result['summaries']['cv_delta_spearman']['point'],0)

    def test_held_family_labels_do_not_change_its_predictor(self):
        rows=self.assays();changed=copy.deepcopy(rows)
        for row in changed:
            if row['cluster']==0:row['measured']=row['measured'][::-1]
        a=profile_increment(rows,condition='no_context',bootstrap=20)
        b=profile_increment(changed,condition='no_context',bootstrap=20)
        np.testing.assert_allclose(a['folds'][0]['profile_model_coefficients'],b['folds'][0]['profile_model_coefficients'],atol=1e-12)
        self.assertGreater(a['summaries']['cv_delta_spearman']['point'],.3)

    def test_family_duplication_preserves_training_weight(self):
        rows=self.assays();duplicated=copy.deepcopy(rows)
        for row in copy.deepcopy(rows):
            if row['cluster']==1:row['assay']+='copy';duplicated.append(row)
        a=profile_increment(rows,condition='no_context',bootstrap=20)
        b=profile_increment(duplicated,condition='no_context',bootstrap=20)
        np.testing.assert_allclose(a['folds'][0]['profile_model_coefficients'],b['folds'][0]['profile_model_coefficients'],atol=1e-12)

    def test_nonfinite_and_unaligned_vectors_fail(self):
        rows=self.assays();rows[0]['measured'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            profile_increment(rows,condition='no_context',bootstrap=20)
        rows=self.assays();rows[0]['mutants'].pop()
        with self.assertRaisesRegex(ValueError,'unaligned'):
            profile_increment(rows,condition='no_context',bootstrap=20)

    def test_context_contrast_sign_and_sparse_stratum(self):
        rows=self.assays()
        for row in rows:
            row['scores'].update(homolog=row['measured'],unrelated=(-np.array(row['measured'])).tolist())
            row.update(wt_log_likelihood={'homolog':-20,'unrelated':-30},wt_scored_tokens=10,
                       homolog_identity=40 if row['cluster']==0 else 60,max_lcs=5)
        result=context_rescue(rows,bootstrap=20)
        self.assertAlmostEqual(result['strata']['pooled']['metrics']['delta_spearman']['point'],2)
        self.assertAlmostEqual(result['strata']['pooled']['metrics']['wt_delta_loglikelihood_per_token']['point'],1)
        sparse=result['strata']['identity_below_50_percent']
        self.assertEqual(sparse['n_families'],1)
        self.assertIsNone(sparse['metrics']['delta_spearman']['interval'])

    def test_context_secondary_contrasts_distinguish_unrelated_harm(self):
        rows=self.assays()
        for row in rows:
            row['scores'].update(no_context=row['measured'],homolog=row['measured'],
                                 unrelated=(-np.array(row['measured'])).tolist())
            row.update(wt_log_likelihood={'homolog':-20,'unrelated':-30},wt_scored_tokens=10,
                       homolog_identity=40,max_lcs=5)
        result=context_rescue(rows,bootstrap=20)
        metrics=result['strata']['pooled']['metrics']
        self.assertAlmostEqual(metrics['delta_spearman']['point'],2)
        self.assertAlmostEqual(metrics['homolog_minus_no_context_spearman']['point'],0)
        self.assertAlmostEqual(metrics['unrelated_minus_no_context_spearman']['point'],-2)
        for row in rows:row['scores']['no_context']=[0]*len(row['mutants'])
        undefined=context_rescue(rows,bootstrap=20)['strata']['pooled']['metrics']
        for key in result['secondary_contrasts']:
            self.assertIsNone(undefined[key]['point'])
            self.assertIsNone(undefined[key]['interval'])
            self.assertEqual(undefined[key]['excluded_assays'],len(rows))
        self.assertAlmostEqual(undefined['delta_spearman']['point'],2)

    def vector_sources(self):
        rows=self.assays()
        for row in rows:
            row.update(wildtype_id=row['assay'],mutant_digest='fixed-draw',
                       original_mutant_digest='original-draw')
        return [dict(arm=f'model{i}',assays=copy.deepcopy(rows[i:])) for i in range(4)]

    def test_common_support_counts_and_preserves_native_rows(self):
        sources=self.vector_sources()
        original=copy.deepcopy(sources)
        panels,support=common_vector_support(sources)
        self.assertEqual(support['n_assays'],9)
        self.assertEqual(support['n_families'],5)
        self.assertEqual(support['n_variants'],360)
        self.assertEqual(support['per_arm']['model0']['native']['n_assays'],12)
        self.assertEqual(len(support['per_arm']['model0']['excluded_assay_ids']),3)
        self.assertEqual([r['assay'] for r in panels[0]],[r['assay'] for r in panels[-1]])
        self.assertEqual(sources,original)

    def test_common_support_rejects_mutation_order_and_label_mismatch(self):
        for field in ('mutants','measured','profile_scores'):
            sources=self.vector_sources()
            sources[-1]['assays'][0][field].reverse()
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'mismatch'):
                common_vector_support(sources)

    def test_empty_common_support_is_explicit(self):
        sources=self.vector_sources()[:2]
        sources[0]['assays']=sources[0]['assays'][:1]
        panels,support=common_vector_support(sources)
        self.assertEqual(panels,[[],[]])
        self.assertEqual((support['n_assays'],support['n_families'],support['n_variants']),(0,0,0))

    def test_cli_retains_native_and_common_support(self):
        sources=self.vector_sources()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            command=[sys.executable,'scripts/capability/context/analyse_profile_increment.py',
                     '--input-root',folder,'--out',str(root/'out'),'--vectors-only',
                     '--common-support','--bootstrap','20']
            for i,source in enumerate(sources):
                source['status']='complete'
                for row in source['assays']:
                    row['scores'].update(homolog=row['measured'],unrelated=row['profile_scores'])
                    row.update(wt_log_likelihood={'homolog':-20,'unrelated':-30},
                               wt_scored_tokens=10,homolog_identity=40,max_lcs=5)
                path=root/f'{i}.json'
                path.write_text(json.dumps(source))
                command.extend(['--vectors',str(path)])
            subprocess.run(command,check=True,capture_output=True,text=True)
            result=json.loads((root/'out'/'profile_increment.json').read_text())
            self.assertEqual([r['support']['n_assays'] for r in result['profile_increments']],[12,11,10,9])
            common=result['common_support']
            self.assertEqual(common['status'],'complete')
            self.assertEqual([r['support']['n_variants'] for r in common['profile_increments']],[360]*4)
            self.assertEqual(len(result['source_sha256']),4)


if __name__=='__main__':unittest.main()
