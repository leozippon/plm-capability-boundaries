"""Leakage, weighting, exact alignment, and linear-solver invariants."""
import unittest
import json
import hashlib
import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import torch

from src.capability.readouts.readout_analysis import evaluate_readouts, family_folds, nested_predict, ridge_predict, row_weights, sequence_features


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_ridge_matches_weighted_closed_form(self):
        rng=np.random.default_rng(4); x=rng.normal(size=(30,5)); y=rng.normal(size=30)
        w=np.arange(1,31,dtype=float);w/=w.sum(); xt=rng.normal(size=(10,5))
        mean=w@x;scale=np.sqrt(w@((x-mean)**2));z=(x-mean)/scale
        ym=w@y; beta=np.linalg.solve(z.T@(w[:,None]*z)+.1*np.eye(5),z.T@(w*(y-ym)))
        pred=ridge_predict(x,y,w,xt,[.1])[:,0]
        np.testing.assert_allclose(pred,(xt-mean)/scale@beta+ym,atol=1e-12)
        # An extreme held-out feature cannot influence another held-out prediction.
        changed=xt.copy(); changed[-1]=1e9
        np.testing.assert_allclose(ridge_predict(x,y,w,changed,[.1])[:-1,0],pred[:-1],atol=1e-12)

    def test_held_family_labels_cannot_change_its_predictions_or_tuning(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(100,4));y=x[:,0]+rng.normal(size=100)*.05
        families=np.repeat(np.arange(10),10);assays=families.copy()
        a,fa=nested_predict(x,y,assays,families)
        json.dumps(fa, allow_nan=False)
        held=fa[0]['held_families'];changed=y.copy();idx=np.isin(families,held);changed[idx]=rng.normal(size=idx.sum())
        b,fb=nested_predict(x,changed,assays,families)
        np.testing.assert_allclose(a[idx],b[idx],atol=1e-12)
        self.assertEqual(fa[0],fb[0])
        self.assertGreater(np.corrcoef(a,y)[0,1],.85)
        for outer in fa:
            self.assertFalse(set(outer['held_families']) & set(outer['training_families']))
            for inner in outer['inner_folds']:
                self.assertFalse(set(inner['validation_families']) & set(outer['held_families']))

    def test_weights_equalize_families_assays_and_reject_crossing_assays(self):
        a=np.array(['a','a','b','c','c','c']);f=np.array([0,0,0,1,1,1]);w=row_weights(a,f)
        self.assertAlmostEqual(w[f==0].sum(),.5);self.assertAlmostEqual(w[a=='a'].sum(),.25)
        with self.assertRaisesRegex(ValueError,'multiple families'):row_weights(np.array(['a','a']),np.array([0,1]))
        with self.assertRaisesRegex(ValueError,'at least'):family_folds([1,2],5,1)

    def test_sequence_features_and_invalid_mutations(self):
        x=sequence_features('ACD',['A1C','A1C:D3A'])
        self.assertEqual(x.shape,(2,444));self.assertEqual(x[1,:400].sum(),2)
        self.assertAlmostEqual(x[1,400],2/3);self.assertAlmostEqual(x[1,401],1/3)
        self.assertEqual(x[1,402],2);self.assertAlmostEqual(x[1,-1],3/1024)
        for mutant in ['D1C','A1C:A1D','A4C','A1X']:
            with self.assertRaises(ValueError):sequence_features('ACD',[mutant])

    def test_synthetic_readout_recovers_signal_and_report_serializes(self):
        rng=np.random.default_rng(10);rows=[]
        for family in range(10):
            r=rng.normal(size=(12,2));m=rng.normal(size=12);p=rng.normal(size=12)
            rows.append(dict(assay=str(family),cluster=family,mutants=[f'A{i+1}C' for i in range(12)],
                             measured=r[:,0],P=p,M=m,S=rng.normal(size=(12,2)),R=r))
        result=evaluate_readouts(rows,bootstrap=20)
        json.dumps(result,allow_nan=False)
        self.assertGreater(result['summaries']['delta_spearman']['point'],.5)
        self.assertEqual(result['n_variants'],120)
        self.assertGreater(result['summaries']['R_minus_raw_M_spearman']['point'],.5)
        self.assertGreater(result['summaries']['B_R_minus_raw_M_spearman']['point'],.5)
        self.assertIn('permuted_B_R',result['folds'])

    def test_artifact_identity_and_exact_row_alignment(self):
        spec=importlib.util.spec_from_file_location('analyse_readout',Path(__file__).resolve().parents[2]/'scripts/capability/readouts/analyse_readout.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cohort=root/'cohort.json';artifact=root/'features.npz';manifest=root/'manifest.json'
            row=dict(assay='a',cluster=1,wildtype_id='wt',wildtype='ACD',mutant_digest='draw',
                     mutants=['A1C','C2A','D3A'],measured=[1.,2.,3.],profile_scores=[3.,1.,2.])
            cohort.write_text(json.dumps(dict(assays=[row])))
            identity=dict(arm='test',cohort_sha256=hashlib.sha256(cohort.read_bytes()).hexdigest(),
                          feature_names=['middle_mean','middle_last','final_mean','final_last'])
            def save(mutants):
                np.savez(artifact,features=np.ones((3,4,6)),likelihood=np.arange(3.),mutants=mutants,
                         measured=row['measured'],profile_scores=row['profile_scores'],
                         metadata=json.dumps(dict(identity=identity,assay='a',mutant_digest='draw')))
                manifest.write_text(json.dumps(dict(identity=identity,status='complete',assays=[dict(
                    assay='a',cluster=1,wildtype_id='wt',mutant_digest='draw',variants=3,file=artifact.name,
                    sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())])))
            save(row['mutants'])
            rows,_,_=module.load_panel(cohort,[manifest],'test','common',20260923,4)
            self.assertEqual(rows[0]['R'].shape,(3,16));self.assertEqual(rows[0]['S'].shape,(3,444))
            save(row['mutants'][::-1])
            with self.assertRaisesRegex(ValueError,'mutation order'):
                module.load_panel(cohort,[manifest],'test','common',20260923,4)
            artifact.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                module.load_panel(cohort,[manifest],'test','common',20260923,4)

    def test_fold_seed_changes_only_splits_and_preserves_projected_features(self):
        rng=np.random.default_rng(15)
        projection=np.random.default_rng(20260923).normal(size=(5,2))/np.sqrt(2)
        rows=[]
        for family in range(10):
            for assay in range(2):
                hidden=rng.normal(size=(6,5));r=hidden@projection
                rows.append(dict(assay=f'{family}_{assay}',cluster=family,
                    mutants=[f'A{i+1}C' for i in range(6)],measured=r[:,0],
                    P=rng.normal(size=6),M=rng.normal(size=6),S=rng.normal(size=(6,2)),R=r))
        original=[r['R'].copy() for r in rows]
        reports=[evaluate_readouts(rows,seed=20260923,fold_seed=fold_seed,bootstrap=20,
                                  permutation_control=False) for fold_seed in (20260923,20260924)]
        self.assertNotEqual(reports[0]['folds']['B'],reports[1]['folds']['B'])
        for row,before in zip(rows,original):np.testing.assert_array_equal(row['R'],before)
        for report in reports:
            self.assertEqual(report['bootstrap_seed'],20260923)
            held=[f for fold in report['folds']['B'] for f in fold['held_families']]
            self.assertEqual(sorted(held),list(range(10)))
            for name,folds in report['folds'].items():
                self.assertEqual([f['held_families'] for f in folds],
                                 [f['held_families'] for f in report['folds']['B']])
                for fold in folds:
                    self.assertFalse(set(fold['held_families']) & set(fold['training_families']))
                    for inner in fold['inner_folds']:
                        self.assertTrue(set(inner['validation_families']) <= set(fold['training_families']))

    def test_nonfinite_fails(self):
        x=np.ones((10,2));x[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            nested_predict(x,np.arange(10),np.arange(10),np.arange(10))


if __name__=='__main__':unittest.main()
