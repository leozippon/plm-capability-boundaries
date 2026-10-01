"""Four invariants of fresh extraction sharding; no models or fits required."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from scripts.capability.readouts.extract_frozen_readout import partition_assays, sha, FEATURE_NAMES
from scripts.capability.readouts.merge_readout_shards import merge_shards
from scripts.capability.readouts.analyse_readout import load_panel


class ShardTests(unittest.TestCase):
    def fixture(self, root):
        rows = [dict(assay=f'a{i}',cluster=i,wildtype_id=f'w{i}',wildtype='ACD',
                     mutants=['A1C','C2A','D3A'],mutant_digest='draw',measured=[1.,2.,3.],
                     profile_scores=[2.,1.,3.]) for i in range(5)]
        cohort=root/'cohort.json';cohort.write_text(json.dumps(dict(assays=rows)))
        identity=dict(arm='test',cohort_sha256=sha(cohort),feature_names=list(FEATURE_NAMES),
                      batch_size=1,budget=1024,smoke_variants=0,assay_limit=0,smoke_length_strata=False,
                      max_score_drift=.001,max_feature_drift=.001,code_sha256={'extractor':'fresh'})
        universe=[dict(assay=r['assay'],max_packed_tokens=100+i*20,variants=3) for i,r in enumerate(rows)]
        assignments=partition_assays(universe,2);paths=[]
        for index, assigned in enumerate(assignments):
            directory=root/f'shard{index}';directory.mkdir();records=[]
            for assay in assigned:
                row=next(r for r in rows if r['assay']==assay);artifact=directory/f'{assay}.npz'
                np.savez(artifact,features=np.ones((3,4,6)),likelihood=np.arange(3.),
                         wt_features=np.ones((4,6)),wt_likelihood=1.,mutants=row['mutants'],
                         measured=row['measured'],profile_scores=row['profile_scores'],
                         metadata=json.dumps(dict(identity=identity,assay=assay,mutant_digest='draw')),
                         batch_check_likelihood_delta_nats=0.,batch_check_feature_relative_l2=0.,
                         batch_check_mutation_feature_relative_l2=0.)
                records.append(dict(assay=assay,cluster=row['cluster'],wildtype_id=row['wildtype_id'],
                                    mutant_digest='draw',variants=3,file=artifact.name,sha256=sha(artifact),
                                    max_packed_tokens=next(r['max_packed_tokens'] for r in universe if r['assay']==assay)))
            path=directory/f'shard_test_{index}.json';paths.append(path)
            path.write_text(json.dumps(dict(status='complete',identity=identity,block_indices=[1,3],
                batch_size=1,assays=records,skipped=[],execution_partition=dict(method='greedy_packed_token_work_v1',
                count=2,index=index,universe=universe,assigned_assays=assigned))))
        return cohort,paths

    def test_partition_deterministic_disjoint_and_label_independent(self):
        universe=[dict(assay=str(i),max_packed_tokens=i+1,variants=128) for i in range(11)]
        parts=partition_assays(universe,3)
        self.assertEqual(sorted(sum(parts,[])),sorted(r['assay'] for r in universe))
        changed=[dict(r,measured=[1e20]) for r in universe]
        self.assertEqual(parts,partition_assays(changed,3))
        reversed_parts=partition_assays(universe[::-1],3)
        self.assertEqual([set(p) for p in parts],[set(p) for p in reversed_parts])
        for count in (0,12):
            with self.assertRaises(ValueError):partition_assays(universe,count)

    def test_incomplete_overlap_and_wrong_universe_rejected(self):
        for mutation in ('missing','running','duplicate_index','missing_assay','wrong_universe'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cohort,paths=self.fixture(root)
                obj=json.loads(paths[1].read_text())
                if mutation=='missing':paths=paths[:1]
                elif mutation=='running':obj['status']='running'
                elif mutation=='duplicate_index':obj['execution_partition']['index']=0
                elif mutation=='missing_assay':obj['assays'].pop()
                else:obj['execution_partition']['universe'].pop()
                if len(paths)>1:paths[1].write_text(json.dumps(obj))
                with self.assertRaises(ValueError):merge_shards(cohort,paths,root/'merged')
                self.assertFalse((root/'merged/manifest_test.json').exists())

    def test_identity_checksum_order_and_precision_rejected(self):
        for mutation in ('identity','checksum','order','precision'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cohort,paths=self.fixture(root);obj=json.loads(paths[0].read_text())
                artifact=paths[0].parent/obj['assays'][0]['file']
                if mutation=='identity':obj['identity']['code_sha256']['extractor']='old'
                elif mutation=='checksum':artifact.write_bytes(b'corrupt')
                else:
                    with np.load(artifact,allow_pickle=False) as saved:values={k:saved[k] for k in saved.files}
                    if mutation=='order':values['mutants']=values['mutants'][::-1]
                    else:values['batch_check_mutation_feature_relative_l2']=.002
                    np.savez(artifact,**values);obj['assays'][0]['sha256']=sha(artifact)
                paths[0].write_text(json.dumps(obj))
                with self.assertRaises(ValueError):merge_shards(cohort,paths,root/'merged')

    def test_merge_is_accepted_by_existing_analysis_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cohort,paths=self.fixture(root)
            merged=merge_shards(cohort,paths[::-1],root/'merged')
            rows,_,_=load_panel(cohort,[merged],'test','common',20260923,4)
            self.assertEqual([r['assay'] for r in rows],[f'a{i}' for i in range(5)])
            self.assertEqual(rows[0]['R'].shape,(3,16))
            payload=json.loads(merged.read_text())
            self.assertEqual(len(payload['execution_merge']['sources']),2)
            self.assertNotIn('execution_partition',payload['identity'])
            with self.assertRaises(ValueError):merge_shards(cohort,paths,root/'merged')


if __name__=='__main__':unittest.main()
