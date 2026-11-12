"""Merge the earlier 8,390-query DIAMOND hit table with the search over the 4,250 added rows (+300 earlier rows as controls).
Hits are keyed by sequence digest; the controls must reproduce the earlier best hit exactly, or the merge refuses."""
import json, pandas as pd, sys
H='q subject pident alnlen nident qstart qend qlen slen evalue bitscore'.split()
def load(hits,table_ids):
    h=pd.read_csv(hits,sep='\t',header=None,names=H); ids=sorted(table_ids); h['digest']=h.q.map(lambda q:ids[int(q[1:])]); return h
old_t=pd.read_csv('../input/checkpoint_structures_12of68.tsv',sep='\t',comment='#'); new_t=pd.read_csv('input/checkpoint_structures_18of68.tsv',sep='\t',comment='#')
ids_old=list(old_t.sequence_sha256); ids_q=[json.loads(l)['id'] for l in open('natural_neighbour/query_cohort.jsonl')]
old=load('../natural_neighbour/work/hits.tsv',ids_old); new=load('natural_neighbour/work/hits.tsv',ids_q)
ctrl=set(json.load(open('natural_neighbour/control_digests.json')))
def best(h): 
    h=h.assign(idq=100*h.nident/h.qlen); return h.sort_values(['digest','idq','bitscore'],ascending=[True,False,False]).drop_duplicates('digest').set_index('digest')
bo,bn=best(old[old.digest.isin(ctrl)]),best(new[new.digest.isin(ctrl)])
common=bo.index.union(bn.index); a=bo.reindex(common); b=bn.reindex(common)
bad=((a.idq.fillna(-1)-b.idq.fillna(-1)).abs()>1e-9).sum(); badsub=(a.subject.fillna('')!=b.subject.fillna('')).sum()
print('controls',len(ctrl),'with hit old/new',len(bo),len(bn),'identity mismatches',int(bad),'subject mismatches',int(badsub))
if bad: sys.exit('REFUSING: controls do not reproduce the earlier best hit')
add=new[~new.digest.isin(ctrl)]; assert not set(add.digest)&set(old.digest)
allh=pd.concat([old,add]); ids=sorted(new_t.sequence_sha256); pos={d:i for i,d in enumerate(ids)}
allh['q']=allh.digest.map(lambda d:'q%05d'%pos[d]); assert allh.q.notna().all()
allh[H].to_csv('natural_neighbour/merged_hits.tsv',sep='\t',header=False,index=False); print('merged hit rows',len(allh),'queries with hits',allh.digest.nunique())
