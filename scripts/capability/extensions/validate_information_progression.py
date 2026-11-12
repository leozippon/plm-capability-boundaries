"""Read-only numerical acceptance; no fits, bootstrap draws, model inference or GPU access."""
import hashlib, json, os, platform, resource, shutil, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path('/Data/lzp/plm-capability-boundaries')
PROD = ROOT/'results/extensions/information_progression_20261006'
OUT = ROOT/'results/extensions/information_progression_20261006_validation'
START = time.monotonic()
def load(p):
    return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def identity(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x,dtype=np.float64).tobytes()).hexdigest()
def resource_receipt():
    return dict(device='cpu',python=platform.python_version(),numpy=np.__version__,disk_free_bytes=shutil.disk_usage(OUT).free,memory=Path('/proc/meminfo').read_text(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,blas_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']})
failures=[]; verified={}; unverified=[]; maximum_errors={}; before=resource_receipt()
def check(condition,message):
    if not condition: failures.append(message)
def hash_check(path,expected,kind):
    path=Path(path)
    if not path.exists():
        unverified.append(dict(path=str(path),kind=kind,reason='missing local source')); return
    actual=digest(path); check(actual==expected,f'{kind} hash mismatch: {path}')
    verified[kind]=verified.get(kind,0)+1

def compare(a,b,key,tol=2e-12):
    if a is None or b is None:
        check(a is None and b is None,f'{key}: undefined disagreement'); return
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    maximum_errors[key]=max(maximum_errors.get(key,0),err)
    check(err<=tol,f'{key}: difference {err}')

c=load(PROD/'contract.json'); completion=load(PROD/'completion.json')
check(completion['status']=='complete' and completion['tests_passed'] and completion['full_panel_fitted'],'completion status')
models=c['models']; seeds=c['seeds']; check(len(models)==33 and len(set(models))==33 and seeds==[20260923,20260924,20260925],'model/seed scope')
expected={(p,a,s) for p in ['full','structural'] for a in models for s in seeds}
cells=completion['completed_cells']
check(len(cells)==198 and {(r['panel'],r['arm'],r['seed']) for r in cells}==expected and completion['expected_cells']==198,'198 completion cells')
check(all(r['identity_verified'] and r['folds_verified'] for r in cells),'completion cell verification')
listed=set(completion['output_sha256']); actual={str(p.relative_to(PROD)) for p in PROD.rglob('*') if p.is_file()}-{'completion.json'}
check(listed==actual,'receipt output inventory')
for p,h in completion['output_sha256'].items(): hash_check(PROD/p,h,'production_output')
for p,h in c['code_sha256'].items(): hash_check(ROOT/p,h,'current_code')
prov=c['provenance']; bundle=ROOT/'results/R2/readout_expansion_20260923/complete'
admission=ROOT/'results/R5/local_context_20260923/20260923233257_e429ce7f31e4/lcgp_admission/local_context_measurement.json'
cohort_path=ROOT/'results/R5/context_mutation_rescue_20260923/cohort.json'
hash_check(admission,prov['admission_sha256'],'admission'); hash_check(cohort_path,prov['cohort_sha256'],'cohort')
hash_check(bundle/'admission/source_registry.json',prov['source_registry_sha256'],'source_registry')
for p,h in prov['profile_sources'].items(): hash_check(ROOT/'results/R5/retrieval_bound_20260807'/p,h,'profile_source')
sites=ROOT/'results/extensions/mutation_structure_20261006/structure/annotation/sites.json.gz'
hash_check(sites,prov['sites_sha256'],'sites'); hash_check(sites.parent/'coverage.json',prov['paired_coverage_sha256'],'coverage')
for p,h in prov['structural_source_sha256'].items(): hash_check(p,h,'structural_source')
original=load(PROD/'original-memberships.json')
def membership(records):
    return [[r['fold'],r['held_families'],r['training_families'],[i['validation_families'] for i in r['inner_folds']]] for r in records]
# Independently verify historical fold records and retained scalar provenance, never fit.
raw_by_model={}
for src in prov['rank_feature_sources']:
    hash_check(ROOT/src['path'],src['sha256'],'rank_source')
    report=load(ROOT/src['path'])
    check(report['arm']==src['arm'] and report['fold_seed']==src['seed'],'rank source model/seed')
    for records in report['folds'].values(): check(membership(records)==original[str(src['seed'])],'historical source fold disagreement')
    pred=report['predictions']
    ids=[(r['assay'],r['mutants']) for r in pred]
    vector=np.concatenate([r['raw_M'] for r in pred])
    check(np.isfinite(vector).all(),'raw M finite')
    if src['arm'] in raw_by_model:
        previous=raw_by_model[src['arm']]
        check(previous[0]==ids and np.array_equal(previous[1],vector),'raw M seed invariance')
    else: raw_by_model[src['arm']]=(ids,vector)
    for p,h in src['extraction_manifest_sha256'].items():
        relative='results/external_baseline/'+p.split('/results/external_baseline/',1)[1]
        hash_check(bundle/relative,h,'extraction_manifest')

cohort={r['assay']:r for r in load(cohort_path)['assays']}
ads=load(admission); selected_assays=sorted(ads['support']['assay_ids'])
full_samples=[]
for a in selected_assays:
    row=cohort[a]
    for j,m in enumerate(row['mutants']):
        sid=hashlib.sha256(json.dumps([prov['cohort_sha256'],a,m],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        full_samples.append(dict(assay=a,cluster=row['cluster'],mutation=m,variant_index=j,original_index=len(full_samples),sample_id=sid))
check(load(PROD/'full/samples.json')==full_samples,'full sample identities reconstructed from cohort')
# Independently reconstruct structural support from exact-site criteria.
import gzip, re
with gzip.open(sites,'rt') as f: site_rows=json.load(f)
coverage_sources={r['sha256']:r for r in load(sites.parent/'coverage.json')['sources']}
site_map={(r['assay_id'],r['wt_position']):r for r in site_rows}
check(len(site_map)==len(site_rows),'unique structural site identities')
structural_samples=[]
methods={'X-RAY DIFFRACTION','SOLUTION NMR','ELECTRON MICROSCOPY','SOLID-STATE NMR','NEUTRON DIFFRACTION','ELECTRON CRYSTALLOGRAPHY'}
# Actual admitted methods are checked against current source declaration below.
import ast
structure_module=ast.parse((ROOT/'src/capability/extensions/structure.py').read_text())
for node in structure_module.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXPERIMENTAL_METHODS' for t in node.targets):
        value=node.value
        methods=set(ast.literal_eval(value.args[0] if isinstance(value,ast.Call) else value))
for sample in full_samples:
    if ':' in sample['mutation']: continue
    match=re.fullmatch(r'([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])',sample['mutation'])
    check(match is not None,'strict mutation syntax')
    site=site_map.get((sample['assay'],int(match[2])))
    if not site: continue
    method=site.get('method') or coverage_sources[site['source_sha256']]['method'] if site.get('status')=='admitted' else ''
    if site['status']=='admitted' and site.get('mapping_kind') in ['full_entity','full_wt_fragment'] and method.upper() in methods and site['coordinate_present'] and site['rsa'] is not None:
        structural_samples.append(sample)
from collections import Counter
cnt=Counter(r['assay'] for r in structural_samples)
structural_samples=[r for r in structural_samples if cnt[r['assay']]>=3]
check(load(PROD/'structural/samples.json')==structural_samples,'structural support exact-site reconstruction')

panels={}; fold_count=0; prediction_vectors=0
for panel in ['full','structural']:
    folder=PROD/panel; samples=load(folder/'samples.json'); n=len(samples)
    check(n==c['panel_support'][panel]['rows'] and identity(samples)==c['panel_support'][panel]['support_sha256'],'support hash/count')
    check(len({(r['assay'],r['mutation']) for r in samples})==n and len({r['original_index'] for r in samples})==n,'unique panel identities')
    aids=np.array([r['assay'] for r in samples]); family=np.array([r['cluster'] for r in samples]); assay_names=sorted(set(aids)); families=sorted(set(family)); indices={a:np.flatnonzero(aids==a) for a in assay_names}
    y=np.array([cohort[r['assay']]['measured'][r['variant_index']] for r in samples]); target=np.empty(n)
    for a,ix in indices.items():
        r=rankdata(y[ix]); sd=np.std(r); target[ix]=(r-r.mean())/sd if sd>0 else 0
    projected=load(folder/'projected-memberships.json')
    for s in seeds:
        realized=[]
        for f,held,training,inner in original[str(s)]:
            tr=sorted(set(training)&set(families))
            realized.append([f,sorted(set(held)&set(families)),tr,[sorted(set(v)&set(tr)) for v in inner]])
        check(projected[str(s)]==realized,'projection from original folds')
        check(set(sum([r[1] for r in realized],[]))==set(families) and len(sum([r[1] for r in realized],[]))==len(families),'outer held coverage')
        for _,h,t,inn in realized:
            check(bool(h) and bool(t) and set(h).isdisjoint(t) and set(h)|set(t)==set(families),'outer partition')
            check(all(v and set(t)-set(v) for v in inn) and set(sum(inn,[]))==set(t) and len(sum(inn,[]))==len(t),'inner partition')
    registry=c['registry'][panel]; designs=list(registry['designs']); contrasts=registry['contrasts']; rows=load(folder/'assay-metrics.json')
    indexed={(r['arm'],r['seed'],r['assay']):r for r in rows}
    check(len(indexed)==len(rows)==33*3*len(assay_names),'assay metric complete unique grid')
    family_values={metric:np.empty((len(models),len(seeds),len(assay_names),len(designs))) for metric in ['spearman','rank_mse']}
    for mi,arm in enumerate(models):
        for si,s in enumerate(seeds):
            stem=f'{arm}-seed{s}'; checks=load(folder/(stem+'-checks.json'))
            with np.load(folder/(stem+'.npz'),allow_pickle=False) as z:
                check(set(z.files)==set(designs)|{'original_index','sample_id','target'},'archive key set')
                check(np.array_equal(z['original_index'],[r['original_index'] for r in samples]) and np.array_equal(z['sample_id'],[r['sample_id'] for r in samples]),'OOF row identities')
                check(np.array_equal(z['target'],target),'target exact independent reranking')
                predictions=np.column_stack([z[d] for d in designs]); check(predictions.shape==(n,len(designs)) and np.isfinite(predictions).all(),'OOF complete finite prediction vectors')
                for di,d in enumerate(designs):
                    ch=checks['designs'][d]; prediction_vectors+=1
                    check(ch['prediction_sha256']==array_hash(z[d]) and ch['target_sha256']==array_hash(target) and ch['support_sha256']==identity(samples) and ch['membership_sha256']==identity(projected[str(s)]) and ch['rows']==n,'cell design receipt identities/hash')
                    records=load(folder/(stem+'-'+d+'-folds.json')); fold_count+=len(records)
                    check(membership(records)==projected[str(s)],'realized all-design folds')
                    for r in records:
                        losses=np.array([v['rank_mse'] for v in r['inner_folds']]); check(losses.shape==(4,5) and np.isfinite(losses).all(),'finite tuning grid')
                        combined=sum(len(v['validation_families'])*np.array(v['rank_mse']) for v in r['inner_folds'])/len(r['training_families'])
                        compare(combined,r['inner_family_rank_mse'],'tuning aggregation')
                        best=4-int(np.argmin(combined[::-1])); check(r['alpha']==c['alphas'][best],'alpha selected from recorded inner losses')
                for ai,a in enumerate(assay_names):
                    ix=indices[a]; pr=predictions[ix]; ranks=rankdata(pr,axis=0); centered=ranks-ranks.mean(0); ty=target[ix]-target[ix].mean(); denominators=np.sqrt(np.sum(centered**2,axis=0)*np.sum(ty**2)); rho=np.divide(np.sum(centered*ty[:,None],axis=0),denominators,out=np.full(len(designs),np.nan),where=denominators>1e-12); mse=np.mean((pr-target[ix,None])**2,axis=0)
                    row=indexed[(arm,s,a)]; check(row['cluster']==str(family[ix][0]),'metric family identity')
                    for metric,values in [('spearman',rho),('rank_mse',mse)]:
                        family_values[metric][mi,si,ai]=values
                        for di,d in enumerate(designs): compare(None if not np.isfinite(values[di]) else float(values[di]),row[metric][d],'assay '+metric)
                        for name,definition in contrasts.items():
                            aa=designs.index(definition['augmented']); bb=designs.index(definition['reduced']); value=(values[aa]-values[bb])*(1 if metric=='spearman' else -1)
                            compare(None if not np.isfinite(value) else float(value),row[metric+'_contrasts'][name],'assay contrast '+metric)
    inference=load(folder/'contrasts.json'); check(inference['status']=='complete' and inference['bootstrap']==2000 and inference['seeds']==seeds and inference['arms']==models and inference['families']==sorted(str(f) for f in families),'inference scope')
    # No bootstrap generation: independently check its deterministic point estimator and interval algebra.
    points={}; endpoint_points={}
    for metric,tensor in family_values.items():
        seed_mean=tensor.mean(axis=1)
        fmean=np.stack([seed_mean[:,[ai for ai,a in enumerate(assay_names) if family[indices[a][0]]==f],:].mean(axis=1) for f in families],axis=1)
        endpoint=fmean.mean(axis=1); endpoint_points[metric]={arm:{d:float(endpoint[mi,di]) for di,d in enumerate(designs)} for mi,arm in enumerate(models)}
        points[metric]={arm:{name:float((endpoint[mi,designs.index(e['augmented'])]-endpoint[mi,designs.index(e['reduced'])])*(1 if metric=='spearman' else -1)) for name,e in contrasts.items()} for mi,arm in enumerate(models)}
    summary={}; tables={}
    for label,g in inference['groups'].items():
        metric='rank_mse' if label=='secondary_rank_mse' else 'spearman'; family_kind='primary' if label=='primary_spearman' else 'supplementary' if label=='supplementary_spearman' else None
        names=[name for name,e in contrasts.items() if family_kind is None or e['family']==family_kind]
        check(g['status']=='complete' and g['hypothesis_count']==33*len(names) and len(g['contrasts'])==33*len(names),'multiplicity scope')
        check({(r['arm'],r['contrast']) for r in g['contrasts']}=={(a,n) for a in models for n in names},'multiplicity identity grid')
        crit=g['joint_critical_max_absolute_deviation']; table=[]
        for r in g['contrasts']:
            value=points[metric][r['arm']][r['contrast']]; compare(value,r['point'],'bootstrap point '+metric); compare([r['point']-crit,r['point']+crit],r['simultaneous'],'simultaneous band algebra')
            lo,hi=r['simultaneous']; state='positive' if lo>0 else 'negative' if hi<0 else 'unresolved'
            table.append(dict(model=r['arm'],contrast=r['contrast'],point=value,simultaneous=r['simultaneous'],state=state))
        tables[label]=table; summary[label]={}
        for name in names:
            selected=[r for r in table if r['contrast']==name]; low=min(selected,key=lambda r:r['point']); high=max(selected,key=lambda r:r['point'])
            summary[label][name]=dict(counts={state:sum(r['state']==state for r in selected) for state in ['positive','negative','unresolved']},range=[low['point'],high['point']],minimum_model=low['model'],maximum_model=high['model'],positive_models=[r['model'] for r in selected if r['state']=='positive'],negative_models=[r['model'] for r in selected if r['state']=='negative'])
    panels[panel]=dict(support=c['panel_support'][panel],summary=summary,all_model_contrasts=tables,endpoint_point_estimates=endpoint_points,multiplicity={label:dict(hypotheses=g['hypothesis_count'],critical=g['joint_critical_max_absolute_deviation']) for label,g in inference['groups'].items()})
    print(panel,json.dumps({label:{name:{k:v for k,v in entry.items() if k not in ['positive_models','negative_models']} for name,entry in group.items()} for label,group in summary.items()}),flush=True)

result=dict(status='passed' if not failures and not unverified else 'failed' if failures else 'passed_with_source_limitations',analysis='independent_numerical_acceptance_information_progression_20261006',production_task='b126b96f8',production_completion_sha256=digest(PROD/'completion.json'),validation_script_sha256=digest(Path(__file__)),validation=dict(failures=failures,unverified=unverified,hashes_verified=verified,cells=198,design_prediction_vectors=prediction_vectors,outer_fold_records=fold_count,max_absolute_errors=maximum_errors,bootstrap_rerun=False,refits=False,model_inference=False,scope='Every receipted production output; all current declared code and available source bytes; identities/targets/finite predictions; original/projected/realized folds; recorded alpha selection; independently recomputed assay metrics, contrast algebra, seed-assay-family point aggregation, simultaneous band algebra. No bootstrap resampling or quantile verification; no regenerated design matrices or fit-coefficient audit.'),panels=panels,limitations=c['limitations'],production_resource_receipt=dict(elapsed_seconds=completion['elapsed_seconds'],runtime=load(PROD/'runtime-end.json')),validation_resource_receipt=dict(start=before,end=resource_receipt(),elapsed_seconds=time.monotonic()-START))
(OUT/'result-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('ACCEPTANCE',result['status'],'failures',failures,'unverified',unverified,'maxerrors',maximum_errors,flush=True)
if failures: raise SystemExit(1)
