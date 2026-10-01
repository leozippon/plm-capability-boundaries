"""Validate and bind paired native term archives to the original frozen R1 rows."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .position_terms import state_parts, mutation_parts, partition_masks


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(1<<22),b''):h.update(part)
    return h.hexdigest()


def validate_pair_arrays(data, cohort_row, packer):
    """Check every retained prefix term and component against its original inputs."""
    indices=[i for i,m in enumerate(cohort_row['mutants']) if ':' not in m]
    if not np.array_equal(data['variant_indices'],indices):raise ValueError('paired variant indices differ')
    if data['mutants'].tolist()!=[cohort_row['mutants'][i] for i in indices]:
        raise ValueError('paired mutation identifiers differ')
    wild=packer.state(cohort_row['wildtype'])
    for name,value in [('wild_ids',wild['ids']),('residue_counts',wild['counts']),
                       ('residue_offset',wild['offset']),('scored_span',wild['span'])]:
        if not np.array_equal(data[name],value):raise ValueError(f'paired {name} does not match packing')
    if data['wild_position_nats'].dtype!=np.float32 or data['mutant_position_nats'].dtype!=np.float32:
        raise ValueError('paired terms must retain float32 precision')
    components=[]
    for j,i in enumerate(indices):
        mutant=packer.state(cohort_row['sequences'][i]);site=int(cohort_row['mutants'][i][1:-1])-1
        if not np.array_equal(data['mutant_ids'][j],mutant['ids']):raise ValueError('paired mutant ids differ')
        wt=np.asarray(data['wild_position_nats'][j]);mt=np.asarray(data['mutant_position_nats'][j])
        if not np.isfinite(wt).all() or not np.isfinite(mt).all():raise ValueError('nonfinite paired terms')
        prefix=partition_masks(wild['counts'],wild['offset'],site)['upstream']
        if not np.array_equal(wt[prefix],mt[prefix]):raise ValueError('paired prefix invariance failed')
        full=float(data['nll'][j,0]-data['nll'][j,1])
        part=mutation_parts(state_parts(wt,wild['counts'],wild['offset'],site),
                            state_parts(mt,mutant['counts'],mutant['offset'],site),full)
        for key in ('own','downstream','upstream','full','closure_nats','closure_bound_nats'):
            if float(data[key][j])!=part[key]:raise ValueError(f'paired {key} reconstruction failed')
        if abs(part['closure_nats'])>part['closure_bound_nats']:raise ValueError('paired closure bound failed')
        components.append(part)
    return indices,components


def attach_paired(loaded,cohort,packer,directory:Path,*,cohort_sha,support_sha):
    """Mutate only the single-substitution score fields after complete validation."""
    paths=sorted(directory.glob('manifest_paired_shard*.json'))
    if not paths:raise ValueError('no paired manifests')
    manifests=[json.loads(p.read_bytes()) for p in paths]
    shards=manifests[0]['shards']
    if {m['shard'] for m in manifests}!=set(range(shards)) or len(manifests)!=shards:
        raise ValueError('incomplete or duplicate paired shards')
    identity=manifests[0]['identity'];receipts={};hashes={str(p):sha(p) for p in paths}
    if identity['cohort_sha256']!=cohort_sha or identity['support_sha256']!=support_sha:
        raise ValueError('paired source identity mismatch')
    if identity['arm']!='progen3-3b' or identity['dtype']!='bfloat16' or identity['batch_size']!=2:
        raise ValueError('unexpected paired scoring protocol')
    for m in manifests:
        if m['status']!='complete' or m['identity']!=identity:raise ValueError('inconsistent paired receipt')
        for item in m['assays']:
            if item['assay'] in receipts:raise ValueError('duplicate paired assay')
            receipts[item['assay']]=item
    source={r['assay']:r for r in cohort['assays']}
    expected={r['assay'] for r in loaded['rows'] if any(e['substitutions']==1 for e in r['entries'])}
    if set(receipts)!=expected:raise ValueError('paired assay support differs')
    references=[];closures=[]
    for row in loaded['rows']:
        if row['assay'] not in expected:continue
        item=receipts[row['assay']];path=directory/item['file']
        digest=sha(path)
        if digest!=item['sha256']:raise ValueError('paired archive checksum mismatch')
        hashes[str(path)]=digest
        with np.load(path,allow_pickle=False) as data:
            metadata=json.loads(str(data['metadata']))
            if (metadata['identity']!=identity or metadata['assay']!=row['assay']
                    or metadata['mutant_digest']!=source[row['assay']]['mutant_digest']):
                raise ValueError('paired archive metadata mismatch')
            indices,parts=validate_pair_arrays(data,source[row['assay']],packer)
            for i,part in zip(indices,parts):
                closures.append(abs(part['closure_nats']))
                row['entries'][i].update({key:part[key] for key in ('own','downstream','upstream','full')})
            wt=np.asarray(data['nll'][:,0],dtype=float)
            references.append(dict(assay=row['assay'],variants=len(wt),wt_nll_min=float(wt.min()),
                                   wt_nll_max=float(wt.max()),wt_nll_std=float(wt.std())))
    return dict(identity=identity,source_sha256=hashes,per_variant_wt_reference_variation=references,
                closure_max_nats=max(closures),retention_max_abs_nats=0.0,
                interpretation='fresh paired numerical score; historical score retained separately, never overwritten in original archives')
