#!/usr/bin/env python3
"""Audit R1 release-family robustness and R4 exact-cohort channel qualification."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.capability.interactions.run_residual_panel import simultaneous_bands
from scripts.capability.stability import qualify_stability_endpoint as qualification
from src.capability.core.io import sha256_file
from src.capability.interactions.pairwise_epistasis import SPLIT_SEEDS, ROSTER

# Release families derive from checkpoint identifiers, not any outcome label.
# Frozen yes/no flags and legacy keys are retained for compatibility, not exposure.
GROUPING = {
    'labels': {
        'yes': 'protein-specialized or protein-adapted release families',
        'no': 'general-purpose text and scientific language–protein release families',
    },
    'interpretation': 'Frozen release grouping, not protein-exposure status.',
    'joint_model_assignments': {'InstructProtein': 'yes', 'Galactica': 'no'},
    'joint_model_note': 'InstructProtein is a joint language–protein model with protein adaptation in the first group; Galactica is a joint scientific language–protein model in the second group.',
    'legacy_keys': ['protein_pretraining', 'protein_minus_text_release_mean'],
    'contrast': 'Equal-weighted release-family mean of the yes group minus the no group.',
}
FAMILIES = {
    'ByGPT5': ('no', ('bygpt5-base-en','bygpt5-medium-en','bygpt5-small-en')),
    'DialoGPT': ('no', ('dialogpt-small',)),
    'Galactica': ('no', ('galactica-125m','galactica-1.3b','galactica-6.7b','galactica-30b')),
    'GPT2': ('no', ('gpt2','gpt2-medium','gpt2-large','gpt2-xl')),
    'Llama2': ('no', ('llama-2-7b',)), 'Llama3': ('no', ('llama-3.2-3b',)),
    'Qwen2.5': ('no', ('qwen2.5-0.5b','qwen2.5-0.5b-instruct','qwen2.5-7b','qwen2.5-32b')),
    'Qwen3': ('no', ('qwen3-8b-base',)),
    'InstructProtein': ('yes', ('instructprotein',)),
    'ProGen2': ('yes', ('progen2-small','progen2-base','progen2-medium','progen2-large','progen2-xlarge')),
    'ProGen3': ('yes', ('progen3-112m','progen3-3b')),
    'ProLLaMA': ('yes', ('prollama-stage-1','prollama')),
    'ProteinGLM': ('yes', ('proteinglm-7b-clm',)),
    'ProtGPT2': ('yes', ('protgpt2',)), 'ProtGPT3': ('yes', ('protgpt3-1.3b',)),
    'RITA': ('yes', ('rita-xl',)),
}


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def lineage(root, out):
    path = root/'results/R5/local_context_20260923/20260923233257_e429ce7f31e4/lcgp_admission/local_context_measurement.json'
    admission = json.loads(path.read_text())
    roster = sorted(a for _, arms in FAMILIES.values() for a in arms)
    if roster != sorted(ROSTER) or len(roster) != len(set(roster)):
        raise ValueError('release-family mapping differs from the 33-arm roster')
    write(out/'lineage_declaration.json', {
        'families': FAMILIES, 'grouping': GROUPING,
        'mapping_basis': 'released checkpoint families, independent of outcomes',
        'admission_sha256': sha256_file(path), 'metric':'increment_M_C_P_wall',
        'statistic':'equal checkpoints within release family; equal seeds within biological group',
        'bootstrap':'10000 paired biological-group draws, max statistic across16 release means',
        'training_ancestry':'Release families are not asserted independent training experiments.',
        'sensitivity':'exclude both Llama2 and ProLLaMA release families; known shared text parent',
    })
    groups, rows_by_arm, sources = None, {}, {}
    for file, expected in admission['report_sha256'].items():
        source = Path(file)
        if sha256_file(source) != expected:
            raise ValueError(f'source report digest mismatch: {source}')
        record = json.loads(source.read_text())
        arm, seed = record['arm'], int(record['fold_seed'])
        if arm not in roster or seed not in SPLIT_SEEDS:
            raise ValueError('unexpected report cell')
        keyed = {}
        for row in record['assays']:
            group = str(row['cluster'])
            value = float(row['C_P_wall+M_spearman']) - float(row['C_P_wall_spearman'])
            keyed.setdefault(group, []).append(value)
        current = sorted(keyed)
        if groups is None:
            groups = current
        if current != groups:
            raise ValueError('biological group support differs across cells')
        values = np.array([np.mean(keyed[g]) for g in groups])
        published = admission['summaries'][f'{arm}/{seed}']['increment_M_C_P_wall']
        if not np.isclose(values.mean(), published['point'], rtol=0, atol=1e-12):
            raise ValueError(f'{arm}/{seed}: retained group values do not reproduce admitted point')
        if seed in rows_by_arm.setdefault(arm, {}):
            raise ValueError('duplicate cell')
        rows_by_arm[arm][seed] = values
        sources[file] = expected
    if set(rows_by_arm) != set(roster) or any(set(v)!=set(SPLIT_SEEDS) for v in rows_by_arm.values()):
        raise ValueError('incomplete33-by3 panel')
    arm_values = {a:np.mean([rows_by_arm[a][s] for s in SPLIT_SEEDS],axis=0) for a in roster}
    labels = list(FAMILIES)
    family_values = {f:np.mean([arm_values[a] for a in arms],axis=0) for f,(_,arms) in FAMILIES.items()}
    panel = simultaneous_bands(np.column_stack([family_values[f] for f in labels]))
    def group_difference(excluded=()):
        specialized_or_adapted = [v for f,v in family_values.items() if f not in excluded and FAMILIES[f][0]=='yes']
        text_and_scientific = [v for f,v in family_values.items() if f not in excluded and FAMILIES[f][0]=='no']
        return simultaneous_bands((np.mean(specialized_or_adapted,axis=0)-np.mean(text_and_scientific,axis=0))[:,None])
    checkpoint = {}
    for a in roster:
        estimates=[admission['summaries'][f'{a}/{s}']['increment_M_C_P_wall'] for s in SPLIT_SEEDS]
        checkpoint[a]={'resolved_positive_all_three':all(r['interval'][0]>0 for r in estimates),
                       'per_seed_point':[r['point'] for r in estimates]}
    # Legacy protein_pretraining and protein_minus_text_release_mean keys remain
    # readable; GROUPING supplies their release-group semantics, not exposure.
    result={'schema':'local_context_release_family_robustness_v1','grouping':GROUPING,
            'admission':{'path':str(path),'sha256':sha256_file(path)},'source_reports_sha256':sources,
            'declaration_sha256':sha256_file(out/'lineage_declaration.json'),'biological_groups':groups,
            'group_unit':'wild-type cluster at50%identity, shared paired across all arms and seeds',
            'family_order':labels,'family_bootstrap':panel,
            'family_records':{f:{'protein_pretraining':FAMILIES[f][0],'checkpoints':FAMILIES[f][1],
                                'point':panel['point'][i], 'simultaneous95':panel['interval'][i]}
                              for i,f in enumerate(labels)},
            'checkpoint_records':checkpoint,'protein_minus_text_release_mean':group_difference(),
            'exclude_shared_llama2_prollama':group_difference(('Llama2','ProLLaMA')),
            'training_ancestry_limit':'Sixteen release families are not 16 independent training trials; ProLLaMA shares Llama2 parent. ProtGPT2 is recorded from-scratch despite GPT2 architecture. Galactica/InstructProtein shared initialization is unrecorded. Biological-group bootstrap conditions on this selected checkpoint panel.'}
    write(out/'lineage_robustness.json',result)


def exact_stability(root, out):
    path=root/'results/gate_stability_20260924/cohort.json'
    cohort=json.loads(path.read_text())
    panel=json.loads((root/'results/gate_stability_20260924/panel/panel.json').read_text())
    if sha256_file(path)!=panel['support']['cohort_sha256']:
        raise ValueError('model cohort digest differs from supporting panel')
    rows=[]
    for background in cohort['backgrounds']:
        for variant in background['variants']:
            rows.append({'WT_name':background['name'],'sequence':variant['sequence'],
                         'position':variant['position'],'kind':'natural','group':background['group'],
                         'cluster':background['cluster'],'combined':variant['ddg'],
                         'trypsin':variant['ddg_trypsin'],'chymotrypsin':variant['ddg_chymotrypsin']})
    frame=pd.DataFrame(rows)
    if len(frame)!=25856 or frame.WT_name.nunique()!=101 or frame.group.nunique()!=101:
        raise ValueError('exact stability model support changed')
    result={'schema':'exact_stability_model_support_qualification_v1','cohort_sha256':sha256_file(path),
            'qualification_code_sha256':sha256_file(Path(qualification.__file__)),
            'endpoint':'ddG=mutant minus WT, combined MegaScale unfolding free energy',
            'method':'original median-aggregated accepted channel labels on exact frozen model support',
            'bootstrap':{'draws':2000,'seed':20260923},
            'agreement':qualification.summarise(frame,draws=2000,seed=20260923),
            'limitation':'Shared assay/model errors contribute to channel agreement; this is not independent ground truth. Single-row aggregation sensitivity not rerun from original rows.'}
    write(out/'exact_stability_qualification.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists():
        raise SystemExit('refuse existing output directory')
    args.out.mkdir(parents=True)
    exact_stability(args.root,args.out)
    lineage(args.root,args.out)
    print('SUPPORT_AUDIT_EXIT=0',flush=True)

if __name__=='__main__':
    main()
