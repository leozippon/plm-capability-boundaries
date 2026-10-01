#!/usr/bin/env python3
"""Extract frozen aggregate evidence and draw the NCS manuscript figures.

No model scoring, fitted analysis or bootstrap is performed. Existing intervals
are copied unchanged. Run with the validated ct Python from any directory.
"""
from pathlib import Path
import csv
import hashlib
import json
import runpy
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
OUT = HERE / 'figure_data' / 'ncs'
SUBMISSION = ROOT / 'manuscript' / 'figures' / 'ncs'
OUT.mkdir(parents=True, exist_ok=True)
SUBMISSION.mkdir(parents=True, exist_ok=True)
SOURCES = {}
if '--from-source-data' in sys.argv:
    runpy.run_path(str(HERE / 'render_ncs_figures.py'), run_name='__main__')
    raise SystemExit(0)

def read_repo(path):
    content = (ROOT/path).read_bytes()
    SOURCES[path] = hashlib.sha256(content).hexdigest()
    return json.loads(content), path

def digest_repo(path):
    """Record a repository file's digest without parsing it.

    The qualified local control's fitted cells and two declared-block tables are retained on
    the cluster rather than in this package, so those rows are transcribed from the canonical
    gate document and carry the document's digest instead of an artifact pointer.
    """
    content = (ROOT/path).read_bytes()
    return hashlib.sha256(content).hexdigest()


def provenance(path, pointer):
    return {'source_path': path, 'source_sha256': SOURCES[path], 'source_pointer': pointer}

def save_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

def estimate(arm, endpoint, value, path, pointer, **extra):
    assert not value.get('degenerate'), (arm, endpoint)
    low, high = value['interval']
    point = value['point']
    assert low <= point <= high
    return dict(arm=arm, endpoint=endpoint, estimate=point, ci_low=low, ci_high=high,
                n_assays=value['n_assays'], n_units=value['n_units'],
                bootstrap_resamples=value['resamples'], unit=value['unit'],
                **extra, **provenance(path,pointer))

# Native scoring: published ladder support stays separate; no cross-stratum rank.
native=[]; paired=[]
for filename,group in [('results/R1/progen2_scale_20260826/scale_capability.json','ProGen2'),
                       ('results/R1/progen3_stage_20260917/second_stage_capability/second_stage_capability.json','ProGen3'),
                       ('results/R1/adaptation_lineage_20260813/adaptation_stage_capability.json','Llama2-ProLLaMA')]:
    data,path=read_repo(filename)
    d=data['dms']
    for endpoint in ['raw_spearman','model_minus_lookup','model_minus_blosum62']:
        for arm,value in d[endpoint]['per_rung'].items():
            native.append(estimate(arm,endpoint,value,path,f'dms.{endpoint}.per_rung.{arm}',comparison_group=group))
        for pair,value in d[endpoint]['adjacent_delta_rho'].items():
            paired.append(estimate(pair,endpoint,value,path,f'dms.{endpoint}.adjacent_delta_rho.{pair}',comparison_group=group))
data,path=read_repo('results/R1/native_dms_20260913/20260915134508_95c515bf6881/rita_xl_independent_analyse/native_dms_comparison.json')
for endpoint in ['raw_spearman','model_minus_lookup','model_minus_blosum62']:
    value=data['groups']['rita-xl'][endpoint]['per_rung']['rita-xl']
    native.append(estimate('rita-xl',endpoint,value,path,f'groups.rita-xl.{endpoint}.per_rung.rita-xl',comparison_group='RITA-native-fp32'))
data,path=read_repo('results/R1/source_recovery_20260917/galactica_native_dms_comparison.json')
assert SOURCES[path]=='38cb2b5dd549ef3e4b9f9fbb9e362a025c9a61f7ac94b1e9c204bc48b6ff7d84'
for endpoint in ['raw_spearman','model_minus_lookup','model_minus_blosum62']:
    for arm,value in data['groups']['galactica'][endpoint]['per_rung'].items():
        native.append(estimate(arm,endpoint,value,path,f'groups.galactica.{endpoint}.per_rung.{arm}',comparison_group='Galactica-native-fp32'))
data,path=read_repo('results/R1/source_recovery_20260917/proteinglm_retrieval_bound.json')
assert SOURCES[path]=='de08ee2921003c557659a33e9d305d897c0c0202b8106bc6826628fa45f1dc78'
for endpoint,key in [('model_minus_lookup','delta_lookup'),('model_minus_blosum62','delta_blosum62')]:
    native.append(estimate('proteinglm-7b-clm',endpoint,data['arms']['proteinglm-7b-clm'][key],path,f'arms.proteinglm-7b-clm.{key}',comparison_group='ProteinGLM-native'))
save_csv('native-fitness-source.csv',native);save_csv('paired-fitness-contrasts.csv',paired)
# Text amino-acid string controls; exact common support, no residue-model claim.
data,path=read_repo('results/R1/source_recovery_20260917/text_aa_analyse.json');text=[]
for arm,block in data['common_all13']['per_model'].items():
    for endpoint in ['raw','model_minus_lookup','model_minus_blosum62']:
        text.append(estimate(arm,endpoint,block[endpoint],path,f'common_all13.per_model.{arm}.{endpoint}',encoding_role=block['encoding_role']))
assert {(r['n_assays'],r['n_units']) for r in text} == {(211,169)}
save_csv('text-aa-fitness-source.csv',text)
# Attempt census: use separate 3B completion; do not convert missing groups to zero.
data,path=read_repo('results/R6/unconditional_diversity_20260919/unconditional_diversity.json')
census={arm:(v,path) for arm,v in data['checkpoints'].items()}
data,path=read_repo('results/R6/unconditional_diversity_3b_20260919/unconditional_diversity.json')
for arm,v in data['checkpoints'].items(): census[arm]=(v,path)
gen=[]
for arm,(v,path) in census.items():
    assert v['n_attempts']==800 and v['n_empty']+v['n_nonempty']==800
    assert v['n_any_profile_unknown']==0
    keys=['n_attempts','n_empty','n_nonempty','n_unique_nonempty_hashes','n_groups_nonempty',
          'n_any_profile_hits','n_any_profile_groups','shingle_length','unit']
    row=dict(arm=arm,**{k:v[k] for k in keys},**provenance(path,f'checkpoints.{arm}'))
    for key,count in [('nonempty_yield','n_nonempty'),('profile_yield','n_any_profile_hits'),('profile_group_yield','n_any_profile_groups'),('exact_unique_yield','n_unique_nonempty_hashes')]:
        row[key]=None if v[count] is None else v[count]/v['n_attempts']
    gen.append(row)
save_csv('generation-yield-source.csv',gen)
# Conditional profile-only cells; old folding endpoints in this artifact are never selected.
data,path=read_repo('results/R6/generation_evidence_20260905/analysis/20260904232436_00129607e3c7/main/generation_biology_analysis.json')
conditional=[]
for v in data['profile_ledger']:
    if v['role']!='generation' or v['condition'] not in ['requested','mismatched']: continue
    keys=['arm','class_key','condition','primary_class','n_attempts','n_target_profile','target_profile_known',
          'target_profile_rate','n_distinct_target_groups','distinct_target_groups_per_attempt']
    conditional.append({k:v[k] for k in keys}|provenance(path,f"profile_ledger[arm={v['arm']},class={v['class_key']},condition={v['condition']}]") )
save_csv('conditional-class-yield-source.csv',conditional)
# Related-context score benefit: paired win rate, not pooled classification AUROC.
# The two ProGen3 arms of the expansion aggregate were withdrawn on 2026-09-23 (batched
# sequence-id assignment) and are taken from the singleton remeasurement instead.
WITHDRAWN_CONTEXT_ARMS={'progen3-112m','progen3-3b'}
context=[]
for filename,drop in [('results/R5/context_homologue_20260826/context_homologue.json',set()),
                      ('results/R5/context_homologue_expansion_20260919/context_homologue.json',WITHDRAWN_CONTEXT_ARMS),
                      ('results/R5/context_homologue_progen3_20260924/context_homologue.json',set())]:
    data,path=read_repo(filename)
    if 's46rm_' in filename:
        assert SOURCES[path]=='cd90106ecc61f8e4e4b3e977882c5a07ab5bc6748a6556ee3cb3a4cb8e281008'
        assert set(data['arms'])==WITHDRAWN_CONTEXT_ARMS
    for arm,v in data['arms'].items():
        if arm in drop:continue
        if v['modality']!='protein' or v.get('status')!='scored':continue
        for stratum in ['pooled','decisive_stratum']:
            block=v[stratum]
            for endpoint in ['auroc','fractional_reduction']:
                e=block[endpoint]
                assert not e['degenerate']
                context.append(dict(arm=arm,stratum=stratum,endpoint='paired_win_rate' if endpoint=='auroc' else endpoint,
                                    estimate=e['mean'],ci_low=e['ci95'][0],ci_high=e['ci95'][1],
                                    n_rows=e['n_rows'],n_groups=e['n_groups'],resamples=e['resamples'],
                                    position_only_nll_nats_per_token=block['position_only_nll_nats_per_token'],
                                    verdict=v['verdict']['outcome'],**provenance(path,f'arms.{arm}.{stratum}.{endpoint}')))
save_csv('homologue-context-source.csv',context)
# Frozen-representation readout: the two contrasts must travel together. R minus the native
# likelihood is a readout statement; the increment over the matched supervised baseline is the
# one a biological reading would need, and the literal-string text arms separate them.
READOUT_ROOT='results/R2/readout_expansion_20260923'
admission,admission_path=read_repo(f'{READOUT_ROOT}/final_admission.json')
strata={arm:block['stratum'] for arm,block in admission['interfaces'].items()}
readout=[]
for report in sorted((ROOT/READOUT_ROOT/'complete/results/external_baseline').glob('*/readout_*/readout_*.json')):
    relative=report.relative_to(ROOT).as_posix()
    data,path=read_repo(relative)
    if data.get('n_assays')!=201 or data.get('n_families')!=163 or data.get('fold_seed')!=20260923:
        SOURCES.pop(relative);continue
    if 'R_minus_raw_M_spearman' not in data.get('summaries',{}):
        SOURCES.pop(relative);continue
    row=dict(arm=data['arm'],interface_stratum=strata[data['arm']],fold_seed=data['fold_seed'],
             n_assays=data['n_assays'],n_units=data['n_families'],n_variants=data['n_variants'])
    for name,key in [('readout_minus_likelihood','R_minus_raw_M_spearman'),
                     ('representation_increment','delta_spearman'),
                     ('matched_baseline','B_spearman'),('native_likelihood','raw_M_spearman')]:
        value=data['summaries'][key]
        assert not value['degenerate'] and value['n_units']==163
        low,high=value['interval'];assert low<=value['point']<=high
        row.update({name:value['point'],f'{name}_ci_low':low,f'{name}_ci_high':high,
                    f'{name}_excludes_zero':value['excludes_zero']})
        row['unit']=value['unit'];row['bootstrap_resamples']=value['resamples']
    readout.append(row|provenance(path,'summaries'))
assert len(readout)==33 and len({r['arm'] for r in readout})==33
assert sum(r['interface_stratum']=='literal_text_AA' for r in readout)==14
save_csv('readout-contrasts-source.csv',readout)
# Follow-up panels preserve their simultaneous families and exact fitting support.
full_residual,full_residual_path=read_repo('results/R3/pairwise_residual_full_panel_20260927/panel.json')
assert len(full_residual['arms'])==33 and full_residual['primary_bootstrap']['family_size']==33
full_rows=[]
for arm,v in full_residual['arms'].items():
    fit_record,fit_path=read_repo(f'results/R3/pairwise_residual_full_panel_20260927/residual_{arm}.json')
    assert SOURCES[fit_path]==full_residual['receipts_sha256'][arm]
    baseline=sum(r['control_group_equal_mse_kcal2']['point'] for r in fit_record['seeds'].values())/3
    full_rows.append(dict(arm=arm,estimate=v['seed_mean_mse_reduction_kcal2'],
        ci_low=v['simultaneous95_interval_kcal2'][0],ci_high=v['simultaneous95_interval_kcal2'][1],
        baseline_mse=baseline,groups=64,cycles=8192,site_pairs=217,simultaneous_family=33,
        bootstrap_resamples=10000,baseline_source_path=fit_path,baseline_source_sha256=SOURCES[fit_path],
        baseline_source_pointer='mean(seeds.*.control_group_equal_mse_kcal2.point)',
        **provenance(full_residual_path,f'arms.{arm}')))
save_csv('residual-full-panel-source.csv',full_rows)
stability_replay,stability_replay_path=read_repo('results/R4/stability_likelihood_replay_20260927/panel.json')
assert len(stability_replay['arms'])==33 and stability_replay['primary']['family_size']==33
stability_replay_rows=[]
for i,arm in enumerate(stability_replay['arms']):
    fit_record,fit_path=read_repo(f'results/R4/stability_likelihood_replay_20260927/{arm}.json')
    assert SOURCES[fit_path]==stability_replay['receipts_sha256'][arm]
    baseline=sum(sum(v['baseline_mse'])/len(v['baseline_mse']) for v in fit_record['seeds'].values())/3
    lo,hi=stability_replay['primary']['interval'][i]
    stability_replay_rows.append(dict(arm=arm,estimate=stability_replay['primary']['point'][i],
        ci_low=lo,ci_high=hi,baseline_mse=baseline,groups=101,variants=25856,sites=5664,
        simultaneous_family=33,bootstrap_resamples=10000,baseline_source_path=fit_path,
        baseline_source_sha256=SOURCES[fit_path],baseline_source_pointer='mean(seeds.*.baseline_mse)',
        **provenance(stability_replay_path,f'primary.point.{i}')))
save_csv('stability-panel-source.csv',stability_replay_rows)
correlation,correlation_path=read_repo('results/shared/followup_support_20260927/lineage_correlation.json')
assert correlation['historical_group_membership_equal'] and correlation['checkpoint_count']==18
save_csv('lineage-correlation-source.csv',[
    dict(analysis=key,estimate=correlation[key]['point'],ci_low=correlation[key]['interval'][0],
         ci_high=correlation[key]['interval'][1],checkpoints=18,release_families=9,
         resampling_clusters=correlation[key]['clusters'],compositions=correlation[key]['compositions'],
         **provenance(correlation_path,key))
    for key in ('cluster_resampling','family_means','merge_galactica_instructprotein_sensitivity')])
lineages,lineage_path=read_repo('results/shared/followup_support_20260927/lineage_robustness.json')
lineage_rows=[]
for family,v in lineages['family_records'].items():
    lineage_rows.append(dict(release_family=family,checkpoints=len(v['checkpoints']),
        protein_pretraining=v['protein_pretraining'],estimate=v['point'],
        ci_low=v['simultaneous95'][0],ci_high=v['simultaneous95'][1],
        biological_groups=163,simultaneous_family=16,bootstrap_resamples=10000,
        **provenance(lineage_path,f'family_records.{family}')))
save_csv('lineage-robustness-source.csv',lineage_rows)
# Tables are rendered from the same retained numbers, not hand-copied estimates.
(SUBMISSION/'full-residual-panel-rows.tex').write_text(''.join(
    f"{r['arm']} & {r['estimate']:+.7f} & [{r['ci_low']:+.7f}, {r['ci_high']:+.7f}] & {r['baseline_mse']:.6f} " + chr(92)*2 + chr(10)
    for r in full_rows))
(SUBMISSION/'lineage-robustness-rows.tex').write_text(''.join(
    f"{r['release_family']} & {r['checkpoints']} & {r['protein_pretraining']} & {r['estimate']:+.5f} & [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}] " + chr(92)*2 + chr(10)
    for r in lineage_rows))
stability_support=[]
for population,relative,variants,backgrounds,sites in [
    ('broad_qualification','results/R4/gate_stability_20260924/endpoint/endpoint_qualification.json',261572,290,16549),
    ('exact_model_cohort','results/shared/followup_support_20260927/exact_stability_qualification.json',25856,101,5664)]:
    data,path=read_repo(relative)
    for metric,v in data['agreement']['natural']['family_group']['agreement'].items():
        stability_support.append(dict(population=population,metric=metric,variants=variants,
            backgrounds=backgrounds,sites=sites,groups=101,estimate=v['point'],ci_low=v['ci95'][0],ci_high=v['ci95'][1],
            **provenance(path,f'agreement.natural.family_group.agreement.{metric}')))
save_csv('stability-support-source.csv',stability_support)

# Endpoint qualification: how much of each measured endpoint reproduces across its own assay
# channels, and whether the structure-conditioned contrast resolves. No model quantity enters.
boundaries=[]
def boundary(family,endpoint,point,low,high,unit,unit_label,n_units,kish,path,pointer,**extra):
    assert low<=point<=high,(endpoint,point,low,high)
    boundaries.append(dict(family=family,endpoint=endpoint,estimate=point,ci_low=low,ci_high=high,
                           unit=unit,unit_label=unit_label,n_units=n_units,effective_units_kish=kish,
                           bootstrap_resamples=2000,**extra,**provenance(path,pointer)))
data,path=read_repo('results/shared/followup_support_20260927/exact_stability_qualification.json')
block=data['agreement']['natural']['family_group']
value=block['agreement']['shared_to_discordance_ratio']
boundary('reliability','single_substitution_stability',value['point'],*value['ci95'],
         'wild-type family group','MegaScale model-cohort singles',block['units'],block['effective_units_kish'],
         path,'agreement.natural.family_group.agreement.shared_to_discordance_ratio',measured_scale='dimensionless ratio')
data,path=read_repo('archive/logs/R3/pairwise_label_instrument_20260923/qualification.json')
block=data['variants']['both_channel_width']['kinds']['natural']['all']['source_cluster']
value=block['agreement']['shared_to_discordance_ratio']
boundary('reliability','double_mutant_cycle',value['point'],*value['ci95'],
         'source cluster','MegaScale double-mutant cycles',block['units'],block['effective_units_kish'],
         path,'variants.both_channel_width.kinds.natural.all.source_cluster.agreement.shared_to_discordance_ratio',
         measured_scale='dimensionless ratio')
data,path=read_repo('archive/logs/shared/gate_higher_order_extended_20260923/his3_synonymous.json')
assert data['assay']=='HIS7_YEAST_Pokusaeva_2019' and data['resampling_unit']=='site tuple'
for order,published in [(2,(0.506,0.178,0.742)),(3,(0.276,0.000,0.496))]:
    block=next(o for o in data['orders'] if o['order']==order)
    value=block['shared_to_discordance_ratio']
    assert all(round(a,3)==b for a,b in zip([value['point']]+list(value['ci95']),published)),(order,value)
    boundary('reliability',f'order_{order}_cycle',value['point'],*value['ci95'],'site tuple',
             f'HIS3 order-{order} cycles',block['site_tuples'],block['effective_site_tuples_kish'],
             path,f'orders[order={order}].shared_to_discordance_ratio',measured_scale='dimensionless ratio')
data,path=read_repo('results/R3/contact_bootstrap_corrected_20260927/epsilon_enrichment.json')
definition=data['supports'][data['primary_support']]['definitions']['heavy_atom']
for endpoint in ['mean_abs_epsilon','mean_abs_epsilon_adjusted','mean_abs_half_channel_difference']:
    block=definition['endpoints'][endpoint]['site_pair_equal']
    boundary('contact_enrichment',endpoint,block['difference'],*block['interval'],'site pair',
             'contact minus matched control',block['control_site_pairs'],block['effective_control_site_pairs'],
             path,f'supports.indel_excluded.definitions.heavy_atom.endpoints.{endpoint}.site_pair_equal',
             measured_scale=block['unit'],contact_site_pairs=block['contact_site_pairs'])
save_csv('endpoint-boundaries-source.csv',boundaries)
# --- Capability map: the gate rows of Table 1 as plottable outcome classes. -----------------
# Counts and outcome classes are the table's own; each row names the gate document that owns
# it. No count is derived here and none is a model-population prevalence.
def transcribed(document, pointer):
    path = document
    return {'source_document': path, 'source_document_sha256': digest_repo(path),
            'source_pointer': pointer}

MAP_ROWS = [
    ('ranking_beyond_profile', 'Ranking beyond an implemented evolutionary profile', 'supported', 2, 14, False,
     'archive/docs/shared/hierarchy-crossed-controls.md', 'checkpoints exceeding the profile over checkpoints carrying the contrast'),
    ('states_beyond_likelihood', 'Frozen-state information beyond the native likelihood', 'supported', 69, 99, True,
     'archive/docs/R2/readout-results.md', 'checkpoint-seed cells, R minus native likelihood'),
    ('states_beyond_matched', 'Frozen-state information beyond matched supervised controls', 'not_detected', 3, 33, True,
     'archive/docs/R2/readout-results.md', 'unconditioned checkpoints with a resolved increment over the matched baseline'),
    ('likelihood_beyond_local', 'Likelihood information beyond a qualified local-context control', 'supported', 14, 33, False,
     'archive/docs/R1/gate-local-context.md', 'full-panel generalizability check, all three split seeds'),
    ('states_beyond_local', 'Frozen-state information beyond the same control', 'supported', 8, 33, True,
     'archive/docs/R1/gate-local-context.md', 'full-panel generalizability check, all three split seeds'),
    ('first_order_nonadditivity', 'First-order likelihood information on measured stability nonadditivity', 'supported', 5, 33, False,
     'archive/docs/R3/pairwise-epistasis-results.md', 'C_G_T_M1|C_G_T resolved at all three seeds'),
    ('pairwise_interaction', 'Measured-singles-adjusted likelihood interaction', 'unresolved', 0, 33, False,
     'archive/docs/R3/residual-panel.md', 'C_G_T+M|C_G_T; full33 simultaneous intervals; label-assisted sensitivity'),
    ('higher_order', 'Higher-order and background dependence', 'measurement_limited', None, 33, False,
     'archive/docs/R3/gate-higher-order-extended.md', 'order-3 cycle below its own channel-noise floor'),
    ('structural_contact', 'Concentration of measured nonadditivity at structural contacts', 'measurement_limited', None, 33, False,
     'archive/docs/R3/gate-structure-contact.md', 'contact minus matched control does not resolve'),
    ('related_context', 'Related-context use beyond matched unrelated context at low overlap', 'supported', 6, 19, False,
     'archive/docs/R5/gate-global-context.md', 'joint criterion in the pooled cohort and the low-overlap stratum'),
    ('retrieval_strata', 'First-order likelihood information under retrieval and provenance stratification', 'supported', None, 33, False,
     'archive/docs/R5/gate-retrieval-memorization.md', 'resolved on every axis with the units to resolve'),
    ('generation_products', 'Generation of products satisfying a stated biological requirement', 'supported', 3, 20, False,
     'archive/docs/R6/gate-generative-control.md', 'complete-domain endpoint beyond every qualified generator'),
]
# A gate whose outcome carries no count of resolved units states why in its own words rather
# than being drawn as if it had resolved none.
NO_COUNT_NOTE = {'higher_order': 'below its own noise floor',
                 'structural_contact': 'control side does not resolve',
                 'retrieval_strata': 'resolved on every axis with units'}
capability = []
for key, question, outcome, resolved, eligible, provisional, document, pointer in MAP_ROWS:
    assert (resolved is None) == (key in NO_COUNT_NOTE), key
    capability.append(dict(gate=key, question=question, outcome_class=outcome, resolved_units=resolved,
                           eligible_units=eligible, provisional=provisional,
                           no_count_note=NO_COUNT_NOTE.get(key, ''), **transcribed(document, pointer)))
save_csv('capability-map-source.csv', capability)

# --- Control ladders: every control block's own contribution, with its own interval. --------
# A block that lowers the set it extends bounds nothing measured over it, so each rung is
# reported as its own quantity beside the model increments taken over it.
CROSSED = {seed: ('archive/logs/shared/crossed_controls_20260923/reports/crossed_controls_proteinglm_'
                  f'{seed}/crossed_controls_proteinglm-7b-clm_fold{seed}.json')
           for seed in (20260923, 20260924, 20260925)}
NESTED = [('C', 'Composition, identity, position, length'),
          ('C_L', 'C + dipeptide and tripeptide differences'),
          ('C_P', 'C + mutation-local profile'),
          ('C_L_P', 'C + both'),
          ('C_L_P_T', 'C + both + tokenisation')]
ladder = []
nested_points = {}
for seed, relative in CROSSED.items():
    data, path = read_repo(relative)
    assert data['arm'] == 'proteinglm-7b-clm' and data['fold_seed'] == seed
    for name, label in NESTED:
        value = data['summaries'][f'{name}_spearman']
        assert not value['degenerate'] and value['n_units'] == 163
        low, high = value['interval']
        assert low <= value['point'] <= high
        nested_points[(seed, name)] = value['point']
        ladder.append(dict(gate='ranking', family='control_set', block=name, block_label=label,
                           baseline='', split_seed=seed, estimate=value['point'], ci_low=low, ci_high=high,
                           unit='held-cluster Spearman', unit_label='dimensionless Spearman',
                           n_units=value['n_units'], interval_retained=True,
                           **provenance(path, f'summaries.{name}_spearman')))
for seed in CROSSED:
    for name, base in [('C_L', 'C'), ('C_L_P', 'C_P')]:
        ladder.append(dict(gate='ranking', family='refused_block', block='dipeptide_tripeptide',
                           block_label='400 dipeptide + projected tripeptide differences',
                           baseline=base, split_seed=seed,
                           estimate=nested_points[(seed, name)] - nested_points[(seed, base)],
                           ci_low='', ci_high='', unit='held-cluster Spearman',
                           unit_label='dimensionless Spearman', n_units=163, interval_retained=False,
                           **provenance(CROSSED[seed], f'summaries.{name}_spearman minus summaries.{base}_spearman')))
# Qualified local-context candidates. The fitted cells are retained on the cluster, so these
# rows are transcribed from the canonical gate document with its digest recorded per row.
LOCAL_CANDIDATES = {
 'wcomp': ('Window chemical-class fractions and entropy', [
   (20260923, (0.02409, 0.01453, 0.03359), (0.02542, 0.01863, 0.03300), (0.10872, 0.08781, 0.13007)),
   (20260924, (0.02659, 0.01705, 0.03611), (0.02363, 0.01687, 0.03113), (0.10933, 0.08815, 0.13188)),
   (20260925, (0.02533, 0.01582, 0.03492), (0.02221, 0.01603, 0.02891), (0.11029, 0.08849, 0.13243))]),
 'wchem': ('Window physicochemical scales', [
   (20260923, (0.01072, 0.00078, 0.02076), (0.00995, 0.00269, 0.01701), (0.19938, 0.16881, 0.23016)),
   (20260924, (0.01194, 0.00204, 0.02176), (0.01005, 0.00325, 0.01705), (0.20486, 0.17428, 0.23570)),
   (20260925, (0.01068, 0.00027, 0.02088), (0.00801, 0.00209, 0.01418), (0.19939, 0.16869, 0.23020))]),
 'wsub': ('Window substitution scores', [
   (20260923, (-0.01245, -0.02260, -0.00291), (0.00290, 0.00012, 0.00563), (0.12395, 0.10061, 0.14818)),
   (20260924, (-0.00164, -0.00830, 0.00514), (0.00391, 0.00116, 0.00670), (0.12420, 0.10046, 0.14794)),
   (20260925, (-0.00127, -0.00792, 0.00539), (0.00279, 0.00031, 0.00528), (0.12395, 0.10053, 0.14761))]),
 'wall': ('Union of the three window blocks, 111 columns', [
   (20260923, (0.03485, 0.02295, 0.04670), (0.03296, 0.02296, 0.04344), (0.25613, 0.22620, 0.28571)),
   (20260924, (0.03661, 0.02524, 0.04857), (0.03316, 0.02372, 0.04360), (0.25981, 0.23101, 0.28977)),
   (20260925, (0.03387, 0.02228, 0.04622), (0.02973, 0.02058, 0.03926), (0.25498, 0.22534, 0.28468))]),
 'rf3': ('Bounded window, radius 3 residues', [
   (20260923, (0.01892, 0.00706, 0.03109), (0.00496, -0.00081, 0.01092), (0.27559, 0.24053, 0.30991)),
   (20260924, (0.01936, 0.00733, 0.03146), (0.00871, 0.00273, 0.01435), (0.27076, 0.23579, 0.30355)),
   (20260925, (0.01763, 0.00456, 0.03056), (0.00082, -0.00545, 0.00701), (0.27380, 0.23866, 0.30722))]),
 'rf7': ('Bounded window, radius 7 residues', [
   (20260923, (0.02559, 0.01289, 0.03828), (0.00085, -0.00495, 0.00669), (0.28260, 0.24773, 0.31588)),
   (20260924, (0.03111, 0.01742, 0.04537), (0.01102, 0.00441, 0.01742), (0.28659, 0.25282, 0.31997)),
   (20260925, (0.02657, 0.01223, 0.04054), (-0.00161, -0.00807, 0.00458), (0.28130, 0.24712, 0.31475))]),
}
QUALIFIED_LOCAL = {'wcomp': True, 'wchem': True, 'wsub': False, 'wall': True, 'rf3': True, 'rf7': False}
for block, (label, records) in LOCAL_CANDIDATES.items():
    for seed, over_c, over_cp, alone in records:
        for baseline, (point, low, high) in [('C', over_c), ('C+P', over_cp), ('', alone)]:
            assert low <= point <= high, (block, seed, baseline)
            ladder.append(dict(gate='ranking',
                               family='candidate_alone' if baseline == '' else 'candidate_contribution',
                               block=block, block_label=label, baseline=baseline, split_seed=seed,
                               estimate=point, ci_low=low, ci_high=high,
                               unit='held-cluster Spearman', unit_label='dimensionless Spearman',
                               n_units=163, interval_retained=True, qualified=QUALIFIED_LOCAL[block],
                               **transcribed('archive/docs/R1/gate-local-context.md', "candidate qualification table, "
                                             f"{block} at split seed {seed}")))
# Interaction-gate rungs. The control, nuisance and tokenisation blocks contain no model
# column, so one arm's record carries every rung; the assertion below holds them to that.
panel_data, interaction_path = read_repo('archive/logs/R3/pairwise_epistasis_20260924/panel/panel.json')
# The sequence/profile control and the nonlinear-additive response carry no model column, so
# one record carries them for the whole panel; the tokenisation and comparator blocks are
# arm-specific by construction and are emitted for every arm instead of being pooled.
RUNGS = [('all', 'C|ADDITIVE_NULL', 'C', 'Sequence and profile control over the additive null', True),
         ('all', 'C_G|C', 'G', 'Nonlinear-additive global response over C', True),
         ('all', 'C_G_T|C_G', 'T', 'Tokenisation descriptors over C+G', False),
         ('q', 'C_G_T_Q|C_G_T', 'Q', 'Pseudolikelihood coupling comparator over C+G+T', False)]
for support, contrast, block, label, shared in RUNGS:
    arms = panel_data['panel'][support][contrast]['arms']
    points = {round(a['per_seed']['20260923']['mse_reduction_kcal2'], 12) for a in arms.values()}
    assert shared == (len(points) == 1), (contrast, len(points))
    for arm, record in (list(arms.items())[:1] if shared else arms.items()):
        for seed, value in record['per_seed'].items():
            low, high = value['interval']
            assert low <= value['mse_reduction_kcal2'] <= high
            ladder.append(dict(gate='interaction',
                               family='control_rung' if shared else 'control_rung_per_arm',
                               block=block, block_label=label, arm='' if shared else arm,
                               baseline=contrast.split('|')[1], split_seed=int(seed),
                               estimate=value['mse_reduction_kcal2'], ci_low=low, ci_high=high,
                               unit='group-equal squared error reduction',
                               unit_label='kcal$^2$/mol$^2$',
                               n_units=panel_data['supports'][support]['groups'], interval_retained=True,
                               **provenance(interaction_path, f'panel.{support}.{contrast}.arms.'
                                                              f'{"*" if shared else arm}.per_seed.{seed}')))
INTERACTION_REFUSED = {
 'W': ('Bounded receptive field, +/-5 residues', [
   (20260923, (-0.00269, -0.00607, 0.00093)), (20260924, (-0.01033, -0.01745, -0.00401)),
   (20260925, (-0.00261, -0.00688, 0.00167))]),
 'X': ('Long-range additive block', [
   (20260923, (0.00434, -0.00391, 0.01268)), (20260924, (0.00322, -0.00540, 0.01200)),
   (20260925, (-0.00051, -0.01106, 0.00989))]),
}
for block, (label, records) in INTERACTION_REFUSED.items():
    for seed, (point, low, high) in records:
        assert low <= point <= high
        ladder.append(dict(gate='interaction', family='refused_block', block=block, block_label=label,
                           baseline='C_G_T', split_seed=seed, estimate=point, ci_low=low, ci_high=high,
                           unit='group-equal squared error reduction', unit_label='kcal$^2$/mol$^2$',
                           n_units=64, interval_retained=True, qualified=False,
                           **transcribed('archive/docs/R5/gate-global-context.md',
                                         f'declared-block contribution table, {block} over C+G+T at split seed {seed}')))
ladder.append(dict(gate='interaction', family='additive_null', block='ADDITIVE_NULL',
                   block_label='Zero-interaction additive null', baseline='', split_seed=20260923,
                   estimate=1.15441, ci_low=0.92605, ci_high=1.39050,
                   unit='group-equal mean squared error', unit_label='kcal$^2$/mol$^2$',
                   n_units=64, interval_retained=True,
                   **transcribed('archive/docs/R3/pairwise-epistasis-results.md', 'additive null on 64 groups')))
save_csv('control-ladder-source.csv', ladder)

# --- Interaction panel: every one of the 33 checkpoints, not a selected few. ----------------
# The first-order term, the likelihood interaction and the representation interaction are
# separate quantities on one support; the indel-restricted evaluation is carried beside the
# unrestricted one because both readings are reported.
PANEL_CONTRASTS = [('all', 'C_G_T_M1|C_G_T', 'first_order_likelihood'),
                   ('all', 'C_G_T+M|C_G_T', 'likelihood_over_matched'),
                   ('all', 'C_G_T_M1+M|C_G_T_M1', 'likelihood_interaction_over_first_order'),
                   ('all', 'C_G_T+R|C_G_T', 'representation_over_matched'),
                   ('all', 'C_G_T_R1+R|C_G_T_R1', 'representation_interaction_over_first_order'),
                   ('q', 'C_G_T_Q+M|C_G_T_Q', 'likelihood_after_comparator'),
                   ('q', 'C_G_T_Q+R|C_G_T_Q', 'representation_after_comparator')]
restricted, restricted_path = read_repo('archive/logs/R3/pairwise_epistasis_20260924/panel/panel_indel_restricted.json')
interaction = []
for support, contrast, endpoint in PANEL_CONTRASTS:
    for arm, record in panel_data['panel'][support][contrast]['arms'].items():
        primary = record['per_seed']['20260923']
        low, high = primary['interval']
        assert low <= primary['mse_reduction_kcal2'] <= high, (arm, contrast)
        row = dict(arm=arm, endpoint=endpoint, contrast=contrast, support=support,
                   tokenisation_stratum=record['stratum'], split_seed=20260923,
                   estimate=primary['mse_reduction_kcal2'], ci_low=low, ci_high=high,
                   seeds_above_zero=record['seeds_with_interval_above_zero'],
                   seeds_below_zero=record['seeds_with_interval_below_zero'],
                   point_range_low=record['point_range'][0], point_range_high=record['point_range'][1],
                   seed_mean_estimate=sum(v['mse_reduction_kcal2'] for v in record['per_seed'].values())
                   / len(record['per_seed']),
                   groups=panel_data['supports'][support]['groups'], cycles=panel_data['supports'][support]['cycles'],
                   site_pairs=panel_data['supports'][support]['site_pairs'],
                   unit='group-equal squared error reduction', unit_label='kcal$^2$/mol$^2$',
                   **provenance(interaction_path, f'panel.{support}.{contrast}.arms.{arm}'))
        if support == 'all' and contrast in restricted['panel']['all']:
            other = restricted['panel']['all'][contrast]['arms'][arm]
            other_primary = other['per_seed']['20260923']
            row.update(indel_restricted_estimate=other_primary['mse_reduction_kcal2'],
                       indel_restricted_ci_low=other_primary['interval'][0],
                       indel_restricted_ci_high=other_primary['interval'][1],
                       indel_restricted_seeds_above_zero=other['seeds_with_interval_above_zero'])
        interaction.append(row)
assert len({r['arm'] for r in interaction}) == 33
save_csv('interaction-panel-source.csv', interaction)

# --- Retrieval, identity and provenance strata. --------------------------------------------
# Strata were declared before any stratified estimate was computed; a stratum estimate is the
# same paired statistic over the same fitted predictions on the units that stratum holds, and
# a band holding fewer than eight units carries no interval and stays unresolved.
report, report_path = read_repo('archive/logs/R5/gate_retrieval_20260923/retrieval_strata_report.json')
STRATA_LEVELS = ['near_duplicate', 'identity_band', 'depth_band', 'source_provenance', 'family_band']
strata = []


def strata_rows(block, arm, level_name, band_name, channel, contrast):
    seeds = sorted(block)
    above = sum(1 for seed in seeds if block[seed]['excludes_zero'] and block[seed]['point'] > 0)
    below = sum(1 for seed in seeds if block[seed]['excludes_zero'] and block[seed]['point'] < 0)
    primary = block['20260923']
    interval = primary['interval']
    row = dict(channel=channel, contrast=contrast, arm=arm, stratification=level_name, band=band_name,
               split_seed=20260923, estimate=primary['point'],
               ci_low='' if interval is None else interval[0], ci_high='' if interval is None else interval[1],
               resolution=primary['resolution'],
               groups=primary.get('groups', primary.get('n_units')),
               effective_units_kish=primary.get('kish_effective_site_pairs_estimator_weights',
                                                primary.get('kish_effective_assays_estimator_weights', '')),
               seeds_above_zero=above, seeds_below_zero=below, seeds=len(seeds),
               **provenance(report_path, f'{channel}.{arm}.contrasts.{contrast}'
                                         f'.<seed>.{"full_support" if band_name == "" else f"strata.{level_name}.bands.{band_name}"}'))
    strata.append(row)


for channel, contrast in [('stability', 'C_G_T_M1|C_G_T'), ('readout_anchor', 'increment_R_C_P')]:
    for arm, record in report[channel].items():
        seeds = record['contrasts'][contrast]
        strata_rows({seed: value['full_support'] for seed, value in seeds.items()}, arm, 'full_support', '', channel, contrast)
        for level in STRATA_LEVELS:
            if level not in seeds['20260923']['strata']:
                continue
            for band in seeds['20260923']['strata'][level]['bands']:
                strata_rows({seed: value['strata'][level]['bands'][band] for seed, value in seeds.items()},
                            arm, level, band, channel, contrast)
save_csv('retrieval-strata-source.csv', strata)

# --- Matched generators: every cell against every cohort, on both oracle endpoints. ---------
# The matched-control rows carry the 97.5% percentile intervals the gate reads its two endpoints
# at, so a cell clearing a generator clears it at a level that admits both endpoints jointly. The
# identity-strata and ceiling rows carry a 95% Wilson interval on a raw rate instead, which is a
# different interval of a different quantity; interval_level says which one each row holds, so the
# two are never read as one column of comparable bounds.
gate, gate_path = read_repo('results/R6/generative_control_20260924/gate_endpoints.json')
COHORTS = ['shuffle', 'hydropathy', 'markov_0', 'markov_2', 'markov_4', 'fragment', 'natural']
generators = []
for cell in gate['cells']:
    for endpoint in ['any_family', 'complete_domain']:
        block = cell['endpoints'][endpoint]
        for cohort in COHORTS:
            control = block['controls'][cohort]
            low, high = control['ci97_5']
            assert low <= control['difference'] <= high, (cell['cell'], endpoint, cohort)
            generators.append(dict(cell=cell['cell'], arm=cell['arm'], condition=cell['condition'],
                                   endpoint=endpoint, cohort=cohort, model_rate=control['model_rate'],
                                   control_rate=control['control_rate'], difference=control['difference'],
                                   ci_low=low, ci_high=high, qualified=control['qualified'],
                                   n_attempts=control['n_attempts'], n_clusters=control['n_clusters'],
                                   resamples=control['resamples'], verdict=block['verdict']['label'],
                                   interval_level='97.5% percentile',
                                   unit='frozen near-duplicate sequence group of the attempt ledger',
                                   **provenance(gate_path, f"cells[cell={cell['cell']}].endpoints.{endpoint}.controls.{cohort}")))
    for band, block in cell['identity_strata'].items():
        record = block['any_family']
        generators.append(dict(cell=cell['cell'], arm=cell['arm'], condition=cell['condition'],
                               endpoint='any_family_by_reference_identity', cohort=band,
                               model_rate=record['model_rate'], control_rate='', difference='',
                               ci_low=record['model_wilson95'][0], ci_high=record['model_wilson95'][1],
                               qualified='', n_attempts=record.get('model_denominator', ''),
                               n_clusters='', resamples='', verdict='',
                               interval_level='95% Wilson',
                               unit='attempt, Wilson interval on the raw rate',
                               **provenance(gate_path, f"cells[cell={cell['cell']}].identity_strata.{band}.any_family")))
for endpoint, rate, wilson in [('any_family', gate['profile_sampler_ceiling']['any_family_rate'],
                                gate['profile_sampler_ceiling']['any_family_wilson95']),
                               ('complete_domain', gate['profile_sampler_ceiling']['complete_domain_rate'],
                                gate['profile_sampler_ceiling']['complete_domain_wilson95'])]:
    generators.append(dict(cell='profile_sampler_ceiling', arm='profile_sampler', condition='oracle_access',
                           endpoint=endpoint, cohort='ceiling', model_rate=rate, control_rate='', difference='',
                           ci_low=wilson[0], ci_high=wilson[1], qualified='',
                           n_attempts=gate['profile_sampler_ceiling']['n_sequences'], n_clusters='', resamples='',
                           verdict='oracle_access_ceiling_not_a_matched_control',
                           interval_level='95% Wilson',
                           unit='emitted sequence, Wilson interval on the raw rate',
                           **provenance(gate_path, 'profile_sampler_ceiling')))
save_csv('matched-generator-source.csv', generators)


# --- The capability map, arm by gate. ------------------------------------------------------
# One cell is one arm at one gate, carrying the outcome class that gate reached for that arm.
# A gate that read no model quantity, an arm a gate never admitted, and an arm whose per-cell
# record is not in this package are three different states and are drawn apart.
GATE_ORDER = [row['gate'] for row in capability]
LOCAL_LIKELIHOOD_RESOLVED = ['progen2-small', 'progen2-base', 'progen2-medium', 'progen2-large',
                             'progen2-xlarge', 'progen3-112m', 'progen3-3b', 'proteinglm-7b-clm',
                             'protgpt2', 'protgpt3-1.3b', 'rita-xl', 'instructprotein',
                             'prollama-stage-1', 'prollama']
LOCAL_STATE_RESOLVED = ['progen2-small', 'progen2-base', 'progen2-medium', 'progen2-large',
                        'progen2-xlarge', 'progen3-112m', 'progen3-3b', 'proteinglm-7b-clm']
LOCAL_STATE_UNRESOLVED = ['protgpt2', 'rita-xl', 'galactica-125m', 'galactica-1.3b', 'galactica-6.7b',
                          'galactica-30b', 'instructprotein', 'protgpt3-1.3b', 'prollama-stage-1',
                          'prollama', 'qwen2.5-32b']
# One declared roster order, by lineage and never by outcome.
MATRIX_ARMS = ['protgpt2', 'progen2-small', 'progen2-base', 'progen2-medium', 'progen2-large',
               'progen2-xlarge', 'progen3-112m', 'progen3-3b', 'protgpt3-1.3b', 'rita-xl',
               'proteinglm-7b-clm', 'galactica-125m', 'galactica-1.3b', 'galactica-6.7b',
               'galactica-30b', 'instructprotein', 'llama-2-7b', 'prollama-stage-1', 'prollama',
               'zymctrl', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'dialogpt-small',
               'bygpt5-small-en', 'bygpt5-base-en', 'bygpt5-medium-en', 'qwen2.5-0.5b',
               'qwen2.5-0.5b-instruct', 'qwen2.5-7b', 'qwen2.5-32b', 'qwen3-8b-base', 'llama-3.2-3b']


def interval_class(low, high):
    if low is None or high is None:
        return 'unstated'
    return 'supported' if low > 0 else ('not_detected' if high < 0 else 'unresolved')


def seed_class(above, below):
    return 'supported' if above == 3 else ('not_detected' if below == 3 else 'unresolved')


matrix = []


def cell(gate, arm, outcome, basis, provisional=False, **origin):
    matrix.append(dict(gate=gate, gate_order=GATE_ORDER.index(gate), arm=arm, outcome=outcome,
                       basis=basis, provisional=provisional, **origin))


native_index = {(row['arm'], row['endpoint']): row for row in native}
text_index = {(row['arm'], row['endpoint']): row for row in text}
for arm in MATRIX_ARMS:
    record = native_index.get((arm, 'model_minus_lookup')) or text_index.get((arm, 'model_minus_lookup'))
    if record is None:
        cell('ranking_beyond_profile', arm, 'not_eligible', 'no paired profile contrast on this arm')
        continue
    cell('ranking_beyond_profile', arm,
         interval_class(float(record['ci_low']), float(record['ci_high'])),
         'paired profile contrast interval',
         **provenance(record['source_path'], record['source_pointer']))
readout_index = {row['arm']: row for row in readout}
for arm in MATRIX_ARMS:
    record = readout_index.get(arm)
    if record is None:
        for gate in ('states_beyond_likelihood', 'states_beyond_matched'):
            cell(gate, arm, 'not_eligible', 'not on the readout anchor panel')
        continue
    for gate, field in (('states_beyond_likelihood', 'readout_minus_likelihood'),
                        ('states_beyond_matched', 'representation_increment')):
        cell(gate, arm, interval_class(record[f'{field}_ci_low'] and float(record[f'{field}_ci_low']),
                                       record[f'{field}_ci_high'] and float(record[f'{field}_ci_high'])),
             'paired readout interval at split seed 20260923', provisional=True,
             **provenance(record['source_path'], record['source_pointer']))
for arm in MATRIX_ARMS:
    if arm == 'zymctrl':
        for gate in ('likelihood_beyond_local', 'states_beyond_local'):
            cell(gate, arm, 'not_eligible', 'the EC-conditioned panel is a different support')
        continue
    cell('likelihood_beyond_local', arm,
         'supported' if arm in LOCAL_LIKELIHOOD_RESOLVED else 'unresolved',
         'resolved at all three split seeds on the full 33-arm panel',
         **transcribed('archive/docs/R1/gate-local-context.md', 'full-panel generalizability check, likelihood'))
    if arm in LOCAL_STATE_RESOLVED:
        outcome = 'supported'
    elif arm in LOCAL_STATE_UNRESOLVED:
        outcome = 'unresolved'
    else:
        outcome = 'unstated'
    cell('states_beyond_local', arm, outcome,
         'resolved at all three split seeds on the full 33-arm panel', provisional=True,
         **transcribed('archive/docs/R1/gate-local-context.md', 'full-panel generalizability check, representation'))
interaction_index = {(row['arm'], row['endpoint']): row for row in interaction}
for arm in MATRIX_ARMS:
    for gate, endpoint, prov in (('first_order_nonadditivity', 'first_order_likelihood', False),
                                 ('pairwise_interaction', 'likelihood_over_matched', False)):
        if gate=='pairwise_interaction' and arm in full_residual['arms']:
            value=full_residual['arms'][arm]
            outcome='supported' if value['resolved_positive'] else ('negative' if value['resolved_negative'] else 'unresolved')
            cell(gate,arm,outcome,'seed-mean33-arm simultaneous band; label-assisted target',
                 **provenance(full_residual_path,f'arms.{arm}'))
            continue
        record = interaction_index.get((arm, endpoint))
        if record is None:
            cell(gate, arm, 'not_eligible', 'not on the pre-declared 33-arm interaction panel')
            continue
        cell(gate, arm, seed_class(int(record['seeds_above_zero']), int(record['seeds_below_zero'])),
             'resolved at all three split seeds', provisional=prov,
             **provenance(record['source_path'], record['source_pointer']))
for gate, reason in (('higher_order', 'the endpoint sits below its own noise floor, so no model quantity was read'),
                     ('structural_contact', 'the contrast does not resolve on its control side, so no model quantity was read')):
    for arm in MATRIX_ARMS:
        cell(gate, arm, 'measurement_limited', reason,
             **transcribed('archive/docs/R3/gate-higher-order-extended.md' if gate == 'higher_order'
                           else 'archive/docs/R3/gate-structure-contact.md', 'gate verdict'))
context_index = {row['arm']: row for row in context}
for arm in MATRIX_ARMS:
    record = context_index.get(arm)
    if record is None:
        cell('related_context', arm, 'not_eligible', 'not in the related-context cohort')
        continue
    verdict = record['verdict']
    cell('related_context', arm,
         'supported' if verdict == 'in_context_relatedness_beyond_composition_and_local_copying'
         else ('not_detected' if verdict == 'no_gain_at_this_budget' else 'unresolved'),
         'joint criterion in both strata',
         **provenance(record['source_path'], record['source_pointer']))
strata_index = {row['arm']: row for row in strata
                if row['channel'] == 'stability' and row['stratification'] == 'full_support'}
for arm in MATRIX_ARMS:
    record = strata_index.get(arm)
    if record is None:
        cell('retrieval_strata', arm, 'not_eligible', 'not on the stability stratification panel')
        continue
    cell('retrieval_strata', arm, seed_class(int(record['seeds_above_zero']), int(record['seeds_below_zero'])),
         'whole-support increment resolved at all three split seeds',
         **provenance(record['source_path'], record['source_pointer']))
generator_index = {}
for record in generators:
    if record['endpoint'] != 'complete_domain' or record['cohort'] != 'fragment':
        continue
    generator_index.setdefault(record['arm'], []).append(record)
for arm in MATRIX_ARMS:
    records = generator_index.get(arm)
    if not records:
        cell('generation_products', arm, 'not_eligible', 'no generation cell for this arm')
        continue
    verdicts = {record['verdict'] for record in records}
    if 'satisfied_beyond_every_qualified_matched_generator' in verdicts:
        outcome = 'supported'
    elif verdicts == {'not_detected_beyond_any_qualified_matched_generator'}:
        outcome = 'not_detected'
    else:
        outcome = 'unresolved'
    cell('generation_products', arm, outcome, 'complete-domain endpoint against every qualified generator',
         **provenance(records[0]['source_path'], records[0]['source_pointer']))
assert len(matrix) == len(GATE_ORDER) * len(MATRIX_ARMS), (len(matrix), len(GATE_ORDER), len(MATRIX_ARMS))
save_csv('gate-matrix-source.csv', matrix)


# --- The higher-order record, deposition by property. --------------------------------------
# Three properties a qualified order-3 measurement needs, read at that order, and the failure
# label the record itself assigns. No measured value enters; this is the survey's own table.
HIGHER_ORDER_RECORD = [
    ('MegaScale', True, False, True, 'order'),
    ('MGnify Stability', False, False, True, 'completeness'),
    ('ProteinGym', True, True, False, 'channel coverage'),
    ('Escobedo 2025', False, True, True, 'breadth'),
    ('SKEMPI v2', True, True, False, 'corner coverage'),
]
record = []
for name, breadth, completeness, channel, failure in HIGHER_ORDER_RECORD:
    for prop, satisfied in (('breadth', breadth), ('completeness', completeness),
                            ('channel coverage', channel)):
        record.append(dict(deposition=name, property=prop, satisfied=satisfied,
                           failure_label=failure,
                           **transcribed('archive/docs/R3/higher-order-qualification-harness.md',
                                         'five-deposition specification table')))
assert len(record) == 15
save_csv('higher-order-record-source.csv', record)


# --- The readout reassessment's depth profile. ---------------------------------------------
# Blocks resolved above zero at every declared split seed of their panel, on the two informative
# axes, with the falsification stratum's best cell beside them, read from the panel aggregate the
# sweep's 54 cell reports were summarised into.  ProtGPT2's own panel declares one seed, so its
# six blocks carry no seed-consistency statement and are not drawn.
AXIS_NAME = {'depth': 'pooled', 'pos': 'position_resolved'}
panel_depth, depth_path = read_repo('results/R2/readout_depth_panel_20260925/'
                                    'readout_depth_panel.json')
depth = []
for axis_key, axis_name in AXIS_NAME.items():
    selected = [row for row in panel_depth['seed_consistent'] if row['stratum'] == 'native_sequence'
                and row['axis'] == axis_key and row['seeds'] == 3]
    for row in sorted(selected, key=lambda entry: -entry['minimum']):
        depth.append(dict(arm=row['arm'], panel=row['panel'], axis=axis_name, block=row['depth'],
                          blocks=row['blocks'], relative_depth=row['relative_depth'],
                          increment_min=round(row['minimum'], 6), increment_max=round(row['maximum'], 6),
                          interface='native_protein', resolved_at_every_seed=True,
                          **provenance(depth_path,
                                       f"seed_consistent, {axis_key} axis, blocks resolved at every declared seed")))
falsified = panel_depth['falsification']['literal_text_AA']
best = falsified['best_cell']
best_cell = next(cell for cell in panel_depth['cells']
                 if cell['arm'] == best['arm'] and cell['panel'] == best['panel'])
best_blocks = len(best_cell['seeds'][best['seed']]['axes'][best['axis']]['profile'])
depth.append(dict(arm=best['arm'], panel=best['panel'], axis=AXIS_NAME[best['axis']],
                  block=best['depth'], blocks=best_blocks,
                  relative_depth=round(best['depth'] / (best_blocks - 1), 4),
                  increment_min=round(falsified['best_point'], 6),
                  increment_max=round(falsified['best_point'], 6),
                  interface='literal_text_AA', resolved_at_every_seed=False,
                  **provenance(depth_path,
                               'falsification, literal amino-acid string stratum, largest point estimate')))
assert len(depth) == 34 and sum(row['interface'] == 'native_protein' for row in depth) == 33
save_csv('depth-profile-source.csv', depth)

# Current effect-size displays: retained predictions only, with target versions explicit.
data,path=read_repo('results/R5/local_context_20260923/20260923233257_e429ce7f31e4/lcgp_admission/local_context_measurement.json')
local=[]
for cell,block in data['summaries'].items():
    arm,seed=cell.split('/')
    for endpoint in ['increment_M_C_P_wall','increment_R_C_P_wall','C_P_wall_spearman','C_P_spearman']:
        local.append(estimate(arm,endpoint,block[endpoint],path,f'summaries.{cell}.{endpoint}',split_seed=seed))
save_csv('local-increments-source.csv',local)
external=[]
for file in sorted((ROOT/'results/R1/external_confirmation_20260924/fits').glob('fit_*.json')):
    data,path=read_repo(file.relative_to(ROOT).as_posix())
    for seed,block in data['per_seed'].items():
        for endpoint in ['primary_likelihood','primary_likelihood_spearman']:
            v=block[endpoint]
            external.append(dict(arm=data['arm'],endpoint=endpoint,split_seed=seed,estimate=v['point'],
                ci_low=v['interval'][0],ci_high=v['interval'][1],n_units=v['groups'],
                n_domains=data['effective_units']['domains'],n_variants=data['effective_units']['variants'],
                unit=v['unit'],**provenance(path,f'per_seed.{seed}.{endpoint}')))
assert len({r['arm'] for r in external})==33
save_csv('external-abundance-source.csv',external)
data,path=read_repo('results/R2/repr_behaviour_gap_20260926/gap_direct_contrast.json')
gaps=[]
for arm,seeds in data['gap_arms'].items():
    for seed,block in seeds.items():
        gaps.append(estimate(arm,'controlled_representation_minus_likelihood',block['d'],path,
            f'gap_arms.{arm}.{seed}.d',split_seed=seed,baseline=block['baseline_identity']['control_set']))
save_csv('controlled-gap-source.csv',gaps)
residual=[]
for arm in ['protgpt2','progen3-3b','prollama']:
    data,path=read_repo(f'results/R3/pairwise_residual_nested_20260927/residual_{arm}.json')
    assert data['schema']=='pairwise_residual_likelihood_interaction_v2'
    for seed,block in data['seeds'].items():
        v=block['mse_reduction_kcal2']
        assert v['interval'][0]<=0<=v['interval'][1]
        residual.append(dict(arm=arm,split_seed=seed,target='nested_adjusted',estimate=v['point'],
            ci_low=v['interval'][0],ci_high=v['interval'][1],groups=data['groups'],cycles=data['cycles'],
            **provenance(path,f'seeds.{seed}.mse_reduction_kcal2')))
        v=panel_data['panel']['all']['C_G_T+M|C_G_T']['arms'][arm]['per_seed'][seed]
        residual.append(dict(arm=arm,split_seed=seed,target='historical_raw',estimate=v['mse_reduction_kcal2'],
            ci_low=v['interval'][0],ci_high=v['interval'][1],groups=data['groups'],cycles=data['cycles'],
            **provenance(interaction_path,f'panel.all.C_G_T+M|C_G_T.arms.{arm}.per_seed.{seed}')))
assert sum(r['target']=='nested_adjusted' for r in residual)==9
save_csv('nested-residual-source.csv',residual)
contact=[]
for version,relative in [('corrected','results/R3/contact_bootstrap_corrected_20260927/epsilon_enrichment.json'),
                         ('historical','archive/logs/shared/gate_structure_contact_20260924/epsilon_enrichment.json')]:
    data,path=read_repo(relative)
    for definition,block in data['supports'][data['primary_support']]['definitions'].items():
        for endpoint,weightings in block['endpoints'].items():
            for weighting,v in weightings.items():
                if version=='corrected': assert not v['excludes_zero']
                contact.append(dict(version=version,definition=definition,endpoint=endpoint,weighting=weighting,
                    estimate=v['difference'],ci_low=v['interval'][0],ci_high=v['interval'][1],
                    n_contact=v['contact_site_pairs'],n_control=v['control_site_pairs'],
                    **provenance(path,f'supports.{data["primary_support"]}.definitions.{definition}.endpoints.{endpoint}.{weighting}')))
assert sum(r['version']=='corrected' for r in contact)==30  # pooled definitions; 14 method-stratum contrasts stay in the receipt
save_csv('contact-sensitivity-source.csv',contact)

# The two context comparators use exactly the shared mutation-ranking support.
data,path=read_repo('results/R5/context_profile_increment_20260923/20260923075017_68b04c315654/context_profile_increment/profile_increment.json')
rescue=[]
for block in data['common_support']['profile_increments']:
    for endpoint in ['delta_spearman','homolog_minus_no_context_spearman']:
        v=block['context_rescue']['strata']['pooled']['metrics'][endpoint]
        rescue.append(estimate(block['arm'],endpoint,v,path,
            f'common_support.profile_increments[arm={block["arm"]}].context_rescue.strata.pooled.metrics.{endpoint}'))
save_csv('context-ranking-source.csv',rescue)

receipt={'scope':'Lightweight extraction and plotting of completed frozen aggregate results; no new model runs or bootstrap.',
         'source_sha256':SOURCES,'source_tables':sorted(p.name for p in OUT.glob('*.csv')),
         'excluded':['Structure-predictor confidence endpoints, including ESMFold v1','Cross-measure association coefficients','Incomplete model runs'],
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'source-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'native_rows':len(native),'text_rows':len(text),'generation_arms':len(gen),'conditional_cells':len(conditional),'context_rows':len(context),'sources':len(SOURCES)},indent=2))

runpy.run_path(str(HERE / "render_ncs_figures.py"), run_name="__main__")
