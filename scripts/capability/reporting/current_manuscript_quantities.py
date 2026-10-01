"""Source mappings for the completed endpoint sensitivities.

Only read retained receipts and declarations; no fitting or resampling occurs.
The main builder owns the shared ledger and older campaign mappings.
"""
from src.capability.core.evidence_ledger import Quantity, resolve


def add_current_quantities(ledger):
    def emit(path, pointer, *, identifier, claim, family, support, unit,
             kind='support_count', value=None, interval=None, draws=None,
             resampling_unit=None, reason=None, interval_kind='95% percentile', seed=()):
        payload = ledger.artifacts.json(path)
        if value is None:
            value = resolve(payload, pointer)
        ledger.add(Quantity(id=identifier, claim=claim, family=family,
            value=float(value), unit=unit, kind=kind, support_id=support,
            interval=interval, interval_kind=interval_kind if interval else None,
            resampling_unit=resampling_unit, resampling_draws=draws,
            no_interval_reason=reason, seed_set=seed,
            source_path=path, source_sha256=ledger.artifacts.sha256(path), source_pointer=pointer))

    support = ledger.declare_support('pairwise_nested_selected',
        'Three selected raw-positive arms, three splits, 64 family groups and 8192 double-mutant cycles; label-assisted sensitivity')
    intervals = []
    for arm in ('progen3-3b', 'prollama', 'protgpt2'):
        path = f'results/R3/pairwise_residual_nested_20260927/residual_{arm}.json'
        payload = ledger.artifacts.json(path)
        if payload['schema'] != 'pairwise_residual_likelihood_interaction_v2':
            raise ValueError('only nested residual v2 is admissible')
        for seed, row in payload['seeds'].items():
            item = row['mse_reduction_kcal2']
            intervals.append(item['interval'])
            emit(path, ('seeds', seed, 'mse_reduction_kcal2'),
                identifier=f'nested_residual/{arm}/{seed}',
                claim=f'{arm} nested measured-singles-adjusted residual likelihood squared error increment',
                family='nested_residual', support=support, unit='kcal^2/mol^2', kind='estimate',
                value=item['point'], interval=tuple(item['interval']), draws=2000,
                resampling_unit='family group', seed=(int(seed),))
    # Draw count is the declared protocol used by this script, not inferred from CI endpoints.
    from src.capability.interactions.pairwise_epistasis import BOOTSTRAP_DRAWS
    if BOOTSTRAP_DRAWS != 2000:
        raise ValueError('residual resampling declaration changed; review frozen receipts')
    emit(path, ('seeds', '<all selected arms: interval count>'),
        identifier='nested_residual/selected_interval_count', claim='nested residual selected-arm intervals spanning zero',
        family='nested_residual', support=support, unit='intervals',
        value=sum(low <= 0 <= high for low, high in intervals))

    path = 'results/R3/contact_bootstrap_corrected_20260927/epsilon_enrichment.json'
    payload = ledger.artifacts.json(path)
    support = ledger.declare_support('contact_corrected_primary',
        'Indel-excluded primary contact support, 63 groups, with matched and method-stratum sensitivities')
    primary = payload['supports'][payload['primary_support']]
    def collect(node):
        if isinstance(node, dict):
            if 'difference' in node and node.get('interval') is not None:
                return [node]
            return [item for child in node.values() for item in collect(child)]
        return []
    contrasts = collect(primary['definitions']) + collect(primary['method_strata']) + collect(primary['flagged_design_backgrounds_removed'])
    emit(path, ('supports', payload['primary_support'], '<definitions, method_strata and flagged_design_backgrounds_removed>'),
        identifier='structure_contact/current_primary_intervals', claim='current primary-support contact intervals spanning zero',
        family='structure_contact', support=support, unit='intervals',
        value=sum(v['interval'][0] <= 0 <= v['interval'][1] for v in contrasts))
    path = 'results/R3/contact_bootstrap_corrected_20260927/endpoint_reaudit.json'
    for i, row in enumerate(ledger.artifacts.json(path)['estimates']):
        if row.get('difference_kcal_mol') is None:
            continue
        emit(path, ('estimates', f'[{i}]'), identifier=f'structure_contact/reaudit/{row["name"]}',
            claim=row['name'].replace('_', ' ') + ' contact minus non-contact mean absolute epsilon',
            family='structure_contact', support=support, unit='kcal/mol', kind='estimate',
            value=row['difference_kcal_mol'], interval=tuple(row['interval_95_kcal_mol']),
            draws=row['bootstrap_draws'], resampling_unit=row['bootstrap_unit'])

    path = 'results/R5/context_profile_increment_20260923/20260923075017_68b04c315654/context_profile_increment/profile_increment.json'
    payload = ledger.artifacts.json(path)
    support = ledger.declare_support('context_four_arm_intersection',
        'Four checkpoint intersection of 164 mutation assays, 134 wild-type families and 20992 variants')
    for field, unit in [('n_assays','assays'), ('n_families','family groups'), ('n_variants','variants')]:
        emit(path, ('common_support','support',field), identifier=f'context/intersection/{field}',
            claim='four-checkpoint context mutation intersection ' + field, family='context', support=support, unit=unit)
    for i, row in enumerate(payload['common_support']['profile_increments']):
        for metric, item in row['context_rescue']['strata']['pooled']['metrics'].items():
            if not isinstance(item, dict) or 'point' not in item or 'interval' not in item:
                continue
            emit(path, ('common_support','profile_increments',f'[{i}]','context_rescue','strata','pooled','metrics',metric),
                identifier=f'context/intersection/{row["arm"]}/{metric}', claim=f'{row["arm"]} context {metric.replace("_", " ")}',
                family='context', support=support, unit='dimensionless Spearman', kind='estimate', value=item['point'],
                interval=tuple(item['interval']), draws=payload['bootstrap'], resampling_unit='wild-type family', seed=(payload['seed'],))

    path = 'results/R1/external_confirmation_20260924/fits/fit_progen3-3b.json'
    support = ledger.declare_support('domain_abundance_fitted', '109568 variants in 428 domains and 96 groups on the fitted domain-abundance support')
    for field,unit in [('variants','variants'),('domains','domains'),('groups','family groups')]:
        emit(path, ('effective_units',field), identifier=f'domainome/fitted/{field}', claim='domain-abundance fitted support '+field,
            family='domainome', support=support, unit=unit)

    path = 'results/R6/generation_length_sensitivity_20260926/length_sensitivity.json'
    payload = ledger.artifacts.json(path)
    support = ledger.declare_support('generation_length_all', '16000 model and fragment attempts, 13980 frozen near-duplicate sequence groups')
    for field, unit in [('n_attempts','attempts'),('n_clusters','groups')]:
        emit(path, ('panel',field), identifier=f'generation_length/{field}', claim='length sensitivity all attempts '+field,
            family='generation_length', support=support, unit=unit)
    restricted = ledger.declare_support('generation_length_bins', '14190 attempts within four declared residue-length bins, fragment-distribution reweighting')
    emit(path, ('reference_distribution','n_attempts_inside_bins'), identifier='generation_length/inside_bins',
        claim='attempts within four declared length bins', family='generation_length', support=restricted, unit='attempts')
    for endpoint,row in payload['panel']['endpoints'].items():
        item=row['all_attempts']
        emit(path, ('panel','endpoints',endpoint,'all_attempts'), identifier=f'generation_length/all/{endpoint}',
            claim='pooled model-minus-fragment all-attempt '+endpoint.replace('_',' '), family='generation_length',
            support=support, unit='dimensionless rate difference', kind='estimate', value=item['difference'],
            interval=tuple(item['ci97_5']), interval_kind='97.5% percentile', draws=item['resamples'],
            resampling_unit=item['unit'], seed=(item['seed'],))
        item=row['length_reweighted']
        emit(path, ('panel','endpoints',endpoint,'length_reweighted','model_minus_fragment'),
            identifier=f'generation_length/reweighted/{endpoint}', claim='length reweighted fragment distribution descriptive differences '+endpoint.replace('_',' '),
            family='generation_length', support=restricted, unit='dimensionless rate difference', kind='estimate',
            reason=item['interval_reason'])
    # The field name is an explicit interval-level declaration in the receipt.
    emit(path, ('panel','endpoints','any_family','all_attempts','ci97_5','<level>'),
        identifier='generation_length/interval_percent', claim='length sensitivity percentile interval level',
        family='generation_length', support=support, unit='percent', kind='constant',
        value=float('97.5') if 'ci97_5' in payload['panel']['endpoints']['any_family']['all_attempts'] else None)

    path = 'results/R5/remote_acquisition_20260925/aggregation_endpoint_audit.json'
    payload=ledger.artifacts.json(path)
    support=ledger.declare_support('aggregation_acquisition', 'MGnify-derived aggregation acquisition overlap and shared-library agreement')
    for identifier,pointer,claim in [
        ('excluded',('overlap_exclusion','excluded_sequences'),'aggregation sequences overlapping staged cohort'),
        ('total',('uniref50_bands','populations','every_quantified_domain','all','queries'),'aggregation quantified sequences before overlap exclusion')]:
        emit(path,pointer,identifier='aggregation/'+identifier,claim=claim,family='aggregation',support=support,unit='sequences')
    row=payload['between_library_channel']['shared_set']['acidic_pH4_uncorrected']
    for key,item in row.items():
        if 'ratio' in key and isinstance(item,dict) and 'ci95' in item:
            emit(path,('between_library_channel','shared_set','acidic_pH4_uncorrected',key),
                identifier='aggregation/acidic/'+key,claim='acidic-condition shared-to-discordance before fitted correction, 825 shared proteins in 385 family groups',
                family='aggregation',support=support,unit='dimensionless ratio',kind='estimate',
                value=item['point'],interval=tuple(item['ci95']),draws=row['draws'],resampling_unit=row['resampling_unit'])

    path = 'results/R1/progen3_batch_closeout_20260927/progen3_batch1_vs_batch16.json'
    payload=ledger.artifacts.json(path)
    support=ledger.declare_support('progen3_batch_contrast', '217 assays and 174 wild-type families, batch size one minus sixteen, fixed checkpoint')
    for arm,row in payload['arms'].items():
        item=row['model_spearman_batch1_minus_batch16']
        emit(path,('arms',arm,'model_spearman_batch1_minus_batch16'),identifier=f'batch_contrast/{arm}',
            claim=f'{arm} batch-one-minus-batch-sixteen Spearman differences',family='batch_contrast',support=support,
            unit='dimensionless Spearman',kind='estimate',value=item['point'],interval=tuple(item['interval']),
            draws=item['resamples'],resampling_unit=item['unit'],seed=(payload['bootstrap']['seed'],))

    path='results/R6/generation_evidence_20260905/analysis/20260904232436_00129607e3c7/main/generation_biology_analysis.json'
    payload=ledger.artifacts.json(path)
    support='conditional_class_cells'
    for arm in ('zymctrl','prollama'):
        allrows=[r for r in payload['profile_ledger'] if r['role']=='generation' and r['arm']==arm]
        rows=[r for r in allrows if r['condition']=='requested' and r['primary_class']]
        for name,value,unit,claim in [
            ('eligible_hits',sum(r['n_target_profile'] for r in rows),'attempts','eligible requested target hits'),
            ('eligible_attempts',sum(r['n_attempts'] for r in rows),'attempts','eligible requested attempts'),
            ('total_attempts',sum(r['n_attempts'] for r in allrows),'attempts','conditional attempts per checkpoint'),
            ('minimum_length',min(r['length_quantiles'][0] for r in rows),'residues','requested output minimum length'),
            ('maximum_length',max(r['length_quantiles'][-1] for r in rows),'residues','requested output maximum length')]:
            emit(path,('profile_ledger',f'<role=generation,arm={arm}; {name}>'),identifier=f'conditional/{arm}/{name}',
                claim=f'{arm} {claim}',family='conditional_generation',support=support,unit=unit,kind='census',value=value)

    path='archive/logs/R6/gate_generative_control/pool/pool_receipt.json'
    emit(path,('reservoir_per_stratum',),identifier='generative_control/reservoir_per_stratum',
        claim='canonical records retained per length stratum',family='generative_control',support='generation_length_all',
        unit='records',kind='constant')
    path='results/shared/designed_referent_20260813/cohort_summary.json'
    rows=ledger.artifacts.json(path)['variant_floor_sweep']
    index=next(i for i,row in enumerate(rows) if row['min_variants']==30)
    support=ledger.declare_support('designed_referent_full', '130 zero-hit designs in 40 design series and 266 natural wild types in 124 groups; full census')
    for field,unit in [('design_variants','variants'),('natural_variants','variants'),('natural_clusters_scored','groups')]:
        emit(path,('variant_floor_sweep',f'[{index}]',field),identifier=f'designed_referent/{field}',
            claim='design and natural cohort '+field.replace('_',' '),family='designed_referent',support=support,unit=unit)

    # Older rounded display summaries remain bound to their canonical records.
    # Unique anchored captures make a moved or ambiguous claim fail explicitly.
    import re
    def record(path, pattern, identifier, claim, family, unit, *, kind='census', interval=False,
               support=None, draws=None, resampling_unit=None, scale=1.0):
        matches=list(re.finditer(pattern,ledger.artifacts.text(path)))
        if len(matches)!=1:
            raise ValueError(f'{identifier}: expected one record anchor, found {len(matches)}')
        values=[float(v.replace(',','').replace('−','-'))*scale for v in matches[0].groups()]
        support=support or ledger.declare_support(identifier.replace('/', '_')+'_record',claim+'; support specified by the canonical record')
        ledger.add(Quantity(id=identifier,claim=claim,family=family,value=values[0],unit=unit,kind=kind,
            support_id=support,interval=tuple(values[1:]) if interval else None,
            interval_kind='95% percentile' if interval else None,resampling_draws=draws,
            resampling_unit=resampling_unit,source_kind='gate_record' if path.endswith('.md') else 'artifact',source_path=path,
            source_sha256=ledger.artifacts.sha256(path),source_pointer=('<anchor>',pattern)))
    for identifier,pattern,claim,unit in [
        ('designed_doubles',r'It declares ([\d,]+) rows as double-mutant designs','MGnify designed doubles','states'),
        ('measured_states',r'Its ([\d,]+) measured states each carry','MGnify measured states','states'),
        ('sample_neighbours',r'at large, (\d+) have at least one','sampled MGnify states with a single-substitution neighbour','states'),
        ('skempi_cycles',r'\*\*(\d+) order-2 contrasts in 78 complexes','SKEMPI complete order-2 contrasts','cycles'),
        ('fyn_identity',r'identical positions, ([\d.]+)% identity','FYN SH3 suppressor and core identity','percent')]:
        record('archive/docs/R3/higher-order-qualification-harness.md',pattern,'higher_order/acquisition/'+identifier,claim,'higher_order',unit)
    for field,pattern in [('total',r'The recovered evidence is ([\d,]+) prediction rows'),
                          ('ok',r'Of these, ([\d,]+) carry status `ok`')]:
        record('archive/docs/R6/structure-instrument-reconciliation.md',pattern,'structure_instrument/'+field,
            'structure reconciliation prediction rows '+field,'structure_instrument','rows')
    record('archive/docs/shared/recomputation-pipeline.md',r'\*\*(\d+) fits in total, no failures\*\*',
        'depth_contrast/total_fits','four-cohort depth-transfer fits','depth_contrast','fits')
    record('archive/docs/R3/pairwise-baseline-readiness.md',r'\| Effective depth per site \(Neff / length\) \| 0.107 \| 0.845 \| ([\d.]+) \|',
        'pairwise/q/median_effective_sequences_per_site','median effective sequences per site, coupling depth','pairwise','sequences per site')
    for path,family,pattern in [
        ('archive/docs/R1/external-confirmation.md','domainome',r'The resolved increments are ([\d.]+)% to [\d.]+%'),
        ('archive/docs/R4/gate-stability.md','stability_controls',r'the resolved increments are ([\d.]+)% to [\d.]+%')]:
        record(path,pattern,family+'/resolved_remaining_error_share/min','resolved improvements percent of baseline remaining squared error',family,'percent',kind='range_bound')
        pattern=pattern.replace('([\\d.]+)% to [\\d.]+%','[\\d.]+% to ([\\d.]+)%')
        record(path,pattern,family+'/resolved_remaining_error_share/max','resolved improvements percent of baseline remaining squared error',family,'percent',kind='range_bound')
    path='archive/docs/R1/external-confirmation.md'
    for bound,pattern in [('min',r'qualified controls remove \*\*([\d.]+)% to [\d.]+%\*\*'),
                          ('max',r'qualified controls remove \*\*[\d.]+% to ([\d.]+)%\*\*')]:
        record(path,pattern,'domainome/control_share/'+bound,'qualified controls percent of no-effect null squared error','domainome','percent',kind='range_bound')
    record(path,r'reduces the null\x27s squared error by \*\*\+([\d.]+) \[\+([\d.]+), \+([\d.]+)\]\*\*',
        'domainome/qualified_control_contribution','primary-seed qualified control improvement squared normalized fitness',
        'domainome','squared normalized fitness',kind='estimate',interval=True,draws=2000,resampling_unit='family group')
    record('archive/docs/shared/cross-measure-association.md',r'ProGen3-3B: (\d+)/800 accepted',
        'generation_census/progen3-3b/accepted','ProGen3-3B compiler-accepted products','generation_census','attempts')
    record('archive/docs/shared/cross-measure-association.md',r'accepted, (\d+) budget-censored prefixes',
        'generation_census/progen3-3b/censored','ProGen3-3B budget-censored continuations','generation_census','attempts')
    # The exact lineage-resample enumeration is not transcribed here: the retained
    # artifact behind it is in this repository and `lineage_correlation/cluster_resampling`
    # reads the same point, interval and 24,310 compositions out of it at full precision.

    # Design constants are read from implementation declarations, never matched
    # to a coincidentally equal fitted estimate from another endpoint.
    record('scripts/capability/position/analyse_position_terms.py',r'MEASURABLE_ALIGNED_FRACTION = ([\d.]+)',
        'position_terms/alignment_gate','declared minimum alignment fraction gate','position_terms','dimensionless fraction',kind='constant')
    record('src/capability/mutation/higher_order_cycle.py',r'return float\(sum\(\((-[\d]+)\) \*\*',
        'higher_order/cycle_sign_base','alternating sign base in order-k cycle formula','higher_order','dimensionless',kind='constant')
    path='results/R1/position_terms_20260926/position_alignment_panel.json'
    arms=ledger.artifacts.json(path)['arms']
    index=next(i for i,row in enumerate(arms) if row['arm']=='qwen2.5-32b')
    emit(path,('arms',f'[{index}]','anchor','aligned_fraction'),
        identifier='position_terms/qwen2.5-32b/alignment_fraction',claim='Qwen2.5-32B alignment fraction',
        family='position_terms',unit='dimensionless fraction',kind='census',
        support=ledger.declare_support('position_alignment_anchor_cohort',
            'the anchor cohort of the 33-arm token-alignment census: every scored variant of an arm '
            'classified aligned or unaligned against its wild type, over all variants; full census'))

    record('src/capability/context/local_context.py',r'RECEPTIVE_FIELD_RADII[^=]*= \((\d+),',
        'local_context/negative_window_offset','seven-residue positional window minimum relative offset',
        'local_context','residues',kind='constant',scale=-1.0)
