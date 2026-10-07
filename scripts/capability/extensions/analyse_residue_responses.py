#!/usr/bin/env python3
"""Prepare geometry-only pairs, validate retained responses, then analyse 4 or 3.

--input JSON: {assays: [{assay, cluster, protein, wildtype, mutants,
sequences, archive, packing, B, measured}]}. archive is an original NPZ;
packing is a JSON ordered list of states {sequence, ids, span, counts, offset}.
Paths resolve relative to the input. B/measurements are needed only for bins.
--structures accepts the structural site's JSON list (or {sites: [...]}); all
WT sites including exclusions must be supplied. Preparation needs only assay,
cluster, protein, wildtype and mutants, not archives or labels. Pair preparation
and contacts require the paired --coverage receipt to verify experimental provenance.
--mode packing --arm ARM accepts original mixed archives without packing paths,
exports tokenizer-only metadata, and writes a directly reusable response input.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import resource
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.capability.extensions.responses import (RECOVERY_REQUIREMENTS, RetainedResponses,
    analyse_contacts, evaluate_bins, mapped_pairs, prepare_structure_pairs, structure_sites, potential_distance_census)


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            sha.update(chunk)
    return sha.hexdigest()


def read_json(path):
    try:
        with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        raise ValueError(f'cannot read required source {path}: {error}') from error


def export_packing(rows, *, input_path, arm_name, output_path, ec_path=None, cohort_path=None):
    """Original tokenizer-only packing and retained-count checks; never forward."""
    from scripts.capability.position.analyse_position_terms import Packer, Retained
    from src.capability.readouts.readout_extraction import load_readout_arm, bind_readout_ec, validate_ec_conditioning
    arm = load_readout_arm(arm_name,None,device=None)
    if arm.model is not None:
        raise ValueError('packing export refuses a loaded model')
    conditioning = None
    if arm_name == 'zymctrl':
        if ec_path is None or cohort_path is None:
            raise ValueError('ZymCTRL packing needs --ec-conditioning and the original --cohort')
        cohort = read_json(cohort_path)
        conditioning = validate_ec_conditioning(read_json(ec_path),cohort,digest(cohort_path))
        wildtypes = {r['assay']:r['wildtype'] for r in cohort['assays']}
        if any(wildtypes.get(r['assay']) != r['wildtype'] for r in rows):
            raise ValueError('packing input WT identity disagrees with EC source cohort')
    directory = output_path.parent/(output_path.stem+'-states')
    directory.mkdir(exist_ok=False)
    exported, receipts = [], []
    seen = set()
    for row in rows:
        if row['assay'] in seen:
            raise ValueError('duplicate packing-export assay')
        seen.add(row['assay'])
        if conditioning is not None:
            if row['assay'] not in conditioning:
                raise ValueError('assay lacks genuine EC conditioning')
            bind_readout_ec(arm,conditioning[row['assay']])
        # A fresh cache per assay also prevents conditioning from sharing packing.
        packer = Packer(arm)
        archive = (input_path.parent/row['archive']).resolve()
        sequences = [row['wildtype'],*row['sequences']]
        with np.load(archive,allow_pickle=False) as data:
            retained = Retained(data)
            states = []
            for i, sequence in enumerate(sequences):
                packed = packer.checked(sequence,retained,i)
                states.append(dict(sequence=sequence,ids=packed['ids'],span=list(packed['span']),
                                   counts=packed['counts'].tolist(),offset=packed['offset']))
            validated = RetainedResponses(data,states,dict(wildtype=row['wildtype'],mutants=row['mutants'],sequences=row['sequences']))
        path = directory/(hashlib.sha256(row['assay'].encode()).hexdigest()+'.json')
        with path.open('x') as handle:
            json.dump(states,handle,allow_nan=False)
        exported.append(dict(row,archive=str(archive),packing=str(path.resolve())))
        receipts.append(dict(assay=row['assay'],archive_sha256=digest(archive),packing_sha256=digest(path),
                             original_states=len(states),selected_indices=validated.selected_indices,exclusions=validated.exclusions))
    return dict(schema='tokenizer_only_response_input_v1',arm=arm_name,assays=exported,
                packing_receipts=receipts,tokenizer_only=True,model_loaded=False,forward_performed=False,
                packing_contract='original analyse_position_terms.Packer.state/checked; retained counts/offsets and full archive identity/closure checked')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','census','packing','contacts','bins','requirements'),required=True)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--structures',type=Path)
    parser.add_argument('--coverage',type=Path,help='paired structural coverage.json, required for prepare/contacts')
    parser.add_argument('--arm',help='packing only: native arm tokenizer, no model loaded')
    parser.add_argument('--ec-conditioning',type=Path,help='ZymCTRL packing: genuine archived EC annotations')
    parser.add_argument('--cohort',type=Path,help='ZymCTRL packing: original EC-bound cohort')
    parser.add_argument('--out',type=Path,required=True,help='new JSON receipt; never overwrite')
    parser.add_argument('--draws',type=int,default=2000)
    parser.add_argument('--seeds',type=int,nargs='+',default=[20260923,20260924,20260925])
    parser.add_argument('--bootstrap-seed',type=int,default=20261006)
    args = parser.parse_args(argv)
    if args.draws < 1:
        parser.error('--draws must be positive')
    if args.out.exists():
        parser.error('refusing to overwrite output receipt')
    parent = args.out.parent
    parent.mkdir(parents=True,exist_ok=True)
    free = shutil.disk_usage(parent).free
    if free < 10 << 20:
        raise RuntimeError('insufficient disk space for response receipt')
    hashes = {}
    result: dict = {}
    if args.mode == 'requirements':
        result = dict(RECOVERY_REQUIREMENTS)
    else:
        if args.input is None:
            parser.error('--input is required')
        source = read_json(args.input)
        rows = source['assays']
        hashes[str(args.input)] = digest(args.input)
        structure = None
        structural_coverage = None
        if args.mode in ('prepare','contacts'):
            if args.structures is None or args.coverage is None:
                parser.error('--structures and --coverage are required for geometry preparation/contact analysis')
            structure = read_json(args.structures)
            if isinstance(structure,dict):
                structure = structure['sites']
            hashes[str(args.structures)] = digest(args.structures)
            structural_coverage = read_json(args.coverage)
            hashes[str(args.coverage)] = digest(args.coverage)
        proteins, contacts, predictive, coverage = [], [], [], []
        excluded_multis, excluded_assays, admission_receipts = [], [], []
        site_assays = {r['assay_id'] for r in structure} if structure is not None else None
        for row in rows:
            if args.mode in ('census','packing'):
                continue
            if args.mode in ('prepare','contacts') and site_assays is not None and row['assay'] not in site_assays:
                excluded_assays.append(row['assay'])
                continue
            sites, receipt = [], {}
            if structure is not None:
                sites, receipt = structure_sites(structure,assay=row['assay'],family=row['cluster'],wildtype=row['wildtype'],coverage=structural_coverage)
            coverage.append(dict(assay=row['assay'],**receipt))
            if args.mode == 'prepare':
                singles = [m for m in row['mutants'] if ':' not in m]
                excluded_multis.extend(dict(assay=row['assay'],mutation=m,reason='multiple_substitutions') for m in row['mutants'] if ':' in m)
                proteins.append(dict(assay=row['assay'],family=row['cluster'],protein=row.get('protein',row['assay']),
                                     wildtype=row['wildtype'],mutations=singles,sites=sites))
                continue
            archive = (args.input.parent/row['archive']).resolve()
            packing = (args.input.parent/row['packing']).resolve()
            # No summary fallback and no inference: missing sources fail explicitly.
            states = read_json(packing)
            hashes[str(archive)] = digest(archive)
            hashes[str(packing)] = digest(packing)
            with np.load(archive,allow_pickle=False) as data:
                retained = RetainedResponses(data,states,dict(wildtype=row['wildtype'],mutants=row['mutants'],sequences=row['sequences']))
                selected = retained.project(row)
            responses = selected['responses']
            status = 'admitted' if len(responses)>=3 else 'insufficient_assay_support'
            admission_receipts.append(dict(assay=row['assay'],original_variants=len(row['mutants']),
                                           selected_variants=len(responses),original_variant_indices=selected['original_variant_indices'],
                                           original_state_indices=selected['original_state_indices'],exclusions=selected['exclusions'],
                                           predictive_status=status))
            for response in responses:
                if args.mode == 'contacts':
                    contacts.append(dict(assay=row['assay'],family=row['cluster'],protein=row['protein'],
                                         mutation=response['mutation'],pairs=mapped_pairs(response,sites,row['wildtype'])))
            if args.mode != 'bins' or status=='admitted':
                predictive.append(dict(assay=row['assay'],cluster=row['cluster'],mutants=selected['mutants'],
                                       sequences=selected['sequences'],original_variant_indices=selected['original_variant_indices'],responses=responses,
                                       **{key:selected[key] for key in ('B','measured') if key in selected}))
        if args.mode == 'prepare':
            result = prepare_structure_pairs(proteins)
            result['excluded_multiple_substitutions'] = excluded_multis
            result['excluded_assays_not_in_site_table'] = excluded_assays
            result['baseline_nll_support'] = 'deferred until original retained WT NLL arrays recovered and validated'
            result['structural_confounds'] = 'receiver identity and RSA available in pair records; distance imbalance reported per matched stratum; token eligibility and baseline predictability not yet admitted'
        elif args.mode == 'census':
            result = potential_distance_census(rows)
        elif args.mode == 'packing':
            if not args.arm:
                parser.error('--arm is required for tokenizer-only packing export')
            result = export_packing(rows,input_path=args.input,arm_name=args.arm,output_path=args.out,
                                    ec_path=args.ec_conditioning,cohort_path=args.cohort)
            for path in (args.ec_conditioning,args.cohort):
                if path is not None:
                    hashes[str(path)] = digest(path)
        elif args.mode == 'contacts':
            result = analyse_contacts(contacts,draws=args.draws,seed=args.bootstrap_seed)
            result['response_coverage'] = [dict(assay=r['assay'],responses=r['responses']) for r in predictive]
        else:
            if len({r['cluster'] for r in predictive}) < 5:
                result = dict(analysis=3,inference=None,inference_status='insufficient assay/family support after biological exclusions',
                              supported_assays=len(predictive),supported_families=len({r['cluster'] for r in predictive}))
            else:
                result = evaluate_bins(predictive,seeds=tuple(args.seeds),draws=args.draws,bootstrap_seed=args.bootstrap_seed)
            result['response_coverage'] = [dict(assay=r['assay'],sequences=r['sequences'],
                                               original_variant_indices=r['original_variant_indices'],responses=r['responses']) for r in predictive]
        result['archive_admission'] = admission_receipts
        result['structure_coverage'] = coverage
    result['source_sha256'] = hashes
    result['analysis_sha256'] = digest(Path(__file__).resolve())
    result['library_sha256'] = digest(ROOT/'src/capability/extensions/responses.py')
    result['created_utc'] = datetime.now(timezone.utc).isoformat()
    result['resources'] = dict(cpu_only=True,disk_free_bytes=free,
                               peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with args.out.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(mode=args.mode,out=str(args.out),status=result.get('inference_status',result.get('response_status',result.get('status'))))))
    return result


if __name__=='__main__':
    main()
