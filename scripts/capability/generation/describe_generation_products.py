#!/usr/bin/env python3
"""Descriptive length/composition/repetition census of complete native products."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from src.capability.core.io import write_json,sha256_file

METRICS={
    'residue_length':'residues',
    'composition_entropy':'nats/residue',
    'longest_identical_run_fraction':'fraction of sequence residues',
    'unique_3mer_fraction':'distinct residue 3-mers / number of overlapping 3-mer positions',
}


def sequence_descriptors(sequence):
    if not sequence or set(sequence)-set('ACDEFGHIKLMNPQRSTVWY'):
        raise ValueError('complete product must be nonempty canonical residues')
    length=len(sequence)
    frequencies=[count/length for count in Counter(sequence).values()]
    longest=max(sum(1 for _ in group) for _,group in itertools.groupby(sequence))
    return dict(residue_length=length,
                composition_entropy=-sum(p*math.log(p) for p in frequencies),
                longest_identical_run_fraction=longest/length,
                unique_3mer_fraction=len({sequence[i:i+3] for i in range(length-2)})/(length-2) if length>=3 else None)


def summarize_values(rows,key):
    import numpy as np
    values=[r[key] for r in rows if r[key] is not None]
    return dict(n_defined=len(values),n_undefined=len(rows)-len(values),
                mean=float(np.mean(values)) if values else None,
                quantiles=dict(zip(('min','q25','median','q75','max'),
                                   np.quantile(values,[0,.25,.5,.75,1],method='linear').tolist())) if values else None)


def describe(rows):
    products=[]
    for index,row in enumerate(rows):
        if not isinstance(row.get('native_complete'),bool) or not isinstance(row.get('any_profile_hit'),bool):
            raise ValueError('missing native-completion or profile annotation')
        if not row['native_complete']:continue
        descriptors=sequence_descriptors(row['sequence'])
        if descriptors['residue_length']!=row['residue_length']:
            raise ValueError('stored residue length mismatch')
        products.append(dict(ledger_row=index,attempt_index=row.get('attempt_index'),
                             id=row.get('id'),sequence_sha256=hashlib.sha256(row['sequence'].encode()).hexdigest(),
                             any_profile_hit=row['any_profile_hit'],**descriptors))
    groups={'all_complete':products,
            'complete_with_pfam':[r for r in products if r['any_profile_hit']],
            'complete_without_pfam':[r for r in products if not r['any_profile_hit']]}
    return dict(n_attempts=len(rows),n_complete=len(products),n_excluded_incomplete=len(rows)-len(products),
                groups={name:dict(n_products=len(group),metrics={key:summarize_values(group,key) for key in METRICS})
                        for name,group in groups.items()},products=products)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger',action='append',required=True,help='label=path to an admitted diagnostic attempt ledger')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu',choices=['cpu'])
    args=p.parse_args()
    cells={}
    for item in args.ledger:
        label,filename=item.split('=',1)
        if label in cells:raise ValueError('duplicate ledger label')
        path=Path(filename)
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        cells[label]=dict(source=str(path),source_sha256=sha256_file(path),**describe(rows))
    write_json(args.out/'generation_product_descriptors.json',dict(
        status='complete',schema_version='generation_product_descriptors_v1',script_sha256=sha256_file(Path(__file__)),
        units=METRICS,cells=cells,
        quantile_method='linear interpolation at 0, 0.25, 0.5, 0.75, 1; observed cohort census, no inferential intervals',
        undefined='unique_3mer_fraction undefined for products shorter than 3 residues; empty groups have undefined summaries',
        limitation='Conditional on native completion and, in strata, Pfam recognition. Composition entropy and repetition descriptors are not biological quality thresholds. Short complete strings may reflect termination behavior but do not establish premature termination. These descriptive associations do not isolate causal mechanisms or a token-budget effect.'))


if __name__=='__main__':main()
