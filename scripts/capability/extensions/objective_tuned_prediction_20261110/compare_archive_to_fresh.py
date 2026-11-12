import sys, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
from scripts.capability.extensions import score_objective_tuned_prediction as S
from src.capability.extensions import objective_tuned_prediction as E
from src.capability.context.profile_increment import standardized_rank
D = Path('results/extensions/objective_tuned_prediction_20261110'); A = D / 'archive-rescore'
table, data = S.load_table(), S.controls()
fresh = np.full(len(data['aids']), np.nan); sp = []
arch = data['features']['protgpt3-1.3b']
for a, vals in S.score_arm('protgpt3-1.3b', table, A / 'scores', None).items():
    idx = np.flatnonzero(data['aids'] == a); assert len(idx) == len(vals)
    fresh[idx] = standardized_rank(vals)
    from scipy.stats import spearmanr
    sp.append(spearmanr(fresh[idx], arch[idx])[0])
d = np.abs(fresh - arch)
cmp = dict(variants=int(len(d)), finite_fresh=int(np.isfinite(fresh).sum()), max_abs_rank_difference=float(d.max()),
           disagreeing_over_1e_6=int((d > 1e-6).sum()), min_assay_spearman=float(min(sp)), assays=len(sp))
json.dump(cmp, open(A / 'archive-vs-fresh.json', 'w'), indent=2); print(cmp)
bad = np.flatnonzero(d > 1e-6); wl = {r['assay']: len(r['wildtype']) for r in table}
per = {}
for i in bad: per[data['aids'][i]] = per.get(data['aids'][i], 0) + 1
lens = sorted(wl[a] for a in per)
cmp.update(assays_with_a_disagreement=len(per), max_per_assay=max(per.values()),
           wildtype_length_range_of_those=[lens[0], lens[-1]], median_length_of_those=lens[len(lens)//2],
           median_length_all=sorted(wl.values())[100],
           max_rank_difference_in_ranks=float(d.max() * 127))
json.dump(cmp, open(A / 'archive-vs-fresh.json', 'w'), indent=2); print(cmp)
