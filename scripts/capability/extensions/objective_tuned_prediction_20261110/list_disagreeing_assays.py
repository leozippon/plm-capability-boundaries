import sys, json
sys.path.insert(0, '.')
import numpy as np
from pathlib import Path
from scripts.capability.extensions import score_objective_tuned_prediction as S
from src.capability.context.profile_increment import standardized_rank
A = Path('results/extensions/objective_tuned_prediction_20261110/archive-rescore')
table, data = S.load_table(), S.controls()
arch = data['features']['protgpt3-1.3b']
fresh = S.score_arm('protgpt3-1.3b', table, A / 'scores', None)
checked = {c['assay'] for c in json.load(open('logs/operations/objective-tuned-prediction-20261110/reuse-equivalence.json'))['cells']}
for r in table:
    idx = np.flatnonzero(data['aids'] == r['assay'])
    n = int((np.abs(standardized_rank(fresh[r['assay']]) - arch[idx]) > 1e-6).sum())
    if n or r['assay'] in checked:
        print(r['assay'], len(r['wildtype']), 'disagree=%d' % n, 'CHECKED-EARLIER' if r['assay'] in checked else '')
