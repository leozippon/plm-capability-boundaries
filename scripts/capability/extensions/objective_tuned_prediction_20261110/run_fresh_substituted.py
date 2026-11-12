import sys, argparse
from pathlib import Path
sys.path.insert(0, '.')
from scripts.capability.extensions import score_objective_tuned_prediction as S
from src.capability.extensions import objective_tuned_prediction as E
E.reuses_archive = lambda base: False   # score every arm from its fresh file, 1.3B included
D = Path('results/extensions/objective_tuned_prediction_20261110'); A = D / 'archive-rescore'
S.run(argparse.Namespace(device=None, scores=A / 'all-scores', dry_run=False, out=A / 'fresh-substituted-run'))
