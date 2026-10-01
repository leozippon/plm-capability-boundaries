#!/usr/bin/env python3
"""Gate: is the residual model-side likelihood information long-range additive context?

This refits the flagship pairwise ladder of `fit_pairwise_epistasis.py` with two
extra declared blocks on identical rows, folds, weighting and label budget: the
global-context block X, which represents a per-position effect depending on
distant sequence content with no term reading both mutated sites, and the
bounded-receptive-field comparator W, whose coordinates read only residues within
the declared radius of their own mutated site. It then re-estimates the
first-order likelihood gain and the likelihood interaction gain over each
context baseline, and reports how much of the model's own first-order likelihood
difference each context baseline reconstructs on held groups.

No model inference runs here. The panel is assembled from the retained
label-free extraction artefacts by the flagship fit script's own loaders and
`build_panel`, so the rows, blocks and digests are that script's rather than a
second implementation of them, and the flagship per-arm record is read only to
refuse a run whose support or folds differ from the fit this gate must compose
with.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.context.global_context import (  # noqa: E402
    BLOCK_DECLARATION, BOUNDED_FEATURE_ORDER, BOUNDED_INDEX, GATE_CONTROL_SETS,
    GATE_DESIGNS, GLOBAL_FEATURE_ORDER, apply_indel_exclusion, cycle_context_features,
    declaration_digest, design_blocks, evaluate_gate, gate_compare)
from src.capability.interactions.pairwise_epistasis import (  # noqa: E402
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, PROJECTION_DIM, ROSTER, SPLIT_SEEDS,
    TOKENISATION_STRATUM, cycle_states, fold_identity, plan_digest, row_identity)
from src.capability.readouts.readout_analysis import row_weights  # noqa: E402


def flagship_module():
    """The admitted fit script, imported rather than copied.

    Its `load_profiles`, `load_arm` and `build_panel` are the single source of
    the rows and of every block the flagship ladder was fitted on; re-deriving
    them here would let this gate's support drift from the fit it must compose
    with. Importing the file executes only its declarations.
    """

    path = ROOT / 'scripts/capability/interactions/fit_pairwise_epistasis.py'
    spec = importlib.util.spec_from_file_location('fit_pairwise_epistasis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attach_context_blocks(plan: dict, groups: set, panel: dict) -> None:
    """Add the declared X and W blocks to a panel built by the flagship script.

    The plan is walked in the flagship's own order and filtered by the same group
    set, and every row's site-pair label is checked against the panel's before the
    block is attached, so a row-order divergence fails here instead of producing a
    silently misaligned column.
    """

    rows, labels = [], []
    for background in plan['backgrounds']:
        if background['group'] not in groups:
            continue
        wildtype, sequences = background['wildtype'], background['sequences']
        for cycle in background['cycles']:
            quartet = cycle_states(wildtype, [sequences[state] for state in cycle['states']])
            positions = tuple(cycle['positions'])
            rows.append(cycle_context_features(wildtype, quartet, positions))
            labels.append(f"{background['name']}:{positions[0]}-{positions[1]}")
    if labels != list(panel['site_pair']):
        raise ValueError('context rows are not aligned to the panel rows')
    values = np.asarray(rows, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite global-context block')
    panel['blocks']['X'] = values
    panel['blocks']['W'] = values[:, list(BOUNDED_INDEX)]


#: Tolerance in squared kcal/mol on a design this gate shares with the admitted
#: pairwise fit. The two runs are the same computation on the same rows, so the
#: only admissible difference is floating-point reduction order under a different
#: thread count; a real divergence in support, folds or recipe is orders larger.
SHARED_DESIGN_TOLERANCE = 1e-9

#: Designs this gate shares with the admitted pairwise fit, under both names.
SHARED_DESIGNS = (('CGT', 'C_G_T'), ('CGT+M', 'C_G_T+M'),
                  ('CGT_M1', 'C_G_T_M1'), ('CGT_M1+M', 'C_G_T_M1+M'))


def kish_effective_site_pairs(panel: dict) -> dict:
    """Kish effective number of site pairs, under both declared weightings.

    ``fitting_weights`` is the weighting every fit here actually uses: each group
    equal, each site pair equal inside a group. ``cycle_share`` weights a site
    pair by its share of its group's cycles, which is the convention behind the
    121.6 and 88.4 counts the readiness record states. Both are reported because
    they differ by about 6%, and only the first describes these fits.
    """

    weights = row_weights(panel['site_pair'], panel['group'])
    pairs = sorted(set(panel['site_pair']))
    masks = [panel['site_pair'] == pair for pair in pairs]
    groups = {pair: panel['group'][mask][0] for pair, mask in zip(pairs, masks)}
    counts = {pair: int(mask.sum()) for pair, mask in zip(pairs, masks)}
    per_group = {}
    for pair, group in groups.items():
        per_group[group] = per_group.get(group, 0) + counts[pair]
    n_groups = len(set(panel['group']))

    def kish(mass: np.ndarray) -> float:
        return float(mass.sum() ** 2 / (mass ** 2).sum())

    return {
        'fitting_weights': kish(np.asarray([weights[mask].sum() for mask in masks])),
        'cycle_share': kish(np.asarray([counts[pair] / (n_groups * per_group[groups[pair]])
                                        for pair in pairs])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expect-plan-sha256', required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--baseline-q', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--extraction', type=Path, required=True)
    parser.add_argument('--flagship-fit', type=Path, required=True,
                        help='the arm\'s admitted pairwise fit record, whose row and fold '
                             'identity digests this run must reproduce')
    parser.add_argument('--expect-declaration-sha256', required=True,
                        help='the global-context block declaration digest, stated in '
                             'archive/docs/R5/gate-global-context.md before any of these fits ran')
    parser.add_argument('--arm', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--bootstrap', type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument('--supports', default='all,q')
    parser.add_argument('--indel-exclusion', type=Path,
                        help='the declared insertion/deletion-construct exclusion over the '
                             'frozen cohort; required by any support named <base>_no_indel, '
                             'which is a sensitivity on rows that deliberately differ from '
                             'the admitted fit')
    args = parser.parse_args()

    if args.arm not in ROSTER:
        raise SystemExit(f'{args.arm} is not on the frozen roster')
    if declaration_digest() != args.expect_declaration_sha256:
        raise SystemExit('global-context declaration digest does not match the declared value')
    flagship = flagship_module()
    plan = json.loads(args.plan.read_text())
    if plan_digest(plan) != args.expect_plan_sha256:
        raise SystemExit('extraction plan digest does not match the declared value')
    cohort = json.loads(args.cohort.read_text())
    if flagship.digest(args.cohort) != plan['cohort_sha256']:
        raise SystemExit('cohort digest does not match the frozen plan')
    baseline = json.loads(args.baseline_q.read_text())
    if flagship.digest(args.baseline_q) != plan['provenance']['baseline_q']['sha256']:
        raise SystemExit('baseline Q digest does not match the frozen plan')
    profiles, profile_meta = flagship.load_profiles(args.profiles, plan)
    if profile_meta['plan_sha256'] != flagship.digest(args.plan):
        raise SystemExit('profile features were not built from this plan file')
    arm_data, manifest = flagship.load_arm(args.extraction / args.arm, args.arm, plan)
    admitted = json.loads(args.flagship_fit.read_text())
    if admitted.get('schema') != 'pairwise_epistasis_fit_v1' or admitted['arm'] != args.arm:
        raise SystemExit('flagship fit record is not this arm\'s admitted pairwise fit')
    exclusion = None
    if args.indel_exclusion is not None:
        exclusion = json.loads(args.indel_exclusion.read_text())
        if exclusion.get('cohort_sha256') != plan['cohort_sha256']:
            raise SystemExit('the indel exclusion was declared over a different cohort')
    if any(support.endswith('_no_indel') for support in args.supports.split(',')) \
            and exclusion is None:
        raise SystemExit('a _no_indel support needs --indel-exclusion')

    import numpy
    record = {'schema': 'global_context_gate_fit_v1', 'arm': args.arm,
              'environment': {'python': sys.version.split()[0], 'numpy': numpy.__version__},
              'tokenisation_stratum': TOKENISATION_STRATUM[args.arm],
              'gate': ('whether residual model-side likelihood information is explained by '
                       'longer-range but still additive sequence context'),
              'declaration': {'sha256': declaration_digest(),
                              'block_widths': {'X': len(GLOBAL_FEATURE_ORDER),
                                               'W': len(BOUNDED_FEATURE_ORDER)},
                              'bounded_radius_residues':
                                  BLOCK_DECLARATION['bounded_radius_residues'],
                              'additive_by_construction':
                                  BLOCK_DECLARATION['additive_by_construction']},
              'extraction_identity': manifest['identity'],
              'inference': 'none; this run refits retained label-free extraction artefacts',
              'code_sha256': {str(path.relative_to(ROOT)): flagship.digest(path) for path in (
                  Path(__file__), ROOT / 'src/capability/context/global_context.py',
                  ROOT / 'scripts/capability/interactions/fit_pairwise_epistasis.py',
                  ROOT / 'src/capability/interactions/pairwise_epistasis.py',
                  ROOT / 'src/capability/readouts/readout_analysis.py')},
              'inputs': {'plan_content_sha256': args.expect_plan_sha256,
                         'cohort_sha256': plan['cohort_sha256'],
                         'baseline_q_sha256': plan['provenance']['baseline_q']['sha256'],
                         'profiles_sha256': flagship.digest(args.profiles),
                         'flagship_fit_sha256': flagship.digest(args.flagship_fit)},
              'recipe': {'split_seeds': list(SPLIT_SEEDS), 'outer_splits': 5, 'inner_splits': 4,
                         'projection': {'dim': PROJECTION_DIM},
                         'bootstrap': {'draws': args.bootstrap, 'seed': BOOTSTRAP_SEED,
                                       'unit': 'held group'},
                         'control_sets': {k: list(v) for k, v in GATE_CONTROL_SETS.items()},
                         'target': 'epsilon = y_AB - y_A - y_B + y_WT, combined dG_ML, kcal/mol',
                         'weighting': ('group equal; site pairs equal within a group; cycles '
                                       'equal within a site pair'),
                         'independent_unit': 'the site pair, never the cycle count'},
              'supports': {}}
    for support in args.supports.split(','):
        base, filtered = support.replace('_no_indel', ''), support.endswith('_no_indel')
        groups = set(plan['supports']['all_groups' if base == 'all' else 'q_groups']['groups'])
        panel = flagship.build_panel(plan, cohort, profiles, arm_data, baseline, groups)
        attach_context_blocks(plan, groups, panel)
        accounting = None
        if filtered:
            panel, accounting = apply_indel_exclusion(panel, plan, exclusion, groups)
        # One declaration, imported rather than restated, as the pairwise-epistasis
        # record asked a later run of this gate to do. It renders the target as a
        # Python float rather than as the numpy scalar the first version of the
        # admitted fit script iterated: both render identically under numpy 1.26,
        # where the admitted digests were produced, but numpy 2.1 renders a scalar
        # as `np.float64(x)`, so the original expression hashes the same 8,192 rows
        # to a different value in a numpy-2 environment.
        identity = row_identity(panel['group'], panel['site_pair'], panel['epsilon'])
        expected = admitted['supports'][base]
        if not filtered and identity != expected['row_identity_sha256']:
            raise SystemExit(f'{support}: rows differ from the admitted pairwise fit')
        if filtered and identity == expected['row_identity_sha256']:
            raise SystemExit(f'{support}: the exclusion changed no row')
        entry = {'row_identity_sha256': identity,
                 'row_identity_rendering': ('group | site pair | repr of the target as a Python '
                                            'float; the admitted fit script renders the numpy '
                                            'scalar, which is the same string under numpy 1.26 '
                                            'and a different one under numpy 2.1'),
                 'matches_admitted_pairwise_fit': not filtered,
                 'indel_exclusion': None if not filtered else {
                     'declaration_sha256': flagship.digest(args.indel_exclusion),
                     'rule': exclusion['rule'], **accounting,
                     'reading': ('a declared sensitivity on rows that deliberately differ from '
                                 'the admitted fit, reported beside it and never in place of it')},
                 'groups': len(groups), 'cycles': int(len(panel['epsilon'])),
                 'site_pairs': len(set(panel['site_pair'])),
                 'kish_effective_site_pairs': kish_effective_site_pairs(panel),
                 'shared_design_max_absolute_difference_kcal2': 0.0,
                 'states': int(len(panel['states']['y'])),
                 'block_dimensions': {k: int(v.shape[1]) for k, v in panel['blocks'].items()},
                 'design_dimensions': {
                     name: int(sum(panel['blocks'][b].shape[1] for b in design_blocks(name)
                                   if b != 'G') + (5 if 'G' in design_blocks(name) else 0))
                     for name in GATE_DESIGNS},
                 'stratum_support': {
                     stratum: {'cycles': int((panel['separation'] == stratum).sum()),
                               'groups': len(set(panel['group'][panel['separation'] == stratum])),
                               'site_pairs': len(set(panel['site_pair'][
                                   panel['separation'] == stratum]))}
                     for stratum in ('1-2', '3-9', '10+')},
                 'seeds': {}}
        for seed in SPLIT_SEEDS:
            outcome = gate_compare(panel, seed=seed, device=args.device)
            evaluation = evaluate_gate(panel, outcome, draws=args.bootstrap, seed=BOOTSTRAP_SEED)
            fold_digest = fold_identity(outcome['folds'])
            if not filtered and fold_digest != expected['seeds'][str(seed)][
                    'fold_identity_sha256']:
                raise SystemExit(f'{support} seed {seed}: folds differ from the admitted fit')
            for mine, theirs in () if filtered else SHARED_DESIGNS:
                if theirs not in expected['seeds'][str(seed)]['designs']:
                    continue
                moved = abs(evaluation['designs'][mine]['group_equal_mse_kcal2']['point']
                            - expected['seeds'][str(seed)]['designs'][theirs][
                                'group_equal_mse_kcal2']['point'])
                if moved > SHARED_DESIGN_TOLERANCE:
                    raise SystemExit(f'{support} seed {seed}: shared design {mine} moved by '
                                     f'{moved} kcal2/mol2 against the admitted fit')
                entry['shared_design_max_absolute_difference_kcal2'] = max(
                    entry['shared_design_max_absolute_difference_kcal2'], moved)
            entry['seeds'][str(seed)] = {
                'fold_identity_sha256': fold_digest,
                'matches_admitted_pairwise_fit': not filtered,
                'folds': [{k: f[k] for k in ('fold', 'held_groups', 'alpha', 'dimensions')}
                          for f in outcome['folds']],
                'nuisance': outcome['nuisance'],
                'designs': evaluation['designs'], 'increments': evaluation['increments'],
                'first_order_reconstruction': evaluation['reconstruction']}
        record['supports'][support] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({'arm': args.arm, 'out': str(args.out),
                      'declaration_sha256': declaration_digest(),
                      'supports': {k: {'groups': v['groups'], 'cycles': v['cycles'],
                                       'site_pairs': v['site_pairs'],
                                       'kish_effective_site_pairs':
                                           {k2: round(v2, 1) for k2, v2
                                            in v['kish_effective_site_pairs'].items()},
                                       'shared_design_max_absolute_difference_kcal2':
                                           v['shared_design_max_absolute_difference_kcal2']}
                                   for k, v in record['supports'].items()}}, indent=1))


if __name__ == '__main__':
    main()
