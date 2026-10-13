"""Deterministic checks for the counterfactual-response diagnostic helpers.

These exercise the arithmetic that must always hold: the larger-score top-26
selection and its exact SHA-256 tie rule, the direct Potts single-substitution
delta against the brute-force parameter score with nonzero couplings, the cached
profile count reconstruction against the existing fit's independent-site model,
the exact sequence/mutation mapping of a full substitution grid, digest-mismatch
refusal without output, and the guarantee that evaluation-domain phenotypes are
never read.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.capability.core.amino_acids import AA20
from src.capability.stability.pairwise_stability import PlmcParameters

SCRIPT = REPO / 'scripts/capability/stability/analyse_counterfactual_responses.py'
FIT = REPO / 'scripts/capability/interactions/fit_pairwise_sequence_baseline.py'


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load(SCRIPT, 'counterfactual_cli')
FIT_MODULE = _load(FIT, 'fit_pairwise_sequence_baseline_for_test')

DEV = 'DEV.pdb'
EVAL = 'EVAL.pdb'


def grid_coordinates(wildtype: str) -> tuple[np.ndarray, np.ndarray]:
    positions, mutants = [0], [-1]
    for site in range(1, len(wildtype) + 1):
        for code, residue in enumerate(AA20):
            if residue != wildtype[site - 1]:
                positions.append(site)
                mutants.append(code)
    return np.array(positions, dtype=np.int32), np.array(mutants, dtype=np.int8)


def synthetic_grid(wildtype: str) -> dict:
    positions, mutants = grid_coordinates(wildtype)
    sequences = [wildtype if code < 0 else wildtype[:site - 1] + AA20[code] + wildtype[site:]
                 for site, code in zip(positions, mutants)]
    state_sha = np.array([M.sha256_text(sequence) for sequence in sequences])
    score = np.linspace(-4.0, 3.0, len(positions))
    return {'position': positions, 'mutant_index': mutants, 'state_sequence_sha256': state_sha,
            'score': score, 'response': score - score[0], 'sequences': sequences}


def single_cohort_row(name: str, wildtype: str, ddg) -> dict:
    variants = []
    for site in range(1, len(wildtype) + 1):
        for residue in AA20:
            if residue == wildtype[site - 1]:
                continue
            variants.append({'sequence': wildtype[:site - 1] + residue + wildtype[site:],
                             'position': site, 'mutant': residue, 'ddg': ddg(site, residue)})
            if len(variants) == M.SINGLES_PER_DOMAIN:
                return {'name': name, 'wildtype': wildtype, 'group': 'g', 'variants': variants}
    raise AssertionError('the synthetic wild type cannot carry 256 single substitutions')


def test_cli_accepts_cpu_queue_dispatch_and_refuses_gpu_execution(capsys):
    with pytest.raises(SystemExit) as accepted:
        M.main(['--device', 'cpu', '--help'])
    assert accepted.value.code == 0
    assert '--device {cpu}' in capsys.readouterr().out
    with pytest.raises(SystemExit) as refused:
        M.main(['--device', 'cuda:0'])
    assert refused.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err


def test_top_selection_uses_larger_score_and_ascending_sha_ties():
    shas = np.array([f'{value:064x}' for value in (5, 4, 3, 2, 1, 0)])
    score = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 0.0])
    # The two 2.0 states tie and their shas are ...04 (index 1) and ...05 (index 0).
    assert M.top_selection(score, shas, k=3).tolist() == [1, 0, 4]
    # Orientation: the larger score wins, and the reverse selects the smallest.
    assert M.top_selection(-score, shas, k=1).tolist() == [5]
    with pytest.raises(ValueError, match='non-finite'):
        M.top_selection(np.array([0.0, np.nan, 1.0]), np.array(['a', 'b', 'c']), k=1)


def test_secondary_selection_reports_selected_minus_panel_mean():
    size = 30
    shas = np.array([f'{value:064x}' for value in range(size)])
    response = np.arange(size, dtype=np.float64)[::-1].copy()
    ddg = np.where(np.arange(size) < M.TOP_K, 2.0, -1.0)
    record = M.secondary_selection(response, ddg, shas)
    assert record['k'] == M.TOP_K
    assert record['panel_n'] == size
    assert record['selected_mean_ddg'] == pytest.approx(2.0)
    panel_mean = (M.TOP_K * 2.0 + (size - M.TOP_K) * -1.0) / size
    assert record['selected_minus_panel'] == pytest.approx(2.0 - panel_mean)
    assert record['tie_rule'].endswith('ascending sequence SHA-256')


def test_potts_direct_delta_equals_brute_score_with_nonzero_couplings():
    rng = np.random.default_rng(20261013)
    target = 'ACDE'
    # Gap-prefixed 21-symbol alphabet, so an AA20 code must be mapped through
    # params.alphabet before indexing; the brute sequence stays AA20.
    alphabet = '-' + AA20
    q = len(alphabet)
    fields = rng.normal(size=(len(target), q)).astype(np.float32)
    pairs = len(target) * (len(target) - 1) // 2
    couplings = rng.normal(size=(pairs, q, q)).astype(np.float32)
    params = PlmcParameters(alphabet, target, np.arange(1, len(target) + 1, dtype=np.int32),
                            fields, couplings)
    assert params.alphabet != AA20 and np.any(params.couplings != 0)
    position = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=np.int32)
    mutant_index = np.array([-1, 1, 2, 3, 0, 2, 3, 0, 1, 2, 0, 1, 2], dtype=np.int8)
    delta, binding = M.potts_single_response(params, target, position, mutant_index, sample=13)
    assert binding['brute_force_checked'] == len(position)
    assert binding['brute_force_max_absolute_difference'] < 1e-4
    for index in range(1, len(position)):
        sequence = list(target)
        sequence[position[index] - 1] = AA20[mutant_index[index]]
        expected = params.score(''.join(sequence)) - params.score(target)
        assert delta[index] == pytest.approx(expected, abs=1e-4)


def test_profile_counts_reproduce_the_existing_fit_independent_site_model():
    rng = np.random.default_rng(11)
    wildtype = 'ACDE'
    rows = rng.integers(0, len(AA20), size=(6, len(wildtype))).astype(np.uint8)
    weights = np.array([0.5, 1.0, 1.5, 0.25, 2.0, 0.75])
    log_profile = FIT_MODULE.independent_site(rows, weights)
    raw = np.zeros((len(wildtype), len(AA20)))
    for code in range(len(AA20)):
        raw[:, code] = (weights[:, None] * (rows == code)).sum(axis=0)
    column_weight = raw.sum(axis=1)
    frequencies = np.zeros_like(raw)
    positive = column_weight > 0
    frequencies[positive] = raw[positive] / column_weight[positive, None]
    position, mutant_index = grid_coordinates(wildtype)
    response, record = M.profile_single_response(frequencies, column_weight, wildtype,
                                                 position, mutant_index)
    assert record['pseudocount'] == M.PSEUDOCOUNT
    wild_codes = [AA20.index(residue) for residue in wildtype]
    for index in range(1, len(position)):
        site, code = int(position[index]) - 1, int(mutant_index[index])
        expected = log_profile[site, code] - log_profile[site, wild_codes[site]]
        assert response[index] == pytest.approx(expected, abs=1e-12)
    assert response[0] == 0.0


def test_grid_mapping_binds_exact_sequence_and_refuses_corruption():
    wildtype = 'ACDE'
    grid = synthetic_grid(wildtype)
    M.verify_grid_mapping(wildtype, grid['position'], grid['mutant_index'],
                          grid['state_sequence_sha256'], grid['score'], grid['response'])
    corrupt = grid['position'].copy()
    corrupt[1] = corrupt[0]
    with pytest.raises(ValueError, match='hash|wild-type|out-of-range'):
        M.verify_grid_mapping(wildtype, corrupt, grid['mutant_index'],
                              grid['state_sequence_sha256'], grid['score'], grid['response'])
    wrong_sha = grid['state_sequence_sha256'].copy()
    wrong_sha[1] = wrong_sha[0]
    with pytest.raises(ValueError, match='hash'):
        M.verify_grid_mapping(wildtype, grid['position'], grid['mutant_index'], wrong_sha,
                              grid['score'], grid['response'])
    shifted = grid['score'] + 1.0
    with pytest.raises(ValueError, match='score - score_WT'):
        M.verify_grid_mapping(wildtype, grid['position'], grid['mutant_index'],
                              grid['state_sequence_sha256'], shifted,
                              grid['response'] + 0.5)


def test_merged_grid_digest_mismatch_refuses_and_writes_nothing(tmp_path):
    assert M.REPORT_FILENAME == 'counterfactual_responses.json'
    wildtype = 'ACDE'
    grid = synthetic_grid(wildtype)
    path = tmp_path / 'arm' / 'DEV.pdb.npz'
    path.parent.mkdir(parents=True)
    metadata = json.dumps({'schema': M.MERGE_SCHEMA, 'arm': 'arm', 'background': DEV,
                           'full_states': len(grid['position']),
                           'wildtype_sha256': M.sha256_text(wildtype)})
    with path.open('wb') as stream:
        np.savez_compressed(stream, score=grid['score'], response=grid['response'],
                            position=grid['position'], mutant_index=grid['mutant_index'],
                            state_sequence_sha256=grid['state_sequence_sha256'],
                            metadata=metadata)
    entry = {'sha256': '0' * 64, 'full_states': len(grid['position']), 'group': 'g',
             'wildtype_sha256': M.sha256_text(wildtype)}
    with pytest.raises(ValueError, match='declared'):
        M.load_merged_grid(path, 'arm', DEV, entry, wildtype)
    assert not (tmp_path / 'out').exists()
    reference_path = tmp_path / 'out' / 'references' / 'DEV.pdb.npz'
    reference_path.parent.mkdir(parents=True)
    reference_path.write_bytes(b'existing')
    with pytest.raises(FileExistsError, match='overwrite'):
        M.write_reference(reference_path, {'domain': DEV})


def test_development_variants_never_reads_evaluation_phenotypes():
    wildtype = 'ACDEFGHIKLMNPQ'
    roles = {'development': [DEV], 'evaluation': [EVAL]}
    cohort = {'schema': M.COHORT_SCHEMA, 'backgrounds': [
        single_cohort_row(DEV, wildtype, lambda site, residue: 0.1 * site),
        single_cohort_row(EVAL, wildtype, lambda site, residue: float('nan'))]}
    out = M.development_variants(cohort, roles)
    assert set(out) == {DEV}
    assert len(out[DEV]['records']) == M.SINGLES_PER_DOMAIN
    shas = [record['sha256'] for record in out[DEV]['records']]
    assert shas == sorted(shas)
    for record in out[DEV]['records']:
        assert record['sequence'][record['position'] - 1] == record['mutant']
    # Poisoning the evaluation ddG to a sentinel and the development ddG to a new
    # value changes only the development result, which is the only one read.
    cohort['backgrounds'][1]['variants'][0]['ddg'] = -12345.0
    again = M.development_variants(cohort, roles)
    assert [record['ddg'] for record in again[DEV]['records']] == \
        [record['ddg'] for record in out[DEV]['records']]
    cohort['backgrounds'][0]['variants'][0]['ddg'] = 99.0
    changed = M.development_variants(cohort, roles)
    assert [record['ddg'] for record in changed[DEV]['records']] != \
        [record['ddg'] for record in out[DEV]['records']]


def test_development_and_evaluation_roles_must_be_disjoint():
    wildtype = 'ACDEFGHIKLMNPQ'
    cohort = {'schema': M.COHORT_SCHEMA, 'backgrounds': [
        single_cohort_row(DEV, wildtype, lambda site, residue: 0.1)]}
    with pytest.raises(ValueError, match='development and evaluation'):
        M.development_variants(cohort, {'development': [DEV], 'evaluation': [DEV]})


def test_real_input_preflight_builds_every_reference_before_reading_any_ddg(tmp_path, monkeypatch):
    """Real-input entrypoint preflight: all twelve references build, then labels.

    It binds the actual frozen protocol, digests, provenance, per-arm grids and
    pinned parameters and constructs all twelve full-grid references. Only after
    that does it reach the measured-label step, which is replaced by a sentinel so
    no ddG is ever read and no output is written. Skipped when the reviewed assets
    are absent.
    """
    protocol_file = REPO / 'data/r6_constraint_response_20261013/diagnostic_protocol.json'
    if not protocol_file.is_file():
        pytest.skip('the frozen 49ba protocol is absent')
    protocol = json.loads(protocol_file.read_text())
    domains = protocol['roles']['development'] + protocol['roles']['evaluation']
    models = protocol['models']
    declared = [
        'data/r6_constraint_response_20261013/diagnostic_protocol.json',
        'results/R6/r6_constraint_response_missing_20261013/merged/manifest.json',
        'results/R4/gate_stability_20260924/cohort/cohort.json',
        'data/r6_mechanisms_20261011/measured_stability_eligibility.json',
        'data/pairwise_assets/same12_provenance_20260924/pairwise_provenance.json',
        'results/R4/gate_stability_20260924/cohort/profile_features.npz',
        'archive/logs/R3/pairwise_homologs_20260924/baseline/baseline_q.json',
    ]
    derived = [f'archive/logs/R3/pairwise_homologs_20260924/baseline/work/{name}/selected.params'
               for name in domains]
    derived += [f'results/R6/r6_constraint_response_missing_20261013/merged/{arm}/{name}.npz'
                for arm in models for name in domains]
    missing = [path for path in declared + derived if not (REPO / path).is_file()]
    if missing:
        pytest.skip(f'the reviewed real inputs are absent: {missing[:3]}')

    built = []
    real_build = M.build_reference

    def capture(*args, **kwargs):
        reference = real_build(*args, **kwargs)
        built.append(reference['domain'])
        return reference

    class Sentinel(Exception):
        pass

    def forbidden(*args, **kwargs):
        raise Sentinel('development_variants was reached')

    monkeypatch.setattr(M, 'build_reference', capture)
    monkeypatch.setattr(M, 'development_variants', forbidden)
    out = tmp_path / 'counterfactual'
    with pytest.raises(Sentinel):
        M.run(REPO / 'data/r6_constraint_response_20261013/diagnostic_protocol.json', REPO,
              REPO / 'archive/logs/R3/pairwise_homologs_20260924/baseline', out)
    assert sorted(built) == sorted(domains), 'the measured step was reached before all references'
    assert len(built) == len(domains) == 12
    assert not out.exists(), 'a reference or report was written before the measured step'