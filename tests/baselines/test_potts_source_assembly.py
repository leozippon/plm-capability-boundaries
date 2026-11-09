"""Final-only explicit source assembly: identity and refusal regression tests."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capability.baselines import assemble_potts_sources as A


def fixture(tmp_path, monkeypatch):
    from src.capability.stability import pairwise_stability
    monkeypatch.setattr(pairwise_stability, 'read_plmc', lambda p: None)
    original_path = tmp_path/'old.json'
    old_params = tmp_path/'old.params'
    old_params.write_bytes(b'old failed parameters')
    original: dict = dict(wildtype_id='q00059', status='final_not_converged_after_caps',
                    selected_lambda_e=80, lambda_h=.01, sweep=[dict(converged=True)]*5,
                    final=dict(status_text='ROUNDING_ERROR'), execution_manifest_sha256='old-execution',
                    params=dict(path=str(old_params), sha256=A.digest(old_params)), failure='old failure')
    original_path.write_text(json.dumps(original))
    recovery = tmp_path/'recovery'
    recovery.mkdir()
    for name in ('full.fa', 'full.weights', 'final.params', 'final.double'):
        (recovery/name).write_bytes(name.encode())
    build = tmp_path/'build'
    build.mkdir()
    (build/'plmc-recovery').write_bytes(b'independent binary')
    producer = dict(testing=False, binary_sha256=A.digest(build/'plmc-recovery'), patched_sources={})
    prep: dict = dict(original_files={str(original_path): A.digest(original_path),
                                str(old_params): A.digest(old_params)},
                inputs={n:A.digest(recovery/n) for n in ('full.fa','full.weights')},
                reused_cv=original['sweep'], selected_lambda_e=80, lambda_h=.01,
                total_outer_attempt_cap=5000, producer=producer, build_dir=str(build))
    receipt = dict(scientific_success=True, original_bytes_unchanged=True, process_returncode=0,
                   true_final_event=dict(accepted='1', unquantized='1', return_code='0',
                                         ratio='0.000963769', cumulative='2201'),
                   history_resets=0, producer=producer, status='converged_independent_producer',
                   elapsed_seconds=12., artifacts={n:A.digest(recovery/n) for n in ('final.params','final.double')})
    (recovery/'preparation.json').write_text(json.dumps(prep))
    (recovery/'receipt.json').write_text(json.dumps(receipt))
    B = SimpleNamespace(STATUS_FITTED='fitted', read_fasta=lambda p:[('focus','AC')],
                        potts_scorer=lambda p,w:SimpleNamespace(delta=lambda c:1.25),
                        mutant_codes=lambda w,s:s, shuffle_sign_check=lambda s:dict(passed=True))
    return original, recovery, B, prep


def test_final_only_replacement_preserves_original_bytes_and_execution_identity(tmp_path, monkeypatch):
    original, recovery, B, prep = fixture(tmp_path, monkeypatch)
    before = copy.deepcopy(original)
    derived = A.recover_final(original, recovery, [dict(sample_id='sample', substitutions=[])], B)
    assert original == before
    A.check_files(prep['original_files'])
    assert derived['status'] == 'fitted'
    assert derived['sweep'] == before['sweep']
    assert derived['sample_scores'] == {'sample':1.25}
    assert derived['params']['path'] == str(recovery/'final.params')
    assert derived['final']['iterations'] == 2201
    assert 'execution_manifest_sha256' not in derived
    assert 'failure' not in derived
    assert derived['plmc'] != 'old-execution'


@pytest.mark.parametrize('change', ['failed', 'gradient', 'cap', 'params', 'old', 'selection', 'test-producer'])
def test_recovery_refuses_false_success_and_source_drift(tmp_path, monkeypatch, change):
    original, recovery, B, _ = fixture(tmp_path, monkeypatch)
    receipt = A.load(recovery/'receipt.json')
    if change == 'failed': receipt['scientific_success'] = False
    elif change == 'gradient': receipt['true_final_event']['ratio'] = 'nan'
    elif change == 'cap': receipt['true_final_event']['cumulative'] = '5001'
    elif change == 'params': (recovery/'final.params').write_bytes(b'altered')
    elif change == 'old': Path(original['params']['path']).write_bytes(b'altered')
    elif change == 'selection': original['sweep'][0]['converged'] = False
    elif change == 'test-producer': receipt['producer']['testing'] = True
    (recovery/'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        A.recover_final(original, recovery, [], B)


def test_independent_final_uses_real_frozen_parameter_reader_and_scorer(tmp_path, monkeypatch):
    import struct
    import numpy as np
    from src.capability.baselines import msa_baselines as B
    original, recovery, _, preparation = fixture(tmp_path, monkeypatch)
    monkeypatch.undo()
    alphabet = ''.join(B.AA20).encode()
    fields = np.eye(20, dtype='<f4')
    raw = (struct.pack('<5i5f', 20, 20, 1, 0, 0, 0., 0., 0., 0., 0.) + alphabet
           + np.ones(1, dtype='<f4').tobytes() + alphabet
           + np.arange(1, 21, dtype='<i4').tobytes()
           + np.zeros(400, dtype='<f4').tobytes() + fields.tobytes()
           + np.zeros(2 * 190 * 400, dtype='<f4').tobytes())
    (recovery/'final.params').write_bytes(raw)
    (recovery/'full.fa').write_text('>focus\n' + alphabet.decode() + '\n')
    preparation['inputs']['full.fa'] = A.digest(recovery/'full.fa')
    (recovery/'preparation.json').write_text(json.dumps(preparation))
    receipt = A.load(recovery/'receipt.json')
    receipt['artifacts']['final.params'] = A.digest(recovery/'final.params')
    (recovery/'receipt.json').write_text(json.dumps(receipt))
    result = A.recover_final(original, recovery,
                             [dict(sample_id='sample', substitutions=[('A', 1, 'W')])], B)
    assert np.isfinite(result['sample_scores']['sample'])
    assert result['sample_scores']['sample'] != 0


def test_historical_nonconvergence_count_is_derived_and_disagreement_fails():
    counts = dict(complete=True, backgrounds_fitted=174,
                  backgrounds_final_not_converged=0, backgrounds_cv_not_converged=0)
    document = dict(counts=counts)
    A.publish_historical_completion(document)
    assert counts['backgrounds_not_converged'] == 0
    counts.update(backgrounds_final_not_converged=1, backgrounds_not_converged=0)
    with pytest.raises(ValueError, match='disagree'):
        A.publish_historical_completion(dict(counts=counts))


@pytest.mark.parametrize('change', [None, 'missing', 'duplicate-other-role', 'unapproved-recovery'])
def test_fixed_assignment_never_uses_duplicate_to_fill_missing_source(change):
    names = [f'q{i:05d}' for i in range(174)]
    partition = dict(out='/frozen', trees={'primary':'potts-cluster'},
                     source_role={n:'primary' for n in names})
    mapping = dict(wildtypes={n:dict(role='primary', checkpoint=f'/frozen/potts-cluster/backgrounds/{n}.json')
                             for n in names})
    if change == 'missing': mapping['wildtypes'].pop(names[0])
    elif change == 'duplicate-other-role':
        mapping['wildtypes'][names[0]]['checkpoint'] = f'/frozen/potts-shard-a/backgrounds/{names[0]}.json'
    elif change == 'unapproved-recovery': mapping['wildtypes'][names[0]]['recovery'] = '/recovery'
    if change:
        with pytest.raises(ValueError):
            A.require_fixed_sources(mapping, partition, names)
    else:
        A.require_fixed_sources(mapping, partition, names)
