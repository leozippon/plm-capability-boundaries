import numpy as np
import pytest
import torch
from src.capability.readouts.readout_extraction import pool_hidden


def test_pool_excludes_prefix_eos_padding_and_preserves_time_first():
    hidden = torch.arange(2*7*3, dtype=torch.float32).reshape(2,7,3)
    spans = [(2,5),(1,3)]
    result = pool_hidden(hidden, spans)
    np.testing.assert_array_equal(result[0,0], hidden[0,2:5].mean(0))
    np.testing.assert_array_equal(result[1,1], hidden[1,2])
    masked = hidden.clone()
    masked[0,:2] = 1e5; masked[0,5:] = -1e5
    masked[1,:1] = 1e5; masked[1,3:] = -1e5
    np.testing.assert_array_equal(result, pool_hidden(masked, spans))
    np.testing.assert_array_equal(result, pool_hidden(hidden.transpose(0,1), spans, time_first=True))


def test_invalid_and_nonfinite_states_fail():
    with pytest.raises(ValueError, match='span'):
        pool_hidden(torch.zeros(1,4,3), [(3,5)])
    with pytest.raises(ValueError, match='Nonfinite'):
        pool_hidden(torch.full((1,4,3), float('nan')), [(1,3)])


def test_mutation_drift_uses_each_mutation_and_zero_reference_fails():
    from src.capability.readouts.readout_extraction import mutation_relative_drift
    reference = np.array([[0., 0.], [1., 0.], [100., 0.]])
    changed = reference.copy(); changed[1, 0] = 1.1
    assert mutation_relative_drift(changed, reference) == pytest.approx(.1)
    assert mutation_relative_drift(np.zeros((3, 2)), np.zeros((3, 2))) == 0
    with pytest.raises(ValueError, match='zero reference'):
        mutation_relative_drift(changed, np.zeros((3, 2)))


def test_progen3_batch_rows_keep_native_sequence_embedding_and_positions(monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    from src.capability.models import progen3
    spec = importlib.util.spec_from_file_location('readout_test_stage46', Path(__file__).resolve().parents[2]/'scripts/capability/stages/context_homologue.py')
    stage = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage)
    monkeypatch.setattr(stage.ch, '_progen3_special_ids', lambda tokenizer: {'<pad>': 0})
    received = []
    def forward(pg, batch):
        received.append(batch)
        return SimpleNamespace(logits=batch['input_ids'].float().unsqueeze(-1))
    monkeypatch.setattr(progen3, 'forward', forward)
    arm = SimpleNamespace(name='progen3-3b', device='cpu', tokenizer=None, serving_provenance={'progen3': object()})
    stage._forward_progen3(arm, [[1, 2, 3], [4, 5]])
    batch = received[0]
    assert torch.equal(batch['sequence_ids'], torch.zeros((2,3), dtype=torch.long))
    assert torch.equal(batch['position_ids'], torch.tensor([[0,1,2],[0,1,0]]))
    assert torch.equal(batch['input_ids'], torch.tensor([[1,2,3],[4,5,0]]))


def test_incomplete_extraction_does_not_publish_queue_completion(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('readout_test_extract', Path(__file__).resolve().parents[2]/'scripts/capability/readouts/extract_frozen_readout.py')
    script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
    manifest, progress = tmp_path/'manifest_arm.json', tmp_path/'progress_arm.json'
    payload = {'identity': {'arm': 'test'}, 'assays': [{'assay': 'first'}]}
    script.publish_progress(manifest, progress, payload, expected_assays=2)
    assert not manifest.exists()  # An interrupted job cannot satisfy the queue's expected file.
    assert json.loads(progress.read_text())['status'] == 'running'
    payload['assays'].append({'assay': 'second'})
    script.publish_progress(manifest, progress, payload, expected_assays=2)
    assert json.loads(manifest.read_text())['status'] == 'complete'
    assert json.loads(progress.read_text()) == json.loads(manifest.read_text())


def test_eligibility_stops_at_first_overbudget_and_preserves_maximum(monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('readout_test_eligibility', Path(__file__).resolve().parents[2]/'scripts/capability/readouts/extract_frozen_readout.py')
    script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
    seen = []
    monkeypatch.setattr(script.ch, 'row_prefix_ids', lambda arm: [101])
    def item_ids(arm, sequence, modality):
        seen.append(sequence)
        return list(range(len(sequence)))
    monkeypatch.setattr(script.ch, 'item_ids', item_ids)
    assert script.eligible_max_tokens(None, ['LONG', 'must not tokenize'], 4) is None
    assert seen == ['LONG']
    seen.clear()
    assert script.eligible_max_tokens(None, ['WT', 'LONG', 'must not tokenize'], 4) is None
    assert seen == ['WT', 'LONG']
    assert script.eligible_max_tokens(None, ['WT', 'MUT', 'W'], 4) == 4


def test_protgpt2_formatting_tokens_are_excluded_but_mixed_tokens_retained():
    from types import SimpleNamespace
    from src.capability.readouts.readout_extraction import representation_positions
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return {1: 'A'*60, 2: '\n', 3: 'C', 4: '\nC'}[ids[0]]
    arm = SimpleNamespace(name='protgpt2', tokenizer=Tokenizer())
    sequence = 'A'*60+'C'
    positions = representation_positions(arm, [sequence], [([0,1,2,3], (1,4), (1,4))])
    assert positions == [[1,3]]
    hidden = torch.tensor([[[0.],[2.],[1000.],[6.]]])
    np.testing.assert_array_equal(pool_hidden(hidden, [(1,4)], positions=positions), [[[4.],[6.]]])
    assert representation_positions(arm, [sequence], [([0,1,4], (1,3), (1,3))]) == [[1,2]]
    with pytest.raises(ValueError, match='reconstruct'):
        representation_positions(arm, ['AC'], [([0,1,3], (1,3), (1,3))])


def test_native_block_dispatch_uses_declared_arm_path_and_rita_override():
    from types import SimpleNamespace
    from src.capability.readouts.readout_extraction import representation_blocks
    sentinel = object()
    arm = SimpleNamespace(name='progen2-small', spec=SimpleNamespace(architecture='progen'), blocks=lambda: sentinel)
    assert representation_blocks(arm) is sentinel
    rita = SimpleNamespace(name='rita-xl', spec=SimpleNamespace(architecture='rita'), model=SimpleNamespace(transformer=SimpleNamespace(layers=sentinel)))
    assert representation_blocks(rita) is sentinel


def test_text_prefix_and_no_prefix_keep_distinct_scoring_and_pooling(monkeypatch):
    from types import SimpleNamespace
    from src.capability.readouts import readout_extraction as extraction
    monkeypatch.setattr(extraction, 'text_ids', lambda arm, sequence: [8,9,10])
    prefixed = SimpleNamespace(serving_provenance={'text_aa_boundary':SimpleNamespace(conditioning_id=8)})
    unprefixed = SimpleNamespace(serving_provenance={'text_aa_boundary':SimpleNamespace(conditioning_id=None)})
    assert extraction.pack_sequence(prefixed, 'AAA') == ([8,9,10],(1,3),(1,3))
    assert extraction.pack_sequence(unprefixed, 'AAA') == ([8,9,10],(1,3),(0,3))


def test_text_forward_attends_eos_prefix_and_masks_padding_by_position():
    from types import SimpleNamespace
    from src.capability.readouts.readout_extraction import forward_readout_rows
    received = {}
    class Model:
        def __call__(self, **kwargs):
            received.update(kwargs)
            return SimpleNamespace(logits=torch.zeros(2,3,12,dtype=torch.float32))
    arm = SimpleNamespace(serving_provenance={'text_aa_boundary':object()},tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=0),device='cpu',model=Model())
    forward_readout_rows(arm, [[0,1,2],[0,3]], None)
    assert received['attention_mask'].tolist() == [[1,1,1],[1,1,0]]
    assert received['use_cache'] is False


def test_ec_conditioning_requires_exact_wt_full_ec_and_no_default():
    from hashlib import sha256
    from types import SimpleNamespace
    from src.capability.readouts.readout_extraction import validate_ec_conditioning, pack_conditioned_sequence, EC_SELECTION_RULE
    cohort = {'assays':[{'assay':'test','wildtype':'ACD'}]}
    payload = dict(schema_version='readout_ec_conditioning_v1',cohort_sha256='cohort',selection_rule=EC_SELECTION_RULE,
                   annotation_source={'swissprot_xml_sha256':'a'*64,'ec_fasta_sha256':'b'*64},assays={})
    assert validate_ec_conditioning(payload, cohort, 'cohort') == {}
    with pytest.raises(ValueError, match='assay-bound EC'):
        pack_conditioned_sequence(SimpleNamespace(serving_provenance=None), 'ACD')
    record = dict(wildtype_sha256=sha256(b'ACD').hexdigest(),accessions=['P12345'],ec_numbers=['1.1.1.1'],ec='1.1.1.1')
    payload['assays']['test'] = record
    assert validate_ec_conditioning(payload, cohort, 'cohort') == {'test':'1.1.1.1'}
    with pytest.raises(ValueError, match='cohort hash'):
        validate_ec_conditioning(payload, cohort, 'wrong')
    record['wildtype_sha256'] = sha256(b'ACE').hexdigest()
    with pytest.raises(ValueError, match='wild-type hash'):
        validate_ec_conditioning(payload, cohort, 'cohort')
    record['wildtype_sha256'] = sha256(b'ACD').hexdigest()
    record['ec_numbers'] = ['1.1.1.-']
    with pytest.raises(ValueError, match='full four-field'):
        validate_ec_conditioning(payload, cohort, 'cohort')
