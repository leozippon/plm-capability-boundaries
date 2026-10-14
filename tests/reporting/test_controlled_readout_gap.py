"""The controlled readout contrast binds all 33 x 3 retained estimate leaves."""
import copy
import json
from pathlib import Path

import pytest

from scripts.capability.reporting import build_derived_numbers as builder
from scripts.capability.reporting import followup_quantities as followup
from src.capability.core.evidence_ledger import Artifacts, Ledger, LedgerError, resolve


@pytest.fixture
def gap_source(tmp_path):
    block = {'point': .0046, 'interval': [-.0069, .0166], 'resamples': 2000,
             'unit': 'wild-type family at 50% identity', 'n_assays': 201,
             'n_units': 163, 'variants': 25728}
    payload = {'gap_arms': {f'arm-{i}': {str(seed): {'seed': str(seed),
                'baseline_identity': {'control_set': 'C_P', 'cohort_sha256': 'c' * 64},
                'd': copy.deepcopy(block)}
                for seed in followup.SPLIT_SEEDS} for i in range(33)}}
    path = tmp_path / followup.CONTROLLED_GAP
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    return tmp_path, path, payload


def check_rows(root):
    ledger = Ledger(Artifacts(root))
    followup._controlled_readout_gap(ledger, followup._emitter(ledger))
    builder.assign_representation_conditions(ledger)
    assert all(q.conditioning == 'readout_selection' for q in ledger.quantities)
    assert len(ledger.quantities) == len({q.id for q in ledger.quantities}) == 99
    # Fold seeds and checkpoints reuse the same biological cohort, not 99 supports.
    assert len({q.support_id for q in ledger.quantities}) == 1
    for quantity in ledger.quantities:
        payload = ledger.artifacts.json(quantity.source_path)
        assert resolve(payload, quantity.source_pointer) == quantity.value
        block = resolve(payload, quantity.source_pointer[:-1])
        assert isinstance(block, dict)
        assert quantity.interval == tuple(block['interval'])
        assert quantity.resampling_draws == block['resamples'] == 2000
        assert quantity.resampling_unit == block['unit']
        assert quantity.seed_set == (int(quantity.source_pointer[2]),)
        assert quantity.kind == 'estimate' and quantity.unit == 'dimensionless Spearman'
        assert '201 assays' in ledger.supports[quantity.support_id].description
    return ledger


def test_exact_leaf_binding(gap_source):
    check_rows(gap_source[0])


@pytest.mark.parametrize('failure', ['roster', 'seed', 'control'])
def test_refuse_inexact_contrast(gap_source, failure):
    root, path, payload = gap_source
    folds = payload['gap_arms']['arm-0']
    if failure == 'roster':
        del folds[str(followup.SPLIT_SEEDS[0])]
    elif failure == 'seed':
        folds[str(followup.SPLIT_SEEDS[0])]['seed'] = '1'
    else:
        folds[str(followup.SPLIT_SEEDS[0])]['baseline_identity']['control_set'] = 'C'
    path.write_text(json.dumps(payload))
    with pytest.raises(LedgerError):
        check_rows(root)


def test_live_retained_99_rows():
    root = Path(__file__).resolve().parents[2]
    if not (root / followup.CONTROLLED_GAP).is_file():
        pytest.skip('Private retained contrast is not distributed')
    check_rows(root)
