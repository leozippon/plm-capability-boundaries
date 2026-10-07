"""Exact recurrence parity and label-blind linkage/exclusion invariants."""
from pathlib import Path
import random
import json

import numpy as np
import pytest

from src.capability.core.amino_acids import AA20
from src.capability.core.family_grouping import batch_align, encode, reference_alignment
from src.capability.extensions.phenotype_homology import (
    Alignment, ExactAligner, align_task, canonical, complex_components,
    export_components, full_run, load_mapping, relation, seqhash, sha, study_links, task_plan)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def aligner():
    return ExactAligner(ROOT / 'runtimes/phenotype-homology-tests')


def compare(aligner, a, b, old=False):
    result = aligner.align(a, b)
    reference = reference_alignment(a, b, gap_open=11, gap_extend=1)
    assert (result.score, result.columns, result.identical,
            result.coverage_a, result.coverage_b) == pytest.approx(reference, abs=1e-12)
    assert 0 <= result.identical <= result.paired_residues <= result.columns
    if old:
        codes, lengths = encode([a, b])
        stats = batch_align(codes, lengths, np.array([[0, 1]]), gap_open=11, gap_extend=1)
        assert (stats.score[0], stats.columns[0], stats.identical[0],
                stats.coverage_a[0], stats.coverage_b[0]) == pytest.approx(reference, abs=1e-12)


def test_random_short_and_frozen_batch(aligner):
    rng = random.Random(20261007)
    for _ in range(45):
        a = ''.join(rng.choices(AA20, k=rng.randrange(1, 65)))
        b = ''.join(rng.choices(AA20, k=rng.randrange(1, 65)))
        compare(aligner, a, b, old=True)
    for a, b in [('A', 'W'), ('AAAA', 'AA'), ('AA', 'AAAA'),
                 ('ACDEFGHIKLMNPQRSTVWY'*2, 'ACDEFGHIKLMNPQRSTVWY'*2),
                 ('WWWWWWAAAAWWWWWW', 'WWWWWWWWWWWW'),
                 ('WWWWWWWWWWWW', 'WWWWWWAAAAWWWWWW')]:
        compare(aligner, a, b, old=True)


def test_long_dynamic_coordinates_and_gaps(aligner):
    rng = random.Random(123)
    core = ''.join(rng.choices(AA20, k=170))
    pairs = [('A'*140 + core, 'W'*150 + core),
             (core, core[:80] + 'GGGGGGGG' + core[80:]),
             (core[:80] + 'GGGGGGGG' + core[80:], core),
             ('A'*260, 'W'*240), (core*2, core*2)]
    pairs += [(''.join(rng.choices(AA20, k=140)), ''.join(rng.choices(AA20, k=150))) for _ in range(3)]
    for a, b in pairs:
        compare(aligner, a, b)
    result = aligner.align(*pairs[0])
    assert result.span_a == result.span_b == len(core)
    # The frozen backend explicitly rejects this width: new path must not truncate.
    codes, lengths = encode([core, core])
    with pytest.raises(ValueError, match='under 128'):
        batch_align(codes, lengths, np.array([[0, 1]]), gap_open=11, gap_extend=1)


def test_rejection_and_runtime_guard(aligner, tmp_path):
    for a in ['', 'AX', 'acde']:
        with pytest.raises(ValueError, match='canonical'):
            aligner.align(a, 'ACDE')
    with pytest.raises(ValueError, match='ignored'):
        ExactAligner(tmp_path)


def test_thresholds_and_short_chance_guard(aligner):
    assert Alignment(1, 30, 9, 80, 80, 30, 30, 30).family()
    assert not Alignment(1, 31, 9, 80, 80, 30, 30, 30).family()
    assert not Alignment(1, 30, 9, 80, 79.99, 30, 30, 30).family()
    assert Alignment(1, 30, 15, 80, 10, 30, 30, 30).anchor()
    assert not Alignment(1, 30, 15, 79.99, 100, 30, 30, 30).anchor()
    # Both match most of their length with 50% identity, but only eight paired
    # residues: old floor alone is not an admitted new homology edge.
    r = align_task(aligner, ('family', 'a', 'b', 'AAAAAAAA', 'AAAASAAA'))
    assert r['minimum_suppressed'] and r['short_chain_pending'] and not r['admitted']
    exact = align_task(aligner, ('family', 'a', 'b', 'AAAAAAAA', 'AAAAAAAA'))
    assert exact['admitted'] and exact['relation'] == 'exact'
    assert relation('ACDE', 'WWACDEWW') == 'query_contained_in_subject'
    assert relation('WWACDEWW', 'ACDE') == 'subject_contained_in_query'


def synthetic():
    sequences = {'a': 'A'*35, 'b': 'C'*35, 'p': 'W'*35, 'x': 'D'*35, 'short': 'ACDE'}
    def row(c, names):
        return dict(complex_id=c, chain_states=[dict(wt_sha256=n, auth_chain=n) for n in names],
                    mutated_auth_chain=names[0], wt_sha256=names[0], variant_key=c)
    rows = [row('one', ['a', 'p']), row('two', ['b', 'p']), row('three', ['x']), row('four', ['short'])]
    return rows, sequences


def test_partner_sharing_anchor_cascade_short_quarantine_and_study_sensitivity():
    rows, chains = synthetic()
    families = {h: h for h in chains}
    hits = [dict(kind='anchor', query='a', admitted=True)]
    result = export_components(rows, chains, families, hits, {'two': ['1234567'], 'three': ['1234567']})
    assert result['preexclusion_membership']['one'] == result['preexclusion_membership']['two']
    assert result['candidate_holdout_membership'] == {'three': 'three'}
    assert result['counts']['direct_anchor_complexes'] == 1
    assert result['counts']['anchor_excluded_complexes'] == 2
    assert result['counts']['candidate_labels'] == 1
    assert result['study_closure']['primary_components'] == 3
    assert result['study_closure']['closure_components'] == 2
    assert any(c['cascade_only_complexes'] == ['two'] for c in result['components'])
    assert result['short_chain_pending_components'] == ['four']
    diagnostic = result['role_diagnostic']
    assert diagnostic['counts']['mutated_chain_families'] == 4
    assert diagnostic['counts']['mutated_role_complex_components'] == 4
    assert diagnostic['counts']['conservative_any_partner_complex_components'] == 3
    assert diagnostic['primary_component_mutated_family_counts']['one'] == 2
    assert not diagnostic['used_for_candidate_or_anchor_exclusion']
    assert diagnostic['extra_alignments'] == 0
    # Mutated OR partner family joins, not only the exact chain owner's label.
    families['x'] = families['a']
    membership = complex_components(rows, families)
    assert membership['one'] == membership['three']


def test_full_run_real_backend_stream_and_exports(aligner, tmp_path):
    a = 'ACDEFGHIKLMNPQRSTVWY'*2
    b = a[:20] + 'V' + a[21:]
    chains = {seqhash(s): s for s in [a, b, 'W'*35]}
    names = list(chains)
    rows = [dict(complex_id=c, variant_key=c, mutated_auth_chain='M', wt_sha256=h,
                 chain_states=[dict(wt_sha256=h, auth_chain='M')])
            for c, h in zip(['one', 'two', 'three'], names)]
    sources = [dict(source='ProteinGym', protein='anchor', sequence=a, sequence_sha256=seqhash(a))]
    result = full_run(aligner, rows, chains, sources, {}, tmp_path, 2)
    assert result['counts']['family_pairs'] == 3
    assert result['counts']['anchor_pairs'] == 3
    assert result['counts']['anchor_exact_containment_links'] == 1
    assert result['counts']['anchor_homolog_links'] == 1
    assert result['component_counts']['anchor_excluded_complexes'] == 2
    assert result['component_counts']['candidate_complexes'] == 1
    groups = json.loads((tmp_path/'component-map.json').read_text())
    assert groups['candidate_holdout_membership'] == {'three': 'three'}
    overlap = json.loads((tmp_path/'anchor-overlap.json').read_text())
    assert overlap['ProteinGym']['admitted_source_proteins'] == ['anchor']
    assert len(overlap['ProteinGym']['family_overlap_binding_chain_families']) == 1
    assert len((tmp_path/'pair-evidence.jsonl').read_text().splitlines()) >= 3


def qualified_mapping(tmp_path) -> tuple[Path, dict, Path]:
    """One admitted unknown construct; two incompatible rows and orphan inventory."""
    phase1 = tmp_path/'binding'
    mapping = phase1/'mapping'/'adjudicated'
    mapping.mkdir(parents=True)
    a, b = 'A'*35, 'C'*35
    def row(key, c, sequence, status):
        return dict(source_row_id=key, complex_id=c, variant_key=key, status=status,
                    construct_status='UNKNOWN_CONSTRUCT' if status == 'qualified' else 'KNOWN_INCOMPATIBLE',
                    mutated_auth_chain='M', wt_sha256=seqhash(sequence),
                    chain_states=[dict(sequence=sequence, wt_sha256=seqhash(sequence), auth_chain='M')])
    rows = [row('good', 'keep', a, 'qualified'),
            row('conflict', 'exclude', b, 'construct_incompatible'),
            row('also-conflict', 'keep', b, 'construct_incompatible')]
    (mapping/'row-mapping.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    # An orphan state inventory must not enter the chain roster.
    (mapping/'states.json').write_text(json.dumps({'states': [dict(sequence=b, sequence_sha256=seqhash(b))]}))
    (mapping/'exact-components.json').write_text(json.dumps({'membership': {'keep': 'keep'}}))
    admission = phase1/'admission.jsonl'
    admission.write_text(json.dumps(dict(source_row_id='good', complex_id='keep',
        original={'Reference': '1234567'}))+'\n')
    gate = dict(status='complete', qualified_row_status='qualified',
        qualified_source_row_ids=['good'], known_incompatible_source_row_ids=['conflict', 'also-conflict'],
        unknown_source_row_ids=['good'],
        admission=dict(path='../../admission.jsonl', sha256=sha(admission)))
    receipt = dict(adjudication_gate=gate, counts=dict(unique_complete_chain_sequences=1,
        operational_complexes=1, exact_identity_components=1, different_single_variants=1,
        validated_operational_labels=1))
    (mapping/'receipt.json').write_text(json.dumps(receipt))
    return mapping, receipt, admission


def test_qualified_roster_ignores_conflicting_rows_and_orphan_states(tmp_path):
    mapping, _, _ = qualified_mapping(tmp_path)
    rows, chains, manifest = load_mapping(mapping)
    assert [r['source_row_id'] for r in rows] == ['good']
    assert chains == {seqhash('A'*35): 'A'*35}
    assert manifest['counts']['validated_operational_labels'] == 1
    assert manifest['adjudication_gate']['unknown_source_row_ids'] == ['good']
    studies, receipt = study_links(mapping, rows)
    assert studies == {'keep': ['1234567']}
    assert receipt['parsed_labels'] == 1


@pytest.mark.parametrize('defect', ['missing_gate', 'incomplete_gate', 'conflict_admitted',
    'unknown_omitted', 'missing_row', 'duplicate_row', 'status_roster_mismatch',
    'stale_census', 'orphan_component', 'unknown_hidden', 'conflict_hidden'])
def test_adjudication_gate_fails_closed(tmp_path, defect):
    mapping, receipt, _ = qualified_mapping(tmp_path)
    gate = receipt['adjudication_gate']
    if defect == 'missing_gate':
        del receipt['adjudication_gate']
    elif defect == 'incomplete_gate':
        gate['status'] = 'pending'
    elif defect == 'conflict_admitted':
        gate['qualified_source_row_ids'].append('conflict')
    elif defect == 'unknown_omitted':
        del gate['unknown_source_row_ids']
    elif defect == 'missing_row':
        gate['qualified_source_row_ids'].append('absent')
    elif defect == 'duplicate_row':
        with (mapping/'row-mapping.jsonl').open('a') as handle:
            handle.write((mapping/'row-mapping.jsonl').read_text().splitlines()[0]+'\n')
    elif defect == 'status_roster_mismatch':
        gate['qualified_row_status'] = 'validated_operational_PDB_reference'
    elif defect == 'stale_census':
        receipt['counts']['validated_operational_labels'] = 4539
    elif defect == 'orphan_component':
        (mapping/'exact-components.json').write_text(json.dumps({'membership': {'keep': 'keep', 'exclude': 'exclude'}}))
    elif defect == 'unknown_hidden':
        gate['unknown_source_row_ids'] = []
    elif defect == 'conflict_hidden':
        gate['known_incompatible_source_row_ids'] = []
    (mapping/'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        load_mapping(mapping)


def test_adjudicated_study_admission_digest_gate(tmp_path):
    mapping, _, admission = qualified_mapping(tmp_path)
    rows, _, _ = load_mapping(mapping)
    admission.write_text(admission.read_text()+'\n')
    with pytest.raises(ValueError, match='admission changed'):
        study_links(mapping, rows)


def test_plan_and_hash():
    chains = {'a': 'A'*8, 'b': 'C'*40}
    sources = [dict(sequence='W'*50), dict(sequence='D'*100)]
    plan = task_plan(chains, sources)
    assert plan['family_pairs'] == 1 and plan['anchor_pairs'] == 4
    assert plan['total_cells'] == 8*40 + (8+40)*(50+100)
    assert plan['short_chain_sequences'] == 1
    assert len(seqhash('ACDE')) == 64
    canonical('ACDE')
