"""Source-grounded first-stage qualification and failure-path tests."""
import csv
import math
import subprocess
import sys
from pathlib import Path

import pytest

from src.capability.extensions.binding_cohort import R_KCAL, admit, parse_mutations, point, temperature

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/skempi2/skempi_v2.csv'


@pytest.fixture(scope='module')
def records():
    return admit(SOURCE)


def test_known_source_rows_and_support(records):
    first = records[0]
    assert first['source_row_id'] == 'skempi2:000001'
    assert first['complex_id'] == '1CSE_E_I'
    assert first['mutations'] == [dict(wt_aa='L', chain='I', author_position=45, insertion_code='', mt_aa='G')]
    assert first['original']['Mutation(s)_cleaned'] == 'LI38G'
    assert first['original']['Reference'] == '9048543'
    assert first['original']['Method'] == 'IASP'
    target = math.log(5.26e-11 / 1.12e-12)
    assert first['ln_kd_ratio'] == pytest.approx(target)
    assert first['ddg_kcal_mol'] == pytest.approx(R_KCAL * 294 * target)
    assert first['ln_kd_ratio'] > 0
    accepted = [r for r in records if r['admitted']]
    assert len(records) == 7085
    assert sum(r['single_substitution'] for r in records) == 5112
    assert len(accepted) == 4829
    assert sum(r['ddg_kcal_mol'] is not None for r in accepted) == 3336
    assert len({r['complex_id'] for r in accepted}) == 316
    assert all(r['likelihood_coverage'] == 'UNKNOWN_before_sequence_join' for r in accepted)
    assert all(r['mapping_status'] == 'labels_ready_sequence_mapping_pending' for r in accepted)


@pytest.mark.parametrize('raw,parsed,status', [('>1E-04','1E-04','censored_bound'),
                                             ('<1E-11','1E-11','censored_bound'),
                                             ('~2.2E-14','2.2E-14','approximate_not_point'),
                                             ('n.b','','missing_nonbinding_or_unfolded'),
                                             ('unf','','missing_nonbinding_or_unfolded'),
                                             ('0','0','nonpositive_or_nonfinite')])
def test_original_affinity_rules(raw, parsed, status):
    assert point(raw, parsed) == (None, status)


@pytest.mark.parametrize('raw,parsed', [('1e-9','2e-9'), ('broken',''), ('1e-9',''), ('n.b','1e-9')])
def test_parse_error_is_not_exclusion(raw, parsed):
    with pytest.raises(ValueError):
        point(raw, parsed)


def test_units_and_temperature():
    with pytest.raises(ValueError, match='units'):
        point('1','1', 'nM')
    assert temperature('298(assumed)') == (None, 'source_assumed_not_measurement')
    assert temperature('') == (None, 'missing')
    assert temperature('200')[0] is None
    assert temperature('294')[0] == 294
    with pytest.raises(ValueError):
        temperature('25 Celsius')


def test_mutation_integrity():
    assert parse_mutations('GH100cA', '1ABC_H_L')[0]['insertion_code'] == 'c'
    assert parse_mutations('LH-1A', '1ABC_H_L')[0]['author_position'] == -1
    for mutation, complex_id in [('LH1*','1ABC_H_L'), ('LA1G','1ABC_H_L'), ('LH1G','1ABC_H_H'), ('L1G','1ABC_H_L')]:
        with pytest.raises(ValueError):
            parse_mutations(mutation, complex_id)


def test_actual_duplicates_conditions_and_source_integrity(records):
    accepted = [r for r in records if r['admitted']]
    duplicates = [r for r in accepted if r['duplicate_group_size'] > 1]
    assert len(duplicates) == 12
    assert len({r['source_row_id'] for r in duplicates}) == 12
    assert len({r['duplicate_group'] for r in duplicates}) == 6
    by_mutation = {}
    for row in accepted:
        by_mutation.setdefault(row['mutation_group'], set()).add(row['condition_group'])
    assert any(len(conditions) > 1 for conditions in by_mutation.values())
    bad = records[6149]
    assert bad['original']['Mutation(s)_PDB'] == 'RA310E,KA312E,RB310E,KA312E'
    assert not bad['admitted']
    assert 'source_integrity_repeated_mutation_site' in bad['exclusion_reasons']
    assert all(r['ddg_kcal_mol'] is None for r in accepted if '(assumed)' in r['original']['Temperature'])
    assert all(not r['admitted'] for r in records if 'censored_bound' in r['affinity_mut_status'])


def test_cli_refuses_existing_output_before_write(tmp_path):
    sentinel = tmp_path / 'sentinel'
    sentinel.write_text('unchanged')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/capability/extensions/prepare_binding_cohort.py'),
                             '--out', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'refusing overwrite' in result.stderr
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == 'unchanged'


def test_malformed_source_fails_with_row_identity(tmp_path):
    with SOURCE.open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter=';')
        row = next(reader)
        fields = reader.fieldnames
        assert fields is not None
    row['Affinity_mut (M)'] = 'garbled'
    path = tmp_path / 'bad.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=';')
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match='source record 1: malformed original'):
        admit(path)
