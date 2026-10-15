"""Figure 6 plots retained R6 estimates and records what the artifacts refuse to issue."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capability.reporting import render_figures as renderer

PLOTTED = renderer.DATA / 'Fig6_guidance_and_selection-plotted.csv'
ARTIFACTS = {renderer.R6_CONFIRMATORY, renderer.R6_ROBUSTNESS}
R6_TABLES = ['guidance-contrasts-source', 'selection-headroom-source', 'progen3-dose-floor-source']


@pytest.fixture(scope='module')
def plotted():
    return pd.read_csv(PLOTTED)


@pytest.fixture
def retained(monkeypatch):
    """The render receipt's own digests, as the published renderer loads them."""
    manifest = json.loads((renderer.DATA / 'figure_validation.json').read_text())
    monkeypatch.setattr(renderer, 'EXPECTED_HASHES', dict(manifest['source_sha256']))
    for name in ['HASHES', 'EXTRACTION_HASHES']:
        monkeypatch.setattr(renderer, name, {})


def test_every_plotted_row_carries_its_own_artifact_digest(plotted):
    """A row is redrawable and attributable on its own, or it is not retained."""
    assert set(plotted.panel) == {'a', 'b', 'c', 'declared-context'}
    assert set(plotted.source_path) == ARTIFACTS
    for relative, digest in plotted[['source_path', 'source_sha256']].drop_duplicates().itertuples(index=False):
        assert renderer.digest(ROOT / relative) == digest
    assert plotted.source_pointer.notna().all()
    assert plotted.n_units.between(1, 8).all()
    assert plotted.interval_coverage_note.dropna().str.contains('0.89').all()


def test_guidance_panels_cover_both_endpoints_in_every_stratum(plotted):
    """Panels a and b are the same two contrasts on the two co-primary endpoints."""
    panels = plotted[plotted.panel.isin(['a', 'b'])]
    assert len(panels) == 2 * len(renderer.GUIDANCE_STRATA) * len(renderer.GUIDANCE_CONTRASTS)
    for letter, endpoint in [('a', renderer.PRIMARY_TM), ('b', renderer.NOVELTY_RATE)]:
        panel = panels[panels.panel == letter]
        assert set(panel.endpoint) == {endpoint}
        assert set(zip(panel.stratum, panel.contrast)) == {
            (stratum, contrast) for stratum in renderer.GUIDANCE_STRATA
            for contrast, _ in renderer.GUIDANCE_CONTRASTS}
    # The second co-primary endpoint is named for what it measures, never for the
    # frozen `feasibility` field it is carried under; no structural confidence enters it.
    novelty = panels[panels.endpoint == renderer.NOVELTY_RATE]
    assert set(novelty.endpoint_label) == {'Novelty-admissibility rate'}
    assert set(novelty.field_name_in_source) == {'coprimary_feasibility', renderer.NOVELTY_RATE}
    definitions = novelty.endpoint_definition.dropna()
    assert not definitions.empty and definitions.str.contains('misnomer').all()
    assert not novelty.endpoint_label.str.contains('feasib|confidence', case=False).any()


def test_unestimable_contrasts_are_recorded_rather_than_dropped(plotted):
    """A cell without an estimate keeps its row, its reason and no interval."""
    blank = plotted[plotted.status == 'unestimable']
    assert set(zip(blank.panel, blank.stratum, blank.contrast)) >= {
        ('a', 'pooled', 'C-Pr'), ('a', 'progen2-medium', 'C-Pr')}
    assert blank[['ci_low', 'ci_high']].isna().all().all()
    assert (blank.n_units == 7).all()
    assert blank.unestimable_reason.str.contains('8-domain floor').all()
    estimated = plotted[plotted.status == 'estimated']
    bounds = estimated[['ci_low', 'estimate', 'ci_high']].to_numpy(dtype=float)
    assert np.isfinite(bounds).all()
    assert ((bounds[:, 0] <= bounds[:, 1]) & (bounds[:, 1] <= bounds[:, 2])).all()


def test_reference_margin_and_oracle_ceiling_are_recorded_as_issuing_no_verdict(plotted):
    """Both reference lines carry what they are not: a verdict and a usable selector."""
    guidance = plotted[plotted.panel.isin(['a', 'b'])]
    assert set(guidance.margin_reference_tm) == {0.05}
    robustness = guidance[guidance.campaign == 'progen3_robustness']
    assert set(robustness.margin_verdict) == {'none issued'}
    assert robustness.margin_note.str.contains('no bound is tested against it').all()
    assert set(guidance[guidance.campaign == 'e20_confirmatory'].margin_verdict) == {'unestimable'}
    selection = plotted[plotted.panel == 'c']
    assert len(selection) == len(renderer.POOLS) * len(renderer.SELECTORS)
    assert not selection.oracle_is_usable_selector.any()
    assert set(selection.oracle_ceiling_captured_fraction) == {1.0}
    assert selection.oracle_ceiling_basis.str.startswith('definitional').all()
    assert selection.oracle_role.str.contains('not a selector').all()
    # Native likelihood loses headroom in every pool basis; nothing clips that sign.
    native = selection[selection.selector == 'native_loglik']
    assert len(native) == len(renderer.POOLS) and (native.estimate < 0).all()


def test_declared_dose_floor_travels_with_the_rows_it_qualifies(plotted):
    """Fourteen declared below-floor dose cells, and the ones each contrast rests on."""
    declared = plotted[plotted.panel == 'declared-context']
    assert len(declared) == 14 and set(declared.declared_total_cells) == {14}
    assert (declared.floor_margin_achieved < declared.floor_margin_required).all()
    assert set(declared.noise_floor_nats) == {0.036}
    robustness = plotted[plotted.panel.isin(['a', 'b']) & plotted.campaign.eq('progen3_robustness')]
    assert set(robustness.dose_floor_declared_total) == {14}
    coupling = renderer.rec(robustness, panel='a', stratum='progen3-3b-bfloat16', contrast='C-Pr')
    assert coupling.dose_floor_cells == 'progen3-3b-bfloat16 C:1IFY'
    assert renderer.rec(robustness, panel='a', stratum='progen3-3b-bfloat16',
                        contrast='Pr-N').dose_floor_cells == 'none for this contrast'


def test_retained_rows_are_refused_when_a_status_and_its_interval_disagree():
    """Neither an invented interval nor a missing reason passes as a retained estimate."""
    row = dict(key='x', estimate=.5, ci_low=.1, ci_high=.9, status='estimated',
               unestimable_reason=None, n_units=8, unit=renderer.DOMAIN_UNIT,
               source_path='a.json', source_sha256='0' * 64, source_pointer='a.b')
    table = pd.DataFrame([row, dict(row, key='y', estimate=.2, ci_low=None, ci_high=None,
                                    status='unestimable', unestimable_reason='too few units',
                                    n_units=7)])
    renderer.validate_retained_estimates(table, 2, ['key'])
    with pytest.raises(ValueError, match='carries an interval or no reason'):
        renderer.validate_retained_estimates(table.assign(ci_low=[.1, .0]), 2, ['key'])
    with pytest.raises(ValueError, match='carries an interval or no reason'):
        renderer.validate_retained_estimates(table.assign(unestimable_reason=None), 2, ['key'])
    with pytest.raises(ValueError, match='Invalid retained R6 interval'):
        renderer.validate_retained_estimates(table.assign(estimate=[1.5, .2]), 2, ['key'])
    with pytest.raises(ValueError, match='independence-unit counts'):
        renderer.validate_retained_estimates(table.assign(n_units=[8, 0]), 2, ['key'])
    with pytest.raises(ValueError, match='Duplicate retained R6 rows'):
        renderer.validate_retained_estimates(table.assign(key='x'), 2, ['key'])


def test_retained_tables_bind_their_artifacts_and_refuse_drift(retained, monkeypatch):
    """Reading a retained table binds the artifact behind it, and drift is refused."""
    assert len(renderer.guidance_contrasts()) == 28
    assert len(renderer.selection_headroom()) == 15
    assert len(renderer.dose_floor_context()) == 14
    assert set(renderer.EXTRACTION_HASHES) == ARTIFACTS
    assert {Path(name).name for name in renderer.HASHES} == {f'{name}.csv' for name in R6_TABLES}
    monkeypatch.setitem(renderer.EXPECTED_HASHES,
                        'scripts/capability/reporting/figure_data/main/guidance-contrasts-source.csv',
                        '0' * 64)
    with pytest.raises(ValueError, match='Source-table digest mismatch'):
        renderer.guidance_contrasts()
    monkeypatch.delitem(renderer.EXPECTED_HASHES,
                        'scripts/capability/reporting/figure_data/main/selection-headroom-source.csv')
    with pytest.raises(ValueError, match='No retained source-table digest'):
        renderer.selection_headroom()
