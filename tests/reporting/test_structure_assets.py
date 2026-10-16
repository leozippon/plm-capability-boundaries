"""The supporting structure illustrations, and the four things that have to hold for them to be honest.

These files are illustrations, not evidence, so the risk they carry is not a wrong number
but a picture that says more than the measurements do. Four properties keep that from
happening and each is checked here.

A panel annotated with a TM-score has to *show* the superposition that attains it. The
endpoint maximises over rigid-body superpositions at the identity correspondence, and the
single all-residue fit scores far lower when the two folds disagree, so a renderer that drew
one while quoting the other would make every low-scoring candidate look worse than its own
measurement. The renderer recovers the attaining transform and refuses to draw anything whose
recomputed score is not the retained one.

A view has to be a property of the structure rather than of the orientation the folding run
happened to emit, or no two renders of the same fold would be comparable and the projection
would not be reproducible.

No structure-derived quantity may rank a candidate. ``plddt_mean``, ``ptm`` and ``tm_score``
are refused by the frozen selector registry the renderer reuses, and the TM oracle is the one
declared exception because it is defined as the ceiling rather than offered as a method.

Finally, a reader has to be able to get from an image back to the candidates behind it, which
is what the manifest is for, and the images have to be reproducible for their digests to mean
anything at all.
"""
import json

import numpy as np
import pytest

from scripts.capability.reporting import render_structure_assets as assets
from src.capability.ladder import fold_geometry as fg


@pytest.fixture(scope='module')
def inputs():
    return assets.Inputs(assets.REPO)


@pytest.fixture(scope='module')
def sample(inputs):
    """A few retained candidates spanning the TM range, one per domain at a declared rank."""

    chosen = []
    for domain in assets.SEALED[:4]:
        pool = sorted((r for r in inputs.records if r['domain'] == domain),
                      key=lambda r: (r['tm_score'], r['id']))
        chosen += [pool[0], pool[len(pool) // 2], pool[-1]]
    return chosen


def test_drawn_superposition_attains_the_retained_tm_score(inputs, sample):
    for record in sample:
        placed = assets.superposed(inputs, record)
        assert placed['tm_score'] == record['tm_score']
        assert placed['tm_score'] == fg.tm_score(inputs.candidate(record)[0],
                                                 inputs.reference(record['domain']))['tm_score']


def test_the_drawn_superposition_is_a_rigid_motion(inputs, sample):
    """It may rotate and translate the candidate and may do nothing else to it."""

    for record in sample:
        original = inputs.candidate(record)[0]
        placed = assets.superposed(inputs, record)
        before = np.linalg.norm(original[:, None, :] - original[None, :, :], axis=-1)
        after = np.linalg.norm(placed['moved'][:, None, :] - placed['moved'][None, :, :], axis=-1)
        assert np.allclose(before, after, atol=1e-9)
        expected = np.linalg.norm(placed['moved'] - inputs.reference(record['domain']), axis=1)
        assert np.allclose(placed['deviation_angstrom'], expected, atol=1e-9)


def test_a_score_that_is_not_the_retained_one_is_refused(inputs, sample):
    tampered = dict(sample[0], tm_score=sample[0]['tm_score'] + 0.01)
    with pytest.raises(ValueError, match='would not be the endpoint'):
        assets.superposed(inputs, tampered)


def test_the_view_does_not_depend_on_the_orientation_of_the_coordinates(inputs):
    """Rotating the input must leave the drawn projection alone, or no render is comparable."""

    generator = np.random.default_rng(20261015)
    rotation, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] = -rotation[:, 0]
    for domain in assets.SEALED:
        coordinates = inputs.reference(domain)
        moved = coordinates @ rotation.T + np.array([13.0, -7.0, 2.5])
        here = assets.project(coordinates, assets.view_frame(coordinates))
        there = assets.project(moved, assets.view_frame(moved))
        assert np.allclose(here, there, atol=1e-6), domain
    frame = assets.view_frame(inputs.reference('1F0M'))[1]
    assert np.isclose(np.linalg.det(frame), 1.0)


@pytest.mark.parametrize('field', sorted(assets.e20.FORBIDDEN_SELECTOR_FIELDS))
def test_a_structure_derived_field_cannot_rank_candidates(inputs, field):
    rows = inputs.admissible(inputs.cell('1IFY', 'progen3-112m-bfloat16', 'N'))
    with pytest.raises(ValueError, match='structure-derived'):
        assets.selector_ranking(inputs, rows, field)


def test_the_oracle_is_the_declared_exception_and_is_labelled_as_one(inputs):
    rows = inputs.admissible(inputs.cell('1IFY', 'progen3-112m-bfloat16', 'N'))
    picked = assets.oracle_ranking(rows)
    assert len(picked) == assets.SELECTION_K
    assert [r['tm_score'] for r in picked] == sorted((r['tm_score'] for r in rows), reverse=True)[:assets.SELECTION_K]
    assert 'not a selector' in assets.ORACLE_STATEMENT


def test_a_coordinate_file_that_does_not_match_its_declared_digest_is_refused(inputs):
    path = assets.REPO / assets.WILD_TYPE
    with pytest.raises(ValueError, match='digest'):
        assets.require_digest(path, '0' * 64, 'tampered reference')


def test_two_renders_of_one_asset_are_byte_identical(inputs, tmp_path):
    """Matplotlib stamps dates and salts SVG ids; without that pinned, no digest means anything."""

    digests = []
    for index in ('first', 'second'):
        drawn = assets.render_tm_effect_scale(inputs)
        files = assets.save(drawn['figure'], 'tm-effect-scale', tmp_path / index)
        digests.append({kind: entry['sha256'] for kind, entry in files.items()})
    assert digests[0] == digests[1]


def test_the_manifest_leads_from_every_image_back_to_its_candidates(tmp_path):
    manifest = assets.render(tmp_path, assets.REPO)
    assert set(manifest['assets']) == {name for name, _ in assets.ASSETS}
    forbidden = set(assets.e20.FORBIDDEN_SELECTOR_FIELDS)
    assert forbidden == set(manifest['forbidden_selector_fields'])
    for name, asset in manifest['assets'].items():
        assert asset['selection_rule'].strip(), name
        assert asset['colour'].strip(), name
        for kind, entry in asset['files'].items():
            path = tmp_path / entry['file']
            assert assets.digest(path) == entry['sha256'], f'{name}.{kind}'
        assert asset['panels'], name
        for record in asset['panels']:
            candidates = record.get('candidates', [record])
            for candidate in candidates:
                assert candidate.get('id') or candidate.get('domain'), name
                assert candidate.get('structure_pdb_sha256') or candidate.get('pdb_sha256'), name
    # Exactly one ranking per selection cell may be the ceiling, and it has to be named one.
    scored = manifest['assets']['selection-failure']['panels']
    by_cell = {(p['domain'], p['model']): [] for p in scored}
    for record in scored:
        by_cell[(record['domain'], record['model'])].append(record)
    for cell, records in by_cell.items():
        ceilings = [r for r in records if not r['scorer_is_admissible']]
        assert len(ceilings) == 1, cell
        assert 'oracle' in ceilings[0]['scorer'].lower()
        assert ceilings[0]['captured_fraction_of_oracle_headroom'] is None
        admissible = sorted(r['scorer'] for r in records if r['scorer_is_admissible'])
        assert admissible == sorted(label for _, label in assets.SCORERS), cell
    # The statement a reader needs is carried by the artifact, not only by its README.
    for phrase in ('ESMFold2', 'no structure coordinates'):
        assert phrase in json.dumps(manifest)
