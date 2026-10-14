"""Focused invariants for the pinned, unchanged-WT illustration renderer."""
import copy
import hashlib

import pytest

import scripts.capability.reporting.render_framework_structures as framework


def test_reconstruct_reference_and_refuse_invalid_substitutions():
    rows = [{"mutant": "A1G", "mutated_sequence": "GCD"},
            {"mutant": "C2S", "mutated_sequence": "ASD"}]
    assert framework.reconstruct_wt(rows) == "ACD"
    assert framework.substitution("ACD", "ASD", 2, "S") == "C2S"
    for sequence, position, mutant in [("ACD", 2, "S"), ("ACD", 2, "C"), ("ACD", 0, "G"), ("AXD", 2, "X")]:
        with pytest.raises(ValueError):
            framework.substitution("ACD", sequence, position, mutant)
    with pytest.raises(ValueError):
        framework.reconstruct_wt([{"mutant": "A1G", "mutated_sequence": "ACD"}])
    with pytest.raises(ValueError):
        framework.reconstruct_wt(rows + [{"mutant": "C2S", "mutated_sequence": "VSD"}])


def test_unique_exact_mapping_and_numbering_failure():
    assert framework.exact_subsequence("ACD", "GGACDSS") == {1: 3, 2: 4, 3: 5}
    for entity in ("ACDACD", "ACE", ""):
        with pytest.raises(ValueError, match="uniquely and exactly"):
            framework.exact_subsequence("ACD", entity)
    residues = [{"coordinate_id": 1, "residue": "G"}]
    with pytest.raises(ValueError, match="disagrees"):
        framework.map_af("ACD", residues, allow_absent_terminal=True)
    with pytest.raises(ValueError, match="uniquely and exactly"):
        framework.map_af("ACD", residues)


def test_median_selection_is_deterministic_and_not_extreme():
    rows = [{"id": name, "label": value} for name, value in [("a", -99), ("b", 2), ("c", 2), ("d", 99)]]
    selected, counts = framework.median_select(rows, "label")
    expected = min(("b", "c"), key=lambda name: hashlib.sha256((framework.SALT + name).encode()).hexdigest())
    assert selected["id"] == expected
    assert framework.median_select(list(reversed(rows)), "label")[0] == selected
    assert counts["median_selection_scalar"] == 2
    absolute, _ = framework.median_select([{"id": "a", "e": -8}, {"id": "b", "e": -2}, {"id": "c", "e": 5}], "e", absolute=True)
    assert absolute["id"] == "c"
    with pytest.raises(ValueError, match="No exact eligible"):
        framework.median_select([{"id": "nan", "label": float("nan")}], "label")


def test_cycle_requires_four_supported_exact_states_and_clean_identity():
    wt, a, b, ab = "ACD", "GCD", "ASD", "GSD"
    sequences = [wt, a, b, ab]
    background = {"name": "1UFM.pdb", "sequences": sequences,
                  "cycles": [{"sequences": sequences, "positions": [1, 2], "epsilon": 0.5}],
                  "measurements": {s: {"value": value, "rows": [{"value": value}]} for s, value in zip(sequences, [3, 2, 1, .5])}}
    clean = {"cohort_sha256": framework.PAIR_SHA, "accounting": {"backgrounds_touched": []}, "states": []}
    assert len(framework.cycle_rows(background, wt, {1: {}, 2: {}}, clean)) == 1
    broken = copy.deepcopy(background)
    del broken["measurements"][a]
    with pytest.raises(ValueError, match="Missing measured"):
        framework.cycle_rows(broken, wt, {1: {}, 2: {}}, clean)
    broken = copy.deepcopy(background)
    broken["cycles"][0]["sequences"][3] = "GSE"
    with pytest.raises(ValueError, match="rectangle"):
        framework.cycle_rows(broken, wt, {1: {}, 2: {}}, clean)
    touched = copy.deepcopy(clean)
    touched["accounting"]["backgrounds_touched"] = ["1UFM.pdb"]
    with pytest.raises(ValueError, match="clean-state identity"):
        framework.cycle_rows(background, wt, {1: {}, 2: {}}, touched)


def test_public_bundle_and_editable_annotation_contract(tmp_path, monkeypatch):
    import json
    import xml.etree.ElementTree as ET
    from PIL import Image
    monkeypatch.setattr(framework, 'ROOT', tmp_path)
    assets = tmp_path / 'manuscript/assets'
    assets.mkdir(parents=True)
    monkeypatch.setattr(framework, 'ASSETS', assets)
    manifest = {'panels': [], 'records': {}, 'inputs': {'dvl1': {'path': 'af.pdb', 'sha256': 'a' * 64}},
                'structures': {'dvl1': {'file': 'af.pdb', 'accession': 'O14640'}}}
    names = ('dvl1-abundance', '1ufm-stability', '1ufm-pair')
    for name in names:
        path = assets / f'{name}.png'
        Image.new('RGBA', (16, 16), '#345678').save(path)
        manifest['records'][name] = {'selected': {'id': name + ':A1G', 'substitution': 'A1G'}}
        manifest['panels'].append({'name': name, 'record': name, 'title': name, 'positions': [1],
                                   'image_sha256': framework.digest(path)})
    framework.annotated_exports(manifest)
    selection = assets / 'selection.json'
    selection.write_text(json.dumps(manifest))
    framework.bundle_insets(manifest)
    bundle = tmp_path / 'scripts/capability/reporting/figure_data/structure-insets'
    receipt = json.loads((bundle / 'manifest.json').read_text())
    assert receipt['schema'] == 'manuscript-structure-insets-v1'
    assert set(receipt['images']) == set(names)
    assert receipt['source_selection']['path'] == 'manuscript/assets/selection.json'
    assert receipt['source_selection']['sha256'] == framework.digest(selection)
    assert set(p.name for p in bundle.iterdir()) == {'manifest.json', *(name + '.png' for name in names)}
    for name in names:
        assert (bundle / (name + '.png')).read_bytes() == (assets / (name + '.png')).read_bytes()
        svg = assets / 'annotated' / (name + '.svg')
        labels = [node.text for node in ET.fromstring(svg.read_text()).iter()
                  if node.tag.rsplit('}', 1)[-1] == 'text']
        assert 'A1G (WT CA site)' in labels
        assert (assets / 'annotated' / (name + '.png')).is_file()
    (assets / (names[0] + '.png')).write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs from final selection'):
        framework.bundle_insets(manifest)


def test_pinned_local_sources_and_marker_numbering():
    if not all((framework.ROOT / path).is_file() for path in framework.SOURCES.values()):
        pytest.skip("Pinned local research payloads are deliberately not distributed")
    manifest, maps, panels = framework.select_sources()
    assert manifest["structures"]["ubc9"]["absent_wt_positions"] == [159]
    assert len(maps["ubc9"]) == 158
    assert len(maps["dvl1"]) == 93
    assert maps["pci"][1]["coordinate_id"] == 5
    assert maps["pci"][1]["author_id"] == "293"
    assert maps["pci"][72]["author_id"] == "364"
    assert manifest["structures"]["pci"]["models"] == 20
    assert manifest["structures"]["pci"]["model"] == 1
    for panel in panels:
        for position in panel["positions"]:
            assert len(maps[panel["structure"]][position]["ca"]) == 3
    for key in ("ubc9", "dvl1", "stability", "pair"):
        assert manifest["records"][key]["selection"]["qualified_eligible"] > 0
