"""Publication anchors, exact centered zero, and side-effect-free CLI contracts."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from matplotlib.colors import TwoSlopeNorm, to_hex
import pytest

from scripts.capability.reporting import palette, render_framework

ROOT = Path(__file__).resolve().parents[2]


def test_named_roles_and_author_supplied_anchor_order():
    assert (palette.BINARY_FIRST, palette.BINARY_SECOND) == ("#44729D", "#D48640")
    assert palette.MODEL_CLASSES == ("#4575B4", "#7BCCC4", "#E6B400", "#C62921")
    assert palette.CONTROL == "#000000"
    assert (palette.NEGATIVE, palette.ZERO, palette.POSITIVE) == ("#ED2122", "#FFFFFF", "#9F8CC3")
    assert palette.BACKBONE == palette.DOWNSTREAM == palette.BINARY_FIRST
    assert palette.SITE == palette.NEGATIVE
    assert palette.SECOND_SITE == palette.COMPLETE_DOMAIN == palette.BINARY_SECOND
    assert palette.NATIVE == palette.ANY_FAMILY == palette.BINARY_FIRST
    assert palette.READOUT == palette.BINARY_SECOND


@pytest.mark.parametrize("limits", [(-1, 1), (-.022, .022), (-.1, .9), (-4, .2)])
def test_signed_map_has_exact_white_zero_even_with_asymmetric_limits(limits):
    norm = TwoSlopeNorm(vmin=limits[0], vcenter=0, vmax=limits[1])
    assert norm(0) == .5
    assert palette.SIGNED(norm(0)) == (1., 1., 1., 1.)
    assert to_hex(palette.SIGNED(norm(limits[0]))).upper() == palette.NEGATIVE
    assert to_hex(palette.SIGNED(norm(limits[1]))).upper() == palette.POSITIVE
    assert norm(limits[0]) < norm(0) < norm(limits[1])
    assert palette.SIGNED.N == 257


def test_framework_help_exits_without_reading_assets_or_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["render_framework.py", "--help"])
    monkeypatch.setattr(render_framework, "ROOT", tmp_path)
    monkeypatch.setattr(render_framework, "OUT", tmp_path / "outputs")

    def forbidden(*args):
        pytest.fail("Help must not read asset/reference hashes")

    monkeypatch.setattr(render_framework, "sha", forbidden)
    with pytest.raises(SystemExit) as exit_info:
        render_framework.main()
    assert exit_info.value.code == 0
    assert list(tmp_path.iterdir()) == []


def test_supplement_only_refuses_changed_shared_palette_in_public_copy(tmp_path):
    reporting = tmp_path / "scripts/capability/reporting"
    data = reporting / "figure_data/main"
    data.mkdir(parents=True)
    for name in ("render_figures.py", "palette.py", "checkpoint_labels.py", "publication_paths.py"):
        shutil.copyfile(ROOT / "scripts/capability/reporting" / name, reporting / name)
    shutil.copyfile(ROOT / "scripts/capability/reporting/figure_data/main/figure_validation.json",
                    data / "figure_validation.json")
    # A valid, styling-only change must invalidate supplements before table reads.
    with (reporting / "palette.py").open("a") as handle:
        handle.write("\n# changed palette dependency\n")
    result = subprocess.run([sys.executable, str(reporting / "render_figures.py"), "--supplement-only"],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "Run the full renderer first" in result.stderr
    assert not (tmp_path / "manuscript").exists()


def test_current_receipts_bind_palette_and_canonical_author_artwork():
    relative = str(palette.PATH.relative_to(ROOT))
    receipt = json.loads((ROOT / "scripts/capability/reporting/figure_data/main/figure_validation.json").read_text())
    assert receipt["renderer_sha256"][relative] == render_framework.sha(palette.PATH)
    framework = json.loads((ROOT / "scripts/capability/reporting/figure_data/main/framework_validation.json").read_text())
    assert framework["source_sha256"][relative] == render_framework.sha(palette.PATH)
    selection_path = ROOT / "manuscript/assets/selection.json"
    if not render_framework.REFERENCE.is_file() or not selection_path.is_file():
        pytest.skip("Author artwork and full structure provenance are not distributed")
    assert render_framework.REFERENCE.name == "framework-original.png"
    assert framework["source_sha256"][str(render_framework.REFERENCE.relative_to(ROOT))] == render_framework.sha(render_framework.REFERENCE)
    selection = json.loads(selection_path.read_text())
    assert selection["palette_dependency"] == {"path": relative, "sha256": render_framework.sha(palette.PATH)}
