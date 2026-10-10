"""The E11 structural panel: two draws, one copying distribution, paired intervals.

These tests guard the properties the reading of E11 rests on rather than the
current arithmetic: that the frozen 2026-10-08 declaration is still byte-identical
after the 2026-10-10 structural extension was added beside it, that a condition
with no conditioning context is never given a manufactured counterpart, that a
family group supplying only one side of a contrast is missing rather than zero,
and that a fold which did not keep its pairwise confidence is refused.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.capability.context import homology_context as H  # noqa: E402

#: The digests every artefact frozen under the 2026-10-08 declaration carries.
#: Pinned as literals: the structural extension was added to this module after
#: those artefacts were written, and a change that moves any of these silently
#: invalidates the E09 scores, the retrieval artefact and the frozen product set
#: rather than failing anywhere a reader would look.
FROZEN_DIGESTS = {
    "search_sha256": "253a0487621bf3773aed578415c157e882f688137c4595eac52cd60b9248bce8",
    "retrieval_sha256": "efee401c48ace5854c7b1ad61b197a2ebb6609260711275a8a32046a93ae7bd0",
    "context_sha256": "ef5284957fa514fb6201cb4a3432ce74f62eafceab69f2520c3e52d2012e7373",
    "declaration_sha256": "4464589dbfbc55c78eaafa18ea9630ba734c2c097231d62e6902c7d4c93de9c3",
}


def product(
    attempt: str,
    condition: str,
    residues: int,
    *,
    target: str = "q0",
    stop: str = "native_terminal",
    lcs: int = 3,
    containment: float = 0.0,
    items: int = 4,
    identity: float | None = None,
    copy: bool = False,
) -> dict:
    row = {
        "attempt_id": attempt,
        "arm": "progen2-medium",
        "target_id": target,
        "condition": condition,
        "residues": residues,
        "stop_status": stop,
        "context_items": 0 if condition == H.NO_CONTEXT else items,
        "copy_statistics": {
            "max_lcs_to_context": lcs,
            "lcs_fraction_of_product": lcs / residues,
            "max_kmer_containment": containment,
        },
        "copy_verdict": {"rules": {}, "is_copy": copy, "fired": ["kmer_containment"] if copy else []},
    }
    if identity is not None:
        row["context_alignment_identity"] = identity
    return row


def folded(row: dict, plddt: float, *, ptm: float = 0.4, pae: float = 20.0) -> dict:
    return {**row, "plddt": plddt, "ptm": ptm, "pae": pae}


def test_the_frozen_declaration_is_unchanged_by_the_structural_extension():
    assert H.declaration_digests() == FROZEN_DIGESTS
    # The extension publishes its own digest and is not a section of the frozen
    # declaration, which is the whole reason the four digests above still hold.
    assert "structure_unmatched" not in H.declaration()
    assert H.structure_extension()["predeclared_utc"] == "2026-10-10"
    assert H.structure_extension_digest() != FROZEN_DIGESTS["declaration_sha256"]


def test_the_extension_declares_both_estimands_and_refuses_to_call_either_stability():
    extension = H.structure_extension()
    assert set(extension["estimands"]) == {"length_matched", "unmatched"}
    assert "fixed product length" in extension["estimands"]["length_matched"]
    assert "confounds homology with product length" in extension["estimands"]["unmatched"]
    disclaimer = extension["not_stability"]
    assert "thermodynamic stability" in disclaimer
    assert "none is measured" in disclaimer
    assert "no wet-lab result" in disclaimer
    # Pairwise confidence is declared as kept, which is what the fold stage honours.
    assert extension["pairwise_fields"] == ["pae", "pde", "distogram_logits"]


def test_the_generation_conditions_have_one_source():
    stage = ROOT / "scripts/capability/context/generate_homolog_conditioned.py"
    module = {}
    exec(  # noqa: S102 - reading one constant out of the stage without importing torch
        "\n".join(
            line
            for line in stage.read_text(encoding="utf-8").splitlines()
            if line.startswith("GENERATION_CONDITIONS")
        ),
        {"H": H},
        module,
    )
    assert module["GENERATION_CONDITIONS"] is H.GENERATION_CONDITIONS


def test_the_unmatched_draw_is_equal_count_and_keeps_its_length_difference():
    attempts = [
        product(f"short{index}", H.CLOSE_HOMOLOG, 150) for index in range(10)
    ] + [product(f"long{index}", H.NO_CONTEXT, 400) for index in range(4)]
    conditions = [H.CLOSE_HOMOLOG, H.NO_CONTEXT]
    selected, record = H.select_unmatched_structure_products(
        attempts, conditions=conditions, samples=8
    )
    # Equal count per condition: the minimum availability, not the cap.
    assert record["drawn_per_condition"] == 4
    assert len(selected) == 8
    per_condition = {
        condition: sum(
            1
            for attempt in attempts
            if attempt["attempt_id"] in selected and attempt["condition"] == condition
        )
        for condition in conditions
    }
    assert per_condition == {H.CLOSE_HOMOLOG: 4, H.NO_CONTEXT: 4}
    # The length difference is reported, not removed: that is the point of the draw.
    spread = record["length_distribution_per_condition"]
    assert spread[H.CLOSE_HOMOLOG]["median_residues"] == 150.0
    assert spread[H.NO_CONTEXT]["median_residues"] == 400.0
    assert "no length stratification" in record["rule"]


def test_the_unmatched_draw_is_deterministic_and_skips_unmeasurable_products():
    attempts = [product(f"a{index}", H.CLOSE_HOMOLOG, 150) for index in range(6)]
    attempts += [product(f"tiny{index}", H.CLOSE_HOMOLOG, 8) for index in range(6)]
    attempts += [product(f"b{index}", H.UNRELATED, 150) for index in range(6)]
    first, record = H.select_unmatched_structure_products(
        attempts, conditions=[H.CLOSE_HOMOLOG, H.UNRELATED], samples=4
    )
    second, _ = H.select_unmatched_structure_products(
        attempts, conditions=[H.CLOSE_HOMOLOG, H.UNRELATED], samples=4
    )
    assert first == second
    assert not any(identifier.startswith("tiny") for identifier in first)
    assert record["available_per_condition"] == {H.CLOSE_HOMOLOG: 6, H.UNRELATED: 6}
    with pytest.raises(ValueError):
        H.select_unmatched_structure_products(attempts, conditions=[H.CLOSE_HOMOLOG], samples=0)


def test_an_empty_context_gets_no_manufactured_identity_counterpart():
    products = [
        product("c0", H.CLOSE_HOMOLOG, 200, identity=61.0),
        product("c1", H.CLOSE_HOMOLOG, 200, identity=None),
        product("n0", H.NO_CONTEXT, 200, identity=None),
    ]
    # The empty-context row has no context key at all; mark the annotated ones.
    products[1]["context_alignment_identity"] = None
    out = H.context_identity_distribution(
        products, conditions=[H.CLOSE_HOMOLOG, H.NO_CONTEXT]
    )
    empty = out[H.NO_CONTEXT]["context_alignment_identity"]
    assert empty["status"] == "not applicable"
    assert "nothing for its" in empty["reason"]
    close = out[H.CLOSE_HOMOLOG]["context_alignment_identity"]
    assert close["status"] == "evaluated"
    assert close["aligned"] == 1 and close["unaligned"] == 1
    # The two summaries are different quantities and both are published.
    assert close["among_aligned_percent"]["mean"] == pytest.approx(61.0)
    assert close["unaligned_as_zero_percent"]["mean"] == pytest.approx(30.5)
    assert close["at_or_over_copy_threshold"] == 1


def test_an_unannotated_product_set_reports_not_evaluated_rather_than_zero():
    products = [product("c0", H.CLOSE_HOMOLOG, 200)]
    out = H.context_identity_distribution(products, conditions=[H.CLOSE_HOMOLOG])
    identity = out[H.CLOSE_HOMOLOG]["context_alignment_identity"]
    assert identity["status"] == "not evaluated"
    # The substring statistics do not need an aligner and are reported regardless.
    assert out[H.CLOSE_HOMOLOG]["max_lcs_to_context"]["quantiles"]["q50"] == 3.0


def test_a_structural_cell_carries_length_and_completeness_with_its_confidence():
    rows = [
        folded(product("a", H.CLOSE_HOMOLOG, 200), 0.5),
        folded(product("b", H.CLOSE_HOMOLOG, 400, stop="budget_censored"), 0.3),
    ]
    cell = H.summarise_structure_cell(rows)
    assert cell["folded"] == 2
    assert cell["native_terminal"] == 1 and cell["budget_censored"] == 1
    assert cell["mean_residues"] == 300.0
    assert cell["residue_range"] == [200, 400]
    assert cell["mean_pae"] == pytest.approx(20.0)


def _panel_rows(difference: float, *, groups: int = 10) -> list[dict]:
    rows = []
    for index in range(groups):
        base = 0.40 + 0.01 * index
        rows.append(
            folded(product(f"u{index}", H.UNRELATED, 200, target=f"q{index}"), base)
        )
        rows.append(
            folded(
                product(f"c{index}", H.CLOSE_HOMOLOG, 200, target=f"q{index}"),
                base + difference,
            )
        )
    return rows


def test_the_contrast_panel_pairs_inside_the_family_group():
    rows = _panel_rows(0.05)
    panel = H.structure_contrast_panel(
        {"len_129_256": rows},
        contrasts=((H.CLOSE_HOMOLOG, H.UNRELATED),),
        draws=400,
    )
    column = panel["columns"][0]
    assert column["status"] == "estimated"
    assert column["paired_groups"] == 10
    # Pairing removes the between-target spread entirely, so the point estimate is
    # the difference itself and the interval is tight around it.
    assert column["point"] == pytest.approx(0.05)
    assert column["pointwise_interval"][0] == pytest.approx(0.05)
    assert column["degenerate"] is False


def test_a_group_on_only_one_side_of_a_contrast_is_missing_not_zero():
    rows = _panel_rows(0.05, groups=9)
    # One more target under the homologue condition alone, scoring far higher. If
    # the absent referent were imputed as zero it would dominate the contrast.
    rows.append(folded(product("lonely", H.CLOSE_HOMOLOG, 200, target="qZ"), 0.95))
    panel = H.structure_contrast_panel(
        {"len_129_256": rows},
        contrasts=((H.CLOSE_HOMOLOG, H.UNRELATED),),
        draws=400,
    )
    column = panel["columns"][0]
    assert column["paired_groups"] == 9
    assert column["left_products"] == 10 and column["right_products"] == 9
    assert column["point"] == pytest.approx(0.05)


def test_a_contrast_without_two_paired_groups_is_reported_not_estimable():
    rows = [
        folded(product("u0", H.UNRELATED, 200, target="q0"), 0.40),
        folded(product("c0", H.CLOSE_HOMOLOG, 200, target="q0"), 0.45),
        folded(product("u1", H.UNRELATED, 200, target="q1"), 0.41),
        folded(product("c1", H.CLOSE_HOMOLOG, 200, target="q1"), 0.47),
        # no_context appears for one group only, so its contrasts cannot be paired.
        folded(product("n0", H.NO_CONTEXT, 200, target="q0"), 0.30),
    ]
    panel = H.structure_contrast_panel({"len_129_256": rows}, draws=400)
    by_contrast = {column["contrast"]: column for column in panel["columns"]}
    assert by_contrast["close_homolog_minus_unrelated"]["status"] == "estimated"
    for name in (
        "close_homolog_minus_no_context",
        "remote_homolog_minus_unrelated",
        "unrelated_minus_no_context",
    ):
        assert by_contrast[name]["status"] == "not estimable"
        assert "two family groups" in by_contrast[name]["reason"]


def test_the_panel_flags_a_group_count_below_the_floor_and_names_the_simultaneity():
    rows = _panel_rows(0.05, groups=4)
    panel = H.structure_contrast_panel(
        {"len_129_256": rows},
        contrasts=((H.CLOSE_HOMOLOG, H.UNRELATED),),
        draws=400,
    )
    column = panel["columns"][0]
    assert column["degenerate"] is True
    assert str(H.GROUP_FLOOR) in column["degenerate_reason"]
    assert "does not control the family" in panel["interval_reading"]


def test_the_simultaneous_interval_is_never_narrower_than_the_pointwise_one():
    rng = np.random.default_rng(11)
    rows: list[dict] = []
    for index in range(12):
        for condition, shift in (
            (H.UNRELATED, 0.0),
            (H.CLOSE_HOMOLOG, 0.04),
            (H.REMOTE_HOMOLOG, 0.02),
            (H.NO_CONTEXT, -0.03),
        ):
            rows.append(
                folded(
                    product(f"{condition}{index}", condition, 200, target=f"q{index}"),
                    0.45 + shift + float(rng.normal(0.0, 0.02)),
                )
            )
    panel = H.structure_contrast_panel({"len_129_256": rows}, draws=2000)
    assert panel["family_size"] == len(H.STRUCTURE_CONTRASTS)
    assert panel["critical_value"] >= 1.96
    for column in panel["columns"]:
        width = column["simultaneous_interval"][1] - column["simultaneous_interval"][0]
        pointwise = column["pointwise_interval"][1] - column["pointwise_interval"][0]
        assert width >= pointwise * 0.95


def test_the_panel_refuses_an_undeclared_confidence_field():
    with pytest.raises(ValueError, match="not a declared confidence field"):
        H.structure_contrast_panel({"all": _panel_rows(0.05)}, field="stability")


def test_condition_levels_are_reported_over_groups_and_warn_against_misreading():
    rows = _panel_rows(0.05)
    levels = H.condition_level_summary(
        {"len_129_256": rows}, conditions=[H.UNRELATED, H.CLOSE_HOMOLOG]
    )
    cell = levels["per_stratum"]["len_129_256"][H.CLOSE_HOMOLOG]
    assert cell["groups"] == 10
    assert cell["mean"] == pytest.approx(0.495)
    assert len(cell["interval"]) == 2
    assert "read the contrast panel" in levels["note"]


# ------------------------------------------------- the two stages, end to end


def load_fold_stage():
    """The fold stage as a module. Loaded by path because it is a stage script."""

    spec = importlib.util.spec_from_file_location(
        "fold_conditioned_products",
        ROOT / "scripts/capability/context/fold_conditioned_products.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_products(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def synthetic_products() -> list[dict]:
    rows: list[dict] = []
    for target in range(10):
        for condition in H.GENERATION_CONDITIONS:
            for attempt in range(6):
                rows.append(
                    product(
                        f"progen2-medium|q{target:02d}|{condition}|{attempt:04d}",
                        condition,
                        150 + 10 * attempt,
                        target=f"q{target:02d}",
                    )
                )
    return rows


def run_draw(tmp_path: Path, rows: list[dict]) -> subprocess.CompletedProcess:
    products = tmp_path / "products.jsonl"
    write_products(products, rows)
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/capability/context/draw_structure_sets.py"),
            "--products",
            str(products),
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )


def test_the_draw_stage_writes_both_sets_and_reproduces_the_frozen_selection(tmp_path):
    rows = synthetic_products()
    # The generation stage draws in the DECLARED condition order, and the draw
    # consumes one generator as the cells are visited, so the order is part of
    # the selection: reproducing it alphabetically recovered 11 of the 128
    # attempts the real E11 run had frozen.
    identifiers, _ = H.select_structure_products(rows, conditions=list(H.GENERATION_CONDITIONS))
    for row in rows:
        row["structure_selected"] = row["attempt_id"] in identifiers
    done = run_draw(tmp_path, rows)
    assert done.returncode == 0, done.stderr
    record = json.loads((tmp_path / "out/structure_draw.json").read_text())
    assert record["reproduced_frozen_selection"] is True
    assert record["structure_extension_sha256"] == H.structure_extension_digest()
    assert set(record["sets"]) == {"length_matched", "unmatched"}
    matched = [
        json.loads(line)
        for line in (tmp_path / "out/length_matched_products.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {row["structure_set"] for row in matched} == {"length_matched"}
    assert all(row["structure_selected"] for row in matched)
    counts = set(record["sets"]["unmatched"]["per_condition"].values())
    assert len(counts) == 1, "the unmatched draw must be equal-count per condition"


def test_the_draw_stage_refuses_a_selection_drawn_in_another_condition_order(tmp_path):
    rows = synthetic_products()
    alphabetical, _ = H.select_structure_products(
        rows, conditions=sorted(H.GENERATION_CONDITIONS)
    )
    declared, _ = H.select_structure_products(rows, conditions=list(H.GENERATION_CONDITIONS))
    assert alphabetical != declared, "the fixture cannot distinguish the two orders"
    for row in rows:
        row["structure_selected"] = row["attempt_id"] in alphabetical
    done = run_draw(tmp_path, rows)
    assert done.returncode != 0
    assert "not the one frozen into" in done.stderr


def test_the_draw_stage_refuses_a_product_file_whose_selection_it_cannot_reproduce(tmp_path):
    rows = synthetic_products()
    for row in rows:
        row["structure_selected"] = row["attempt_id"].endswith("0000")
    done = run_draw(tmp_path, rows)
    assert done.returncode != 0
    assert "not the one frozen into" in done.stderr


def test_the_draw_stage_refuses_an_undeclared_condition_set(tmp_path):
    rows = [row for row in synthetic_products() if row["condition"] != H.NO_CONTEXT]
    done = run_draw(tmp_path, rows)
    assert done.returncode != 0
    assert "different experiment" in done.stderr


def test_a_fold_keeps_its_pairwise_confidence_or_refuses_to_record(tmp_path):
    torch = pytest.importorskip("torch")
    fold = load_fold_stage()

    samples, length = 4, 12
    output = {
        "pae": torch.rand(samples, length, length) * 30.0,
        "pde": torch.rand(samples, length, length) * 30.0,
        "distogram_logits": torch.rand(1, length, length, 8),
        "plddt": torch.rand(samples, length),
        "plddt_ca": torch.rand(samples, length),
        "ptm": torch.rand(samples),
        "complex_plddt": torch.tensor([0.1, 0.9, 0.2, 0.3]),
    }
    path = fold.save_pairwise(output, tmp_path, "arm|q00|close_homolog|0001")
    with np.load(path) as archive:
        assert set(fold.PAIRWISE_FIELDS) <= set(archive)
        # Whole arrays, not scalars: the sample axis and both residue axes survive.
        assert archive["pae"].shape == (samples, length, length)
        assert archive["distogram_logits"].shape == (1, length, length, 8)
        assert archive["pae"].dtype == np.float16
        assert archive["plddt"].dtype == np.float32
    assert "|" not in path.name

    best = fold.best_sample(output)
    assert best["index"] == 1 and best["ranked_on"] == "complex_plddt"
    assert best["plddt"] == pytest.approx(float(output["plddt"][1].mean()))

    with pytest.raises(RuntimeError, match="pairwise confidence was not kept"):
        fold.save_pairwise({"plddt": torch.rand(samples, length)}, tmp_path, "bare")
