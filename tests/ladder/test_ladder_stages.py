"""The ladder's stages, end to end on fabricated folds, with no GPU and no checkpoint.

The pipeline this exercises is the real one: the backbone selector's admission
and attrition, the composition-matched extent reference, the fold-cohort union,
the parent comparison against stored prediction objects, and the analysis that
turns all of it into a ladder with intervals and a divergence rung. Only the
three things that need a GPU -- the two samplers and the likelihood readout --
are replaced by fabricated products, and they are fabricated with a *planted*
relation so the analysis has to find the divergence where it was put.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.capability.ladder import design, runtime

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "capability" / "ladder"


def _stage(name: str):
    spec = importlib.util.spec_from_file_location(f"ladder_stage_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------ draw statuses


def test_a_draw_that_filled_its_window_is_recorded_as_filled():
    assert runtime.classify_draw("ACDEFGH", "ACDEFGH", 5, end_delimiter="<end>") == runtime.DRAW_FILLED


def test_a_decoder_that_ended_the_protein_early_is_distinguished_from_one_that_ran_out():
    assert (
        runtime.classify_draw("ACD<end>", "ACD", 10, end_delimiter="<end>")
        == runtime.DRAW_TERMINATED_SHORT
    )
    assert (
        runtime.classify_draw("ACDEFG", "ACDEFG", 10, end_delimiter="<end>")
        == runtime.DRAW_BUDGET_EXHAUSTED
    )
    assert (
        runtime.classify_draw("ACD and then prose", "ACD", 10, end_delimiter="<end>")
        == runtime.DRAW_OFF_ALPHABET
    )


def test_a_fasta_newline_does_not_count_as_wandering_off_the_alphabet():
    assert (
        runtime.classify_draw("ACD\nEFG", "ACDEFG", 10, end_delimiter="<|endoftext|>")
        == runtime.DRAW_BUDGET_EXHAUSTED
    )


# --------------------------------------------------------- backbone selection


def _candidate(index: int, *, length: int, sequence: str | None = None, ok: bool = True):
    body = sequence or ("ACDEFGHIKLMNPQRSTVWY" * 40)[:length]
    return {
        "id": f"nat_P{index:04d}",
        "accession": f"P{index:04d}",
        "length": length,
        "sequence": body,
        "roles": ["natural"],
        "structure": {
            "status": "ok" if ok else "error",
            "mean_ca_plddt": 96.0,
            "fraction_ca_plddt_ge70": 0.99,
            "ptm": 0.93,
            "mean_pae_angstrom": 3.1,
            "sequence_sha256": design.sequence_digest(body),
            "object_directory": f"objects/{design.sequence_digest(body)}",
        },
    }


def _diverse_pool():
    """A pool wide enough to fill every stratum with distinct, non-duplicate members."""

    generator = np.random.default_rng(99)
    rows = []
    labels = {}
    index = 0
    for low, high in design.length_strata():
        for offset in range(8):
            length = low + offset
            sequence = "".join(generator.choice(list(design.AA20), size=length))
            rows.append(_candidate(index, length=length, sequence=sequence))
            labels[f"P{index:04d}"] = [f"{1 + index % 6}.{index}.{offset}.1"]
            index += 1
    return rows, labels


def test_the_selector_fills_every_stratum_with_distinct_ec_numbers():
    stage = _stage("build_ladder_backbones")
    rows, labels = _diverse_pool()
    admitted, census = stage.select(rows, labels)
    assert len(admitted) == design.N_BACKBONES
    assert census["n_admitted"] == design.N_BACKBONES
    per_stratum = {index: 0 for index in range(design.BACKBONE_STRATA)}
    for member in admitted:
        per_stratum[member["stratum"]] += 1
        assert design.BACKBONE_LENGTH_BAND[0] <= member["length"] <= design.BACKBONE_LENGTH_BAND[1]
        assert set(member["window_spans"]) == {f"k{k}" for k in design.WINDOW_EXTENTS}
    assert set(per_stratum.values()) == {design.BACKBONES_PER_STRATUM}
    assert len({member["ec_label"] for member in admitted}) == design.N_BACKBONES


def test_the_selector_records_why_each_candidate_was_refused():
    stage = _stage("build_ladder_backbones")
    rows, labels = _diverse_pool()
    rows.append(_candidate(900, length=80))
    labels["P0900"] = ["9.9.9.9"]
    rows.append(_candidate(901, length=200, ok=False))
    labels["P0901"] = ["9.9.9.8"]
    rows.append(_candidate(902, length=200))
    _admitted, census = stage.select(rows, labels)
    assert census["attrition"]["outside_length_band"] >= 1
    assert census["attrition"]["fold_not_ok"] >= 1
    assert census["attrition"]["no_single_ec_label"] >= 1


def test_the_selector_refuses_a_near_duplicate_of_an_admitted_backbone():
    stage = _stage("build_ladder_backbones")
    rows, labels = _diverse_pool()
    twin = dict(rows[0])
    twin["id"] = "nat_TWIN"
    twin["accession"] = "PTWIN"
    labels["PTWIN"] = ["8.8.8.8"]
    rows.append(twin)
    admitted, census = stage.select(rows, labels)
    accessions = {member["accession"] for member in admitted}
    assert not {"P0000", "PTWIN"}.issubset(accessions)
    assert census["attrition"].get("near_duplicate_of_admitted", 0) >= 1


def test_a_pool_too_small_for_the_declared_set_is_refused_not_shrunk(tmp_path):
    stage = _stage("build_ladder_backbones")
    rows, labels = _diverse_pool()
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows[:3]), encoding="utf-8")
    fasta = tmp_path / "ec.fasta"
    fasta.write_text(
        "".join(f">{accession}|{value[0]}\nAAAA\n" for accession, value in labels.items()),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {"out": tmp_path / "out", "fold_index": [index], "ec_fasta": fasta, "device": "cpu"},
    )()
    with pytest.raises(SystemExit, match="admissible backbones"):
        stage.run(args)


# ------------------------------------------------- the composition reference


def test_the_extent_reference_draws_from_the_parent_composition_and_is_reproducible():
    stage = _stage("shuffle_ladder_windows")
    parent = "A" * 90 + "C" * 10
    window = stage.draw_window(parent, 40, seed=7)
    assert len(window) == 40
    assert set(window) <= {"A", "C"}
    assert window.count("A") > window.count("C")
    assert stage.draw_window(parent, 40, seed=7) == window
    assert stage.draw_window(parent, 40, seed=8) != window


# ------------------------------------------------------ the end-to-end ladder

N_BACKBONES = 8
LENGTH = 60
RUNGS = ("k1", "k2", "k5")
DRAWS = 4
ARM = "protgpt2"


def _helix(n, *, rise=1.5, radius=2.3, turn=100.0):
    angles = np.deg2rad(turn * np.arange(n))
    return np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), rise * np.arange(n)], axis=1
    )


def _pdb(coordinates):
    lines = []
    for index, (x, y, z) in enumerate(coordinates, start=1):
        lines.append(
            f"ATOM  {index:5d} {'CA':<4s} {'ALA':>3s} A{index:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C  "
        )
    return "\n".join(lines + ["TER", "END"]) + "\n"


def _write_object(root: Path, digest: str, coordinates: np.ndarray):
    directory = root / "objects" / digest
    directory.mkdir(parents=True, exist_ok=True)
    length = coordinates.shape[0]
    np.savez_compressed(
        directory / "prediction.npz",
        ca_plddt_0_100=np.full(length, 92.0),
        predicted_aligned_error_angstrom=np.full((length, length), 2.5),
        ptm=np.asarray([0.9]),
        diffusion_sample_index=np.asarray([0]),
        length=np.asarray([length]),
    )
    (directory / "prediction.pdb").write_text(_pdb(coordinates), encoding="utf-8")
    (directory / "result.json").write_text(
        json.dumps({"status": "ok", "sequence_sha256": digest, "files_sha256": {}}),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, planted_divergence_rung: str = "k5"):
    """Eight backbones, three rungs, four draws, with a planted divergence."""

    generator = np.random.default_rng(2026)
    parent_trace = _helix(LENGTH)
    backbones = []
    variants = []
    likelihood = []
    variant_root = tmp_path / "variant_folds"
    parent_root = tmp_path / "parent_folds"
    for index in range(N_BACKBONES):
        sequence = "".join(generator.choice(list(design.AA20), size=LENGTH))
        digest = design.sequence_digest(sequence)
        backbone = {
            "backbone_id": f"lb_B{index}",
            "accession": f"B{index}",
            "ec_label": f"1.{index}.1.1",
            "stratum": index % design.BACKBONE_STRATA,
            "length": LENGTH,
            "sequence": sequence,
            "parent_sequence_sha256": digest,
            "parent_object_directory": f"objects/{digest}",
            "parent_mean_ca_plddt": 95.0,
            "parent_fraction_ca_plddt_ge70": 0.99,
            "parent_ptm": 0.92,
            "parent_mean_pae_angstrom": 3.0,
            "window_spans": {rung: list(design.window_span(LENGTH, design.rung_extent(rung))) for rung in RUNGS},
        }
        backbones.append(backbone)
        _write_object(parent_root, digest, parent_trace)
        likelihood.append(
            {
                "sequence_sha256": digest,
                "role": "parent",
                "scored_by": ARM,
                "mean_nll_per_token_nats": 2.0,
                "likelihood_kind": design.LIKELIHOOD_CAUSAL_NLL,
                "length": LENGTH,
            }
        )
        for rung in RUNGS:
            start, stop = backbone["window_spans"][rung]
            extent = design.rung_extent(rung)
            for draw in range(DRAWS):
                window = "".join(generator.choice(list(design.AA20), size=extent))
                sequence_variant = design.splice(sequence, start, window)
                # Make every variant string distinct, so one fold per row.
                displacement = 1.0 + 3.0 * draw
                trace = parent_trace.copy()
                trace[start:stop] += np.asarray([displacement, 0.0, 0.0])
                digest_variant = design.sequence_digest(sequence_variant)
                _write_object(variant_root, digest_variant, trace)
                variants.append(
                    design.variant_record(
                        arm_name=ARM,
                        condition=design.CONDITION_CAUSAL_PREFIX,
                        backbone=backbone,
                        rung=rung,
                        draw=draw,
                        sequence=sequence_variant,
                        status="filled",
                    )
                )
                # Below the planted divergence the likelihood tracks the
                # displacement exactly; at and above it, it is noise.
                nll = (
                    displacement
                    if rung != planted_divergence_rung
                    else float(generator.uniform(0.0, 10.0))
                )
                likelihood.append(
                    {
                        "sequence_sha256": digest_variant,
                        "role": "variant",
                        "scored_by": ARM,
                        "mean_nll_per_token_nats": nll,
                        "likelihood_kind": design.LIKELIHOOD_CAUSAL_NLL,
                        "length": LENGTH,
                    }
                )
    paths = {
        "backbones": tmp_path / "ladder_backbones.jsonl",
        "variants": tmp_path / "ladder_variants.jsonl",
        "likelihood": tmp_path / "ladder_likelihood.jsonl",
    }
    design.write_jsonl(paths["backbones"], backbones)
    design.write_jsonl(paths["variants"], variants)
    design.write_jsonl(paths["likelihood"], likelihood)
    return paths | {"variant_root": variant_root, "parent_root": parent_root}


def test_the_fold_cohort_unions_the_arms_and_keeps_unfilled_rows_out_of_it(tmp_path):
    stage = _stage("build_ladder_fold_cohort")
    fixture = _fixture(tmp_path)
    rows = design.read_jsonl(fixture["variants"])
    rows.append({**rows[0], "id": "lv_unfilled", "status": "terminated_short", "sequence": "", "sequence_sha256": ""})
    design.write_jsonl(tmp_path / "with_failure.jsonl", rows)
    args = type(
        "Args",
        (),
        {"out": tmp_path / "cohort", "variants": [tmp_path / "with_failure.jsonl"], "device": "cpu"},
    )()
    payload = stage.run(args)
    assert payload["n_rows"] == N_BACKBONES * len(RUNGS) * DRAWS
    assert payload["skipped_unfilled"] == {f"{ARM}|terminated_short": 1}
    cohort = design.read_jsonl(tmp_path / "cohort" / "ladder_fold_cohort.jsonl")
    assert all(set(row) >= {"id", "sequence", "sequence_sha256"} for row in cohort)


def test_a_missing_parent_fold_is_a_refusal_rather_than_a_refold(tmp_path):
    stage = _stage("compare_ladder_folds")
    fixture = _fixture(tmp_path)
    empty = tmp_path / "no_parents"
    (empty / "objects").mkdir(parents=True)
    args = type(
        "Args",
        (),
        {
            "out": tmp_path / "structure",
            "device": "cpu",
            "backbones": fixture["backbones"],
            "variants": [fixture["variants"]],
            "structure": [fixture["variant_root"]],
            "parent_structure": [empty],
        },
    )()
    with pytest.raises(SystemExit, match="not available"):
        stage.run(args)


def test_the_pipeline_finds_the_divergence_where_it_was_planted(tmp_path):
    compare = _stage("compare_ladder_folds")
    analyse = _stage("analyse_ladder_divergence")
    fixture = _fixture(tmp_path, planted_divergence_rung="k5")

    structure_out = tmp_path / "structure"
    compare.run(
        type(
            "Args",
            (),
            {
                "out": structure_out,
                "device": "cpu",
                "backbones": fixture["backbones"],
                "variants": [fixture["variants"]],
                "structure": [fixture["variant_root"]],
                "parent_structure": [fixture["parent_root"]],
            },
        )()
    )
    structure_rows = design.read_jsonl(structure_out / "ladder_structure.jsonl")
    assert len(structure_rows) == N_BACKBONES * len(RUNGS) * DRAWS
    assert all(row["fold_status"] == "ok" for row in structure_rows)
    assert all(design.PRIMARY_STRUCTURE_READOUT in row for row in structure_rows)
    # Length is reported beside every confidence number.
    assert all(row["length"] == LENGTH for row in structure_rows)

    analysis_out = tmp_path / "analysis"
    payload = analyse.run(
        type(
            "Args",
            (),
            {
                "out": analysis_out,
                "device": "cpu",
                "backbones": fixture["backbones"],
                "variants": [fixture["variants"]],
                "likelihood": [fixture["likelihood"]],
                "structure": [structure_out / "ladder_structure.jsonl"],
                "draws": 400,
                "seed": 7,
            },
        )()
    )
    key = f"{ARM}|{ARM}|{design.CONDITION_CAUSAL_PREFIX}"
    ladder = payload["ladders"][key]
    assert ladder["status"] == "complete"
    points = {row["rung"]: row["point"] for row in ladder["ladder"]}
    assert points["k1"] > 0.8 and points["k2"] > 0.8
    assert abs(points["k5"]) < 0.6
    assert ladder["divergence"]["loses_significance"] == "k5"
    assert ladder["divergence"]["undetermined_at_bottom"] is False
    # Every cell carries its own sample size and backbone count.
    for row in ladder["ladder"]:
        cell = payload["cells"][f"{key}|{row['rung']}"]
        primary = cell["correlations"][design.PRIMARY_STRUCTURE_READOUT]["raw"]
        assert primary["n_observations"] == N_BACKBONES * DRAWS
        assert primary["n_units"] == N_BACKBONES
        assert cell["distributions"]["length"]["mean"] == LENGTH
    assert (analysis_out / "ladder_divergence.md").is_file()
    report = (analysis_out / "ladder_divergence.md").read_text(encoding="utf-8")
    assert "rho(likelihood, TM to parent)" in report
    assert "confidence_is_not_stability" in report


def test_an_empty_join_is_refused_rather_than_reported_as_a_null_result(tmp_path):
    analyse = _stage("analyse_ladder_divergence")
    fixture = _fixture(tmp_path)
    design.write_jsonl(
        tmp_path / "no_folds.jsonl",
        [
            {
                "id": row["id"],
                "arm": row["arm"],
                "condition": row["condition"],
                "rung": row["rung"],
                "backbone_id": row["backbone_id"],
                "draw": row["draw"],
                "fold_status": "missing",
            }
            for row in design.read_jsonl(fixture["variants"])
        ],
    )
    with pytest.raises(SystemExit, match="nothing joined"):
        analyse.run(
            type(
                "Args",
                (),
                {
                    "out": tmp_path / "empty",
                    "device": "cpu",
                    "backbones": fixture["backbones"],
                    "variants": [fixture["variants"]],
                    "likelihood": [fixture["likelihood"]],
                    "structure": [tmp_path / "no_folds.jsonl"],
                    "draws": 50,
                    "seed": 1,
                },
            )()
        )
