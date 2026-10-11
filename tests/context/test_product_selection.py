"""E11's product pool and the selection-novelty reading built on E17's selector.

The property these tests defend is that nothing about the selector is
reimplemented here. E17's declared selector, its selection fractions, its
size-matched random baseline and its near-duplicate threshold are imported, and a
drift in any of them must fail here rather than quietly produce a second method
whose difference from the first is uninterpretable.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.capability.context import homology_context as H  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402

FREEZE = ROOT / "scripts/capability/context/freeze_product_pool.py"
SELECT = ROOT / "scripts/capability/context/select_product_novelty.py"


def attempt(index: int, condition: str, *, target: str, residues: int = 200, lcs: int = 3):
    return {
        "attempt_id": f"progen2-medium|{target}|{condition}|{index:04d}",
        "arm": "progen2-medium",
        "target_id": target,
        "condition": condition,
        "residues": residues,
        "stop_status": "native_terminal",
        "context_items": 0 if condition == H.NO_CONTEXT else 4,
        "context_subjects": [] if condition == H.NO_CONTEXT else ["UniRef90_A"],
        "sequence": "".join("ACDEFGHIKLMNPQRSTVWY"[(index + pos) % 20] for pos in range(residues)),
        "copy_statistics": {
            "max_lcs_to_context": lcs,
            "lcs_fraction_of_product": lcs / residues,
            "max_kmer_containment": 0.0,
        },
    }


def products(n_per_cell: int = 6, targets: int = 10) -> list[dict]:
    rows = []
    for group in range(targets):
        for condition in H.GENERATION_CONDITIONS:
            for index in range(n_per_cell):
                rows.append(attempt(index, condition, target=f"q{group:02d}"))
    return rows


def write_products(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def annotation_file(path: Path, rows: list[dict], *, identity_of) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": H.SCHEMA_VERSION,
                "stage": "annotate",
                "status": "complete",
                **H.declaration_digests(),
                "corpus_search": {"command": ["diamond", "--masking", "0"], "masking": 0},
                "corpus_database": {"path": "uniref90_full.dmnd"},
                "annotations": [
                    {
                        "attempt_id": row["attempt_id"],
                        "condition": row["condition"],
                        "target_id": row["target_id"],
                        "residues": row["residues"],
                        "context_alignment_identity": None,
                        "corpus_max_identity": identity_of(position, row),
                        "corpus_hits": 0 if identity_of(position, row) is None else 1,
                    }
                    for position, row in enumerate(rows)
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def run_stage(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


def test_the_pool_is_the_novelty_censuss_own_denominator(tmp_path):
    rows = products()
    rows.append(attempt(99, H.CLOSE_HOMOLOG, target="q00", residues=H.MIN_PRODUCT_RESIDUES - 1))
    path = write_products(tmp_path / "products.jsonl", rows)
    done = run_stage(FREEZE, "--products", str(path), "--out", str(tmp_path / "pool"))
    assert done.returncode == 0, done.stderr
    record = json.loads((tmp_path / "pool/product_pool.json").read_text())
    assert record["members"] == len(rows) - 1
    assert record["excluded"] == {"short": 1, "copy": 0}
    assert record["arm"] == "progen2-medium"
    members = [
        json.loads(line)
        for line in (tmp_path / "pool/product_pool.jsonl").read_text().splitlines()
    ]
    # The row shape E17's scorer consumes, unchanged.
    assert all({"id", "arm", "roles", "sequence", "length"} <= set(row) for row in members)
    assert all(row["roles"] == ["pool"] for row in members)
    assert all(row["length"] == len(row["sequence"]) for row in members)


def test_a_pool_of_two_arms_is_refused_because_likelihoods_are_within_arm(tmp_path):
    rows = products()
    rows[0] = {**rows[0], "arm": "protgpt2"}
    path = write_products(tmp_path / "products.jsonl", rows)
    done = run_stage(FREEZE, "--products", str(path), "--out", str(tmp_path / "pool"))
    assert done.returncode != 0
    assert "comparable within an" in done.stderr


def test_a_copy_is_kept_out_of_the_pool_when_the_rule_can_be_evaluated(tmp_path):
    rows = products()
    rows[0]["copy_statistics"]["max_lcs_to_context"] = H.COPY_LCS_RESIDUES + 1
    path = write_products(tmp_path / "products.jsonl", rows)
    done = run_stage(FREEZE, "--products", str(path), "--out", str(tmp_path / "pool"))
    assert done.returncode == 0, done.stderr
    record = json.loads((tmp_path / "pool/product_pool.json").read_text())
    assert record["excluded"]["copy"] == 1


def build_selection_inputs(tmp_path, *, identity_of, score_of):
    rows = products()
    path = write_products(tmp_path / "products.jsonl", rows)
    assert run_stage(FREEZE, "--products", str(path), "--out", str(tmp_path / "pool")).returncode == 0
    members = [
        json.loads(line)
        for line in (tmp_path / "pool/product_pool.jsonl").read_text().splitlines()
    ]
    likelihood_dir = tmp_path / "likelihood"
    likelihood_dir.mkdir()
    (likelihood_dir / "generated_likelihood.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "id": row["id"],
                    "arm": row["arm"],
                    "length": row["length"],
                    "scored_tokens": row["length"],
                    "nll_sum_nats": score_of(row) * row["length"],
                    "mean_nll_per_token_nats": score_of(row),
                    "mean_nll_per_residue_nats": score_of(row),
                }
            )
            + "\n"
            for row in members
        ),
        encoding="utf-8",
    )
    (likelihood_dir / "generated_likelihood.json").write_text(
        json.dumps({"status": "complete", "arm": "progen2-medium"}), encoding="utf-8"
    )
    annotations = annotation_file(tmp_path / "annotations.json", rows, identity_of=identity_of)
    return tmp_path / "pool/product_pool.json", likelihood_dir / "generated_likelihood.json", annotations


def test_selection_that_concentrates_on_near_duplicates_is_measured_against_a_matched_draw(tmp_path):
    # The most likely products (lowest NLL) are exactly the near-duplicates, so a
    # likelihood selector must find them and beat a size-matched random draw.
    def identity_of(position, row):
        return 99.0 if position % 10 == 0 else 35.0

    def score_of(row):
        return 0.1 if row["id"].endswith(("0000",)) else 5.0

    pool, likelihood, annotations = build_selection_inputs(
        tmp_path, identity_of=identity_of, score_of=score_of
    )
    done = run_stage(
        SELECT, "--pool", str(pool), "--likelihood", str(likelihood),
        "--annotations", str(annotations), "--out", str(tmp_path / "sel"),
        "--resamples", "400",
    )
    assert done.returncode == 0, done.stderr
    record = json.loads((tmp_path / "sel/product_selection_novelty.json").read_text())
    assert record["near_duplicate_identity_percent"] == gp.NEAR_DUPLICATE_IDENTITY
    assert record["selection_fractions"] == list(gp.SELECTION_FRACTIONS)
    assert record["random_baseline_keys"] == gp.RANDOM_BASELINE_KEYS
    cell = record["per_condition"][H.CLOSE_HOMOLOG]
    assert cell["status"] == "estimated"
    top = cell["fractions"]["top_0.05"]
    # The comparator is a size-matched random draw with its own interval, and the
    # pool mean is reported beside it rather than in place of it.
    assert top["random_keys"] == gp.RANDOM_BASELINE_KEYS
    assert len(top["near_duplicate_share_random_interval"]) == 2
    assert top["near_duplicate_share_pool_mean"] == pytest.approx(cell["near_duplicate_share_pool"])
    assert top["near_duplicate_share_selected"] > top["near_duplicate_share_random"]
    assert top["enrichment_vs_random"] is None or top["enrichment_vs_random"] > 1.0
    assert top["n_selected"] == gp.selection_count(cell["members"], 0.05)


def test_a_pool_with_no_near_duplicate_reports_a_floor_not_an_enrichment(tmp_path):
    pool, likelihood, annotations = build_selection_inputs(
        tmp_path, identity_of=lambda position, row: 40.0, score_of=lambda row: float(len(row["id"]))
    )
    done = run_stage(
        SELECT, "--pool", str(pool), "--likelihood", str(likelihood),
        "--annotations", str(annotations), "--out", str(tmp_path / "sel"),
        "--resamples", "400",
    )
    assert done.returncode == 0, done.stderr
    record = json.loads((tmp_path / "sel/product_selection_novelty.json").read_text())
    cell = record["per_condition"][H.UNRELATED]
    assert cell["near_duplicates_in_pool"] == 0
    top = cell["fractions"]["top_0.05"]
    assert top["near_duplicate_share_selected"] == 0.0
    # No random draw contains one either, so the ratio is undefined and says so
    # rather than dividing by zero or reporting an infinite enrichment.
    assert top["enrichment_vs_random"] is None
    assert "floor on the pool" in top["enrichment_note"]


def test_a_product_with_no_detectable_relative_counts_as_zero_identity(tmp_path):
    pool, likelihood, annotations = build_selection_inputs(
        tmp_path,
        identity_of=lambda position, row: None if position % 2 else 96.0,
        score_of=lambda row: float(len(row["id"])),
    )
    done = run_stage(
        SELECT, "--pool", str(pool), "--likelihood", str(likelihood),
        "--annotations", str(annotations), "--out", str(tmp_path / "sel"),
        "--resamples", "400",
    )
    assert done.returncode == 0, done.stderr
    record = json.loads((tmp_path / "sel/product_selection_novelty.json").read_text())
    cell = record["per_condition"]["all_conditions"]
    assert cell["with_detectable_homology"] == cell["members"] // 2
    assert cell["near_duplicate_share_pool"] == pytest.approx(0.5, abs=0.02)


def test_the_convention_check_reports_all_three_readings(tmp_path):
    pool, likelihood, annotations = build_selection_inputs(
        tmp_path, identity_of=lambda position, row: 40.0, score_of=lambda row: float(len(row["id"]))
    )
    done = run_stage(
        SELECT, "--pool", str(pool), "--likelihood", str(likelihood),
        "--annotations", str(annotations), "--out", str(tmp_path / "sel"),
        "--resamples", "400",
    )
    assert done.returncode == 0, done.stderr
    record = json.loads((tmp_path / "sel/product_selection_novelty.json").read_text())
    convention = record["per_condition"]["all_conditions"]["selector_convention"]
    assert convention["declared"].startswith("mean_nll_per_token_nats")
    # The fixture scores per token and per residue identically, which is what a
    # residue tokenisation does; the check must say so rather than assume it.
    assert convention["per_token_equals_per_residue"] is True
    assert convention["per_token_vs_per_residue_spearman"] == pytest.approx(1.0)
    for key in ("per_token_vs_length_spearman", "summed_vs_length_spearman"):
        assert key in convention


def test_a_pool_member_without_a_likelihood_stops_the_reading(tmp_path):
    pool, likelihood, annotations = build_selection_inputs(
        tmp_path, identity_of=lambda position, row: 40.0, score_of=lambda row: 1.0
    )
    sidecar = likelihood.parent / "generated_likelihood.jsonl"
    lines = sidecar.read_text().splitlines()
    sidecar.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    done = run_stage(
        SELECT, "--pool", str(pool), "--likelihood", str(likelihood),
        "--annotations", str(annotations), "--out", str(tmp_path / "sel"),
    )
    assert done.returncode != 0
    assert "no likelihood" in done.stderr


def test_an_annotation_without_a_corpus_search_is_refused(tmp_path):
    pool, likelihood, annotations = build_selection_inputs(
        tmp_path, identity_of=lambda position, row: 40.0, score_of=lambda row: 1.0
    )
    record = json.loads(annotations.read_text())
    record["corpus_search"] = None
    annotations.write_text(json.dumps(record), encoding="utf-8")
    done = run_stage(
        SELECT, "--pool", str(pool), "--likelihood", str(likelihood),
        "--annotations", str(annotations), "--out", str(tmp_path / "sel"),
    )
    assert done.returncode != 0
    assert "no corpus search" in done.stderr
