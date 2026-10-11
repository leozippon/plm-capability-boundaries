"""Conditions the E17 likelihood-ranking diagnostics must always satisfy.

The cases here are the ones that decide whether a ranking result means anything:
that a spurious association produced by a known confounder disappears once the
confounder is in the control design, that a genuine negative tail association is
found while a whole-pool statistic can miss it, that likelihood is oriented so a
lower negative log-likelihood is a higher likelihood, that a cluster bootstrap
retains a repeated unit in full instead of deduplicating it, that a constant,
rank-deficient or saturated support is reported unavailable rather than filled,
and that a duplicated identifier, a mismatched pool digest or a missing novelty
row stops the run instead of producing a partial artefact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from scripts.capability.evaluation import analyse_likelihood_ranking as alr
from src.capability.evaluation import generated_phenotype as gp
from src.capability.evaluation import likelihood_ranking as lr


# ------------------------------------------------------------------ unit checks


def test_known_confounding_is_removed_by_the_declared_control():
    rng = np.random.default_rng(11)
    n = 400
    confounder = rng.normal(size=n)
    likelihood = confounder + rng.normal(scale=0.5, size=n)
    endpoint = confounder + rng.normal(scale=0.5, size=n)

    raw = lr.whole_pool_spearman(likelihood, endpoint)
    assert raw["spearman"] < -0.6

    adjusted = lr.partial_spearman(
        likelihood, endpoint, np.column_stack([np.ones(n), confounder])
    )
    assert adjusted["status"] == "measured"
    assert abs(adjusted["spearman"]) < 0.15

    # With only an intercept the association is the unadjusted one, so it is the
    # confounder column, not residualization in general, that removes it.
    still = lr.partial_spearman(likelihood, endpoint, np.ones((n, 1)))
    assert abs(still["spearman"]) > 0.6


def test_a_genuine_negative_tail_is_found_even_when_the_pool_is_flat():
    rng = np.random.default_rng(12)
    n = 300
    likelihood = rng.normal(size=n)
    endpoint = rng.normal(size=n)
    tail = np.argsort(likelihood, kind="stable")[:30]
    endpoint[tail] = 3.0 * likelihood[tail]

    units = [f"u{index}" for index in range(n)]
    whole = lr.whole_pool_spearman(likelihood, endpoint)
    tail_result = lr.tail_spearman(likelihood, endpoint, units, fraction=0.10)

    assert tail_result["n_selected"] == 30
    assert tail_result["status"] == "measured"
    assert tail_result["spearman"] < -0.9
    assert abs(whole["spearman"]) < 0.4


def test_lower_likelihood_is_treated_as_more_likely():
    rng = np.random.default_rng(13)
    n = 200
    likelihood = rng.normal(3.5, 0.5, size=n)
    endpoint = 80.0 - 8.0 * likelihood + rng.normal(scale=3.0, size=n)

    diagnostics = lr.selection_diagnostics(
        likelihood, endpoint, np.ones((n, 1)), fraction=0.10, rung_name="intercept"
    )
    assert diagnostics["raw"]["selected_mean"] > diagnostics["raw"]["pool_expected_random"]
    assert lr.whole_pool_spearman(likelihood, endpoint)["spearman"] > 0.0

    inverted = lr.selection_diagnostics(
        -likelihood, endpoint, np.ones((n, 1)), fraction=0.10, rung_name="intercept"
    )
    assert inverted["raw"]["selected_mean"] < inverted["raw"]["pool_expected_random"]


def test_cluster_resampling_preserves_repeated_units_in_full():
    assert lr.resample_indices_for_chosen(["a", "a", "b"], ["a", "a", "b"]).tolist() == [
        0,
        1,
        0,
        1,
        2,
    ]
    assert lr.resample_indices_for_chosen(["a", "a", "b"], ["b", "b"]).tolist() == [2, 2]
    with pytest.raises(ValueError):
        lr.resample_indices_for_chosen(["a"], ["unknown"])


def test_constant_deficient_and_saturated_supports_are_unavailable():
    values = np.arange(10.0)
    constant = lr.whole_pool_spearman(values, np.ones(10))
    assert constant["status"] == "unavailable"
    assert constant["reason"] == "endpoint_constant"
    assert constant["spearman"] is None

    duplicate = np.column_stack([np.ones(10), values, values])
    assert lr.partial_spearman(values, values, duplicate)["reason"] == "rank_deficient"

    n = 5
    dummies = np.zeros((n, n - 1))
    dummies[: n - 1] = np.eye(n - 1)
    saturated = lr.partial_spearman(
        np.arange(n, dtype=float), np.arange(n, dtype=float), np.column_stack([np.ones(n), dummies])
    )
    assert saturated["status"] == "unavailable"
    assert saturated["reason"] == "saturated"
    assert saturated["residual_df"] == 0


def test_a_tail_below_the_unit_floor_is_flagged_not_withheld():
    rng = np.random.default_rng(14)
    n = 40
    likelihood = rng.normal(size=n)
    endpoint = -likelihood + rng.normal(scale=0.1, size=n)
    units = [f"c{index % 3}" for index in range(n)]

    result = lr.tail_spearman(likelihood, endpoint, units, fraction=0.50)
    assert result["n_selected"] == 20
    assert result["n_units_in_tail"] == 3
    assert result["status"] == "measured_below_unit_floor"
    assert result["spearman"] is not None

    withheld = lr.tail_spearman(likelihood, np.ones(n), units, fraction=0.50)
    assert withheld["status"] == "unavailable"
    assert withheld["reason"] == "endpoint_constant"
    assert withheld["spearman"] is None


def test_the_tail_bootstrap_reselects_the_tail_and_respects_the_floor():
    rng = np.random.default_rng(15)
    n = 240
    likelihood = rng.normal(size=n)
    endpoint = -2.0 * likelihood + rng.normal(scale=1.0, size=n)
    units = [f"c{index}" for index in range(n)]

    point = lr.tail_spearman(likelihood, endpoint, units, fraction=1.0)
    bootstrap = lr.cluster_bootstrap_tail_spearman(
        likelihood, endpoint, units, fraction=1.0, seed=5, n_bootstrap=200
    )
    assert point["status"] == "measured"
    assert bootstrap["status"] == "measured"
    assert bootstrap["spearman_ci95"][0] <= bootstrap["spearman_ci95"][1]
    assert bootstrap["valid_replicate_fraction"] is not None

    floor = lr.cluster_bootstrap_tail_spearman(
        likelihood,
        endpoint,
        [f"c{index % 3}" for index in range(n)],
        fraction=0.50,
        seed=5,
        n_bootstrap=200,
    )
    assert floor["status"] == "unavailable"
    assert floor["reason"] == "below_unit_floor"


def test_the_family_control_needs_variation_and_a_canonical_sequence():
    assert not lr.has_family_variation(("__none__",))
    assert lr.has_family_variation(("__none__", "PF00001"))
    assert lr.has_family_variation(("PF00001", "PF00002"))

    features = [
        lr.make_row_features(
            "ACDEFGHIKLMNPQRSTVWY", 20, nearest_corpus_identity=1.0, families=[]
        )
        for _ in range(3)
    ]
    assert lr.family_vocabulary(features) == ("__none__",)
    with pytest.raises(ValueError):
        lr.make_row_features(
            "ACDEFGHIKLMNPQRSTVWX", 21, nearest_corpus_identity=1.0, families=[]
        )


def test_bootstrap_withholds_an_interval_below_the_unit_floor():
    values = np.arange(20.0)
    design = np.column_stack([np.ones(20), values])
    result = lr.cluster_bootstrap_association(
        values, values, design, [f"c{index % 5}" for index in range(20)], seed=1, n_bootstrap=200
    )
    assert result["status"] == "unavailable"
    assert result["reason"] == "below_unit_floor"
    assert result["adjusted_ci95"] is None


def test_declared_columns_and_the_dropped_residue():
    composition = lr.rung_column_names(lr.RUNG_COMPOSITION, ())
    assert composition[:2] == ("intercept", "length_scaled")
    assert f"composition_{lr.DROPPED_RESIDUE}" not in composition
    assert sum(name.startswith("composition_") for name in composition) == 19
    assert "entropy_bits" in composition and "homopolymer_fraction" in composition

    full = lr.rung_column_names(lr.RUNG_FULL, ("__none__", "PF00001", "PF00002"))
    assert full[-2:] == ("family_PF00001", "family_PF00002")
    assert lr.rung_added_columns(lr.RUNG_FULL, ("__none__", "PF00001")) == (
        "nearest_corpus_identity_scaled",
        "family_PF00001",
    )


def test_partial_spearman_refuses_misaligned_or_non_finite_input():
    with pytest.raises(ValueError):
        lr.partial_spearman([1.0, 2.0], [1.0], np.ones((2, 1)))
    with pytest.raises(ValueError):
        lr.whole_pool_spearman([1.0, float("nan")], [1.0, 2.0])


def test_tail_interval_requires_point_tail_and_original_draw_unit_support():
    x = np.arange(100.0)
    units = np.repeat(np.arange(10), 10)
    point = lr.tail_spearman(x, -x, units, fraction=0.1)
    assert point["n_units_in_tail"] == 1
    withheld = lr.cluster_bootstrap_tail_spearman(
        x, -x, units, fraction=0.1, seed=5, n_bootstrap=400
    )
    assert withheld["reason"] == "point_tail_below_unit_floor"
    assert withheld["spearman_ci95"] is None

    # Full tail has ten units in the point pool, but repeated cluster copies
    # must not inflate a draw's original-unit support. No retry or deduplication.
    draws = lr.cluster_bootstrap_tail_spearman(
        x, -x, units, fraction=1.0, seed=5, n_bootstrap=400
    )
    rng = np.random.default_rng(5)
    expected_invalid = sum(len(set(rng.choice(10, size=10))) < 8 for _ in range(400))
    assert draws["invalid_reasons"]["tail_below_unit_floor"] == expected_invalid
    assert draws["n_valid_draws"] + draws["n_invalid_draws"] == 400
    assert draws["reason"] == "valid_replicate_fraction_below_floor"
    assert draws["spearman_ci95"] is None


def test_observed_family_reference_avoids_dummy_trap_and_stays_fixed():
    rng = np.random.default_rng(19)
    features = []
    for i in range(120):
        sequence = "".join(rng.choice(list(lr.AA20), size=int(rng.integers(40, 200))))
        features.append(lr.make_row_features(
            sequence, len(sequence), nearest_corpus_identity=float(rng.uniform(0, 90)),
            families=["PF00002" if i % 2 else "PF00001"],
        ))
    vocabulary = lr.family_vocabulary(features)
    assert lr.family_reference(vocabulary) == "PF00001"
    design, names = lr.build_design(features, lr.RUNG_FULL, vocabulary)
    assert "family_PF00001" not in names and "family_PF00002" in names
    assert lr.design_diagnostics(design)["rank_deficiency"] == 0
    missing_category, same_names = lr.build_design(features[::2], lr.RUNG_FULL, vocabulary)
    assert same_names == names
    assert lr.design_diagnostics(missing_category)["rank_deficiency"] > 0


def test_adjusted_selected_tail_is_spearman_of_high_likelihood_residuals():
    from scipy import stats

    x = np.array([0., 1., 2., 4., 7., 11., 16., 22., 29., 37.])
    y = np.array([8., 2., 4., 1., 9., 6., 3., 7., 5., 0.])
    design = np.column_stack([np.ones(10), np.arange(10) % 3])
    rx, ry = lr.residualize(x, design), lr.residualize(y, design)
    selected = np.argsort(rx, kind="stable")[:8]
    expected = stats.spearmanr(-rx[selected], ry[selected])[0]
    pearson = np.corrcoef(-rx[selected], ry[selected])[0, 1]
    result = lr.selection_diagnostics(x, y, design, fraction=0.8, rung_name="test")
    assert result["adjusted"]["selected_tail_spearman"] == pytest.approx(expected)
    assert abs(expected - pearson) > 0.01
    assert lr.partial_spearman(x, y, np.ones((10, 1)))["spearman"] == pytest.approx(
        lr.whole_pool_spearman(x, y)["spearman"]
    )
    bootstrap = lr.cluster_bootstrap_association(
        x, -x, np.ones((10, 1)), list(range(10)), seed=5, n_bootstrap=400
    )
    assert bootstrap["adjusted_ci95"] == pytest.approx([1., 1.])
    assert bootstrap["unadjusted_ci95"] == pytest.approx([1., 1.])
    assert bootstrap["paired_difference_ci95"] == pytest.approx([0., 0.], abs=1e-12)


# --------------------------------------------------------------- end-to-end tree


@dataclass(frozen=True)
class Tree:
    root: Path
    cohort: Path
    fold: Path
    oracle: Path
    novelty: Path
    likelihood: dict[str, Path]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def refresh_annotations_hash(receipt_path: Path) -> None:
    receipt = json.loads(receipt_path.read_text())
    receipt["annotations_sha256"] = hashlib.sha256(
        receipt_path.with_suffix(".jsonl").read_bytes()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt))


def build_tree(root: Path, *, n_per_arm: int = 48, class_keys: int = 12) -> Tree:
    root.mkdir(parents=True, exist_ok=True)
    cohort_dir = root / "cohort"
    fold_dir = root / "fold"
    oracle_dir = root / "oracle"
    novelty_dir = root / "novelty"
    for directory in (cohort_dir, fold_dir, oracle_dir, novelty_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)
    alphabet = list(gp.AA20)
    arms = {
        "protgpt2": {"unit": "id", "condition": "unconditioned"},
        "zymctrl": {"unit": "class_key", "condition": "requested"},
    }
    cohort_rows: list[dict] = []
    structure_rows: list[dict] = []
    recognition_rows: list[dict] = []
    novelty_rows: list[dict] = []
    likelihood_rows: dict[str, list[dict]] = {arm: [] for arm in arms}
    for arm, meta in arms.items():
        for index in range(n_per_arm):
            identifier = f"ge_{arm[:3]}{index:022d}"
            length = int(rng.integers(60, 300))
            sequence = "".join(rng.choice(alphabet, size=length))
            class_key = f"1.{index % class_keys}.0.0" if meta["unit"] == "class_key" else None
            likelihood = float(rng.normal(3.4, 0.4))
            plddt = float(np.clip(82.0 - 9.0 * likelihood + rng.normal(0, 4.0), 5.0, 99.0))
            ptm = float(np.clip(0.75 - 0.12 * likelihood + rng.normal(0, 0.08), 0.05, 0.99))
            cohort_rows.append(
                {
                    "id": identifier,
                    "arm": arm,
                    "campaign": "replicate_1",
                    "condition": meta["condition"],
                    "class_key": class_key,
                    "length": length,
                    "sequence": sequence,
                    "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    "decoder_stop": "eos",
                    "exact_duplicate_group": f"{arm}|{identifier}",
                    "role": "pool",
                    "roles": ["pool"],
                    "stratum": "native",
                }
            )
            structure_rows.append(
                {
                    "id": identifier,
                    "arm": arm,
                    "sequence": sequence,
                    "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    "structure": {
                        "status": "ok", "mean_ca_plddt": plddt, "ptm": ptm,
                        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    },
                }
            )
            family = [None, "PF00001", "PF00002", "PF00003"][index % 4]
            recognition_rows.append(
                {
                    "id": identifier,
                    "arm": arm,
                    "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    "profile_search_status": "searched",
                    "pfam_families": [] if family is None else [family],
                    "complete_domain": family is not None,
                    "any_profile_hit": family is not None,
                }
            )
            novelty_rows.append(
                {"id": identifier, "arm": arm, "searched": True,
                 "nearest_corpus_identity": float(rng.uniform(0.0, 90.0))}
            )
            likelihood_rows[arm].append(
                {
                    "id": identifier,
                    "arm": arm,
                    "pool_arm": arm,
                    "scorer_arm": arm,
                    "mean_nll_per_token_nats": likelihood,
                    "mean_nll_per_residue_nats": likelihood / 2.0,
                    "nll_sum_nats": likelihood * length / 2.0,
                    "scored_tokens": length // 2,
                }
            )
    cohort_rows.sort(key=lambda row: row["id"])
    write_jsonl(cohort_dir / "evaluation_cohort.jsonl", cohort_rows)
    pool = [row for row in cohort_rows if "pool" in row["roles"]]
    pools = {}
    for arm, meta in arms.items():
        members = [row for row in pool if row["arm"] == arm]
        units = {
            str(row["id"]) if meta["unit"] == "id" else str(row["class_key"]) for row in members
        }
        pools[arm] = {
            "condition": meta["condition"],
            "independence_unit": meta["unit"],
            "n_pool": len(members),
            "n_units": len(units),
        }
    (cohort_dir / "e17_declaration.json").write_text(
        json.dumps(
            {
                "schema_version": "d1_generated_evaluation_cohort_v1",
                "pools": pools,
                "pool_digest": gp.cohort_digest(pool),
                "n_pool": len(pool),
                "selection_fractions": list(gp.SELECTION_FRACTIONS),
            },
            indent=1,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (cohort_dir / "residue_background.json").write_text(
        json.dumps({"schema_version": "synthetic", "source": "synthetic", "n_records": 10}),
        encoding="utf-8",
    )

    write_jsonl(fold_dir / "index-000-of-001.jsonl", structure_rows)
    (fold_dir / "structure_evidence.json").write_text(
        json.dumps({"status_counts": {"ok": len(structure_rows)}}), encoding="utf-8"
    )
    write_jsonl(oracle_dir / "family_recognition.jsonl", recognition_rows)
    (oracle_dir / "family_recognition.json").write_text(
        json.dumps({"status": "complete", "annotations_sha256": hashlib.sha256(
            (oracle_dir / "family_recognition.jsonl").read_bytes()).hexdigest()}), encoding="utf-8"
    )
    (oracle_dir / "oracle_control.json").write_text(
        json.dumps({"passed": True, "n_positive": 1, "n_negative": 1}), encoding="utf-8"
    )
    cohort_sha = hashlib.sha256((cohort_dir / "evaluation_cohort.jsonl").read_bytes()).hexdigest()
    (novelty_dir / "generated_novelty.json").write_text(
        json.dumps(
            {
                "schema_version": "d1_generated_novelty_v1",
                "status": "complete",
                "cohort_sha256": cohort_sha,
                "n_searched": len(novelty_rows),
                "n_with_any_hit": 10,
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(novelty_dir / "generated_novelty.jsonl", novelty_rows)
    refresh_annotations_hash(novelty_dir / "generated_novelty.json")

    likelihood_dirs: dict[str, Path] = {}
    for arm in arms:
        directory = root / f"lik_{arm}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "generated_likelihood.json").write_text(
            json.dumps({"arm": arm, "status": "complete", "cohort_sha256": cohort_sha,
                        "pool_digest": gp.cohort_digest([row for row in pool if row["arm"] == arm])}),
            encoding="utf-8"
        )
        write_jsonl(directory / "generated_likelihood.jsonl", likelihood_rows[arm])
        refresh_annotations_hash(directory / "generated_likelihood.json")
        likelihood_dirs[arm] = directory

    return Tree(
        root=root,
        cohort=cohort_dir,
        fold=fold_dir,
        oracle=oracle_dir,
        novelty=novelty_dir,
        likelihood=likelihood_dirs,
    )


def make_args(tree: Tree, out: Path, *, with_novelty: bool = True, draws: int = 200) -> argparse.Namespace:
    return argparse.Namespace(
        cohort_dir=tree.cohort,
        structure=[tree.fold],
        recognition=tree.oracle,
        likelihood=[tree.likelihood["protgpt2"], tree.likelihood["zymctrl"]],
        novelty=tree.novelty if with_novelty else None,
        draws=draws,
        seed=3,
        device="cpu",
        out=out,
    )


def test_the_stage_emits_a_strict_complete_record_with_support_counts(tmp_path: Path):
    tree = build_tree(tmp_path / "tree")
    out = tmp_path / "out"
    record = alr.run(make_args(tree, out))

    assert record["status"] == "complete"
    artifact = out / alr.COMPLETION
    assert artifact.is_file()
    # write_json refuses NaN; re-serialising the loaded record proves none survived.
    json.dumps(record, allow_nan=False)
    assert sorted(record["arms"]) == ["protgpt2", "zymctrl"]
    assert record["endpoints"] == list(lr.ENDPOINTS)
    assert any(entry["role"] == "cohort_jsonl" for entry in record["source_hashes"])
    assert record["cohort"]["pool_digest"] == record["cohort"]["declaration_pool_digest"]

    for arm in record["arms"].values():
        assert arm["recognition_present"] == arm["n_pool"]
        assert arm["novelty_present"] == arm["n_pool"]
        for endpoint in lr.ENDPOINTS:
            block = arm["endpoints"][endpoint]
            assert block["whole_pool_spearman"]["status"] == "measured"
            assert set(block["tail_spearman"]) == {
                f"{fraction:.2f}" for fraction in lr.SELECTION_FRACTIONS
            }
            full_tail = block["tail_spearman"]["1.00"]
            assert full_tail["n_selected"] == block["n_evaluable"]
            assert full_tail["bootstrap"]["status"] in {"measured", "unavailable"}
            if full_tail["bootstrap"]["status"] == "unavailable":
                assert full_tail["bootstrap"]["reason"] == "valid_replicate_fraction_below_floor"
            full = block["control_ladder"][lr.RUNG_FULL.name]
            assert full["status"] in {"measured", "unavailable"}
            assert full["n_columns"] == len(full["column_names"])
            if full["status"] == "measured":
                partial = full["partial_spearman"]
                assert partial["rank"] <= partial["n_columns"]
                assert partial["residual_df"] == partial["n_rows"] - partial["rank"]
                assert full["bootstrap"]["valid_replicate_fraction"] is not None
            selection = block["selection"]["0.10"]
            assert "oracle_bound" in selection["raw"]
            assert "pool_expected_random" in selection["raw"]


def test_a_novelty_free_run_reports_the_full_rung_unavailable_not_complete(tmp_path: Path):
    tree = build_tree(tmp_path / "tree")
    record = alr.run(make_args(tree, tmp_path / "out", with_novelty=False))
    assert record["status"] == "complete"
    for arm in record["arms"].values():
        assert arm["novelty_present"] is None
        for endpoint in lr.ENDPOINTS:
            full = arm["endpoints"][endpoint]["control_ladder"][lr.RUNG_FULL.name]
            assert full["status"] == "unavailable"
            assert full["reason"] == "novelty_output_not_supplied"
            assert full["n_support"] == 0
            assert full["excluded_reasons"]["novelty_output_not_supplied"] == arm["endpoints"][
                endpoint
            ]["n_evaluable"]
            assert arm["endpoints"][endpoint]["selection"]["0.10"]["adjusted"][
                lr.RUNG_FULL.name
            ]["status"] == "unavailable"


def test_a_duplicate_likelihood_identifier_stops_the_run(tmp_path: Path):
    tree = build_tree(tmp_path / "tree")
    sidecar = tree.likelihood["protgpt2"] / "generated_likelihood.jsonl"
    sidecar.write_text(sidecar.read_text(encoding="utf-8") + sidecar.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        alr.run(make_args(tree, out))
    assert not (out / alr.COMPLETION).exists()


def test_a_duplicate_structure_identifier_across_shards_stops_the_run(tmp_path: Path):
    tree = build_tree(tmp_path / "tree")
    first = json.loads((tree.fold / "index-000-of-001.jsonl").read_text().splitlines()[0])
    write_jsonl(tree.fold / "index-001-of-002.jsonl", [first])
    with pytest.raises(SystemExit):
        alr.run(make_args(tree, tmp_path / "out"))


def test_a_pool_digest_mismatch_stops_the_run(tmp_path: Path):
    tree = build_tree(tmp_path / "tree")
    declaration = json.loads((tree.cohort / "e17_declaration.json").read_text(encoding="utf-8"))
    declaration["pool_digest"] = "0" * 64
    (tree.cohort / "e17_declaration.json").write_text(json.dumps(declaration), encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        alr.run(make_args(tree, out))
    assert not (out / alr.COMPLETION).exists()


def test_a_missing_novelty_row_stops_the_run(tmp_path: Path):
    tree = build_tree(tmp_path / "tree")
    sidecar = tree.novelty / "generated_novelty.jsonl"
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    sidecar.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        alr.run(make_args(tree, out))
    assert not (out / alr.COMPLETION).exists()


@pytest.mark.parametrize("producer,field,value", [
    ("likelihood", "status", "failed"),
    ("likelihood", "arm", "wrong_arm"),
    ("likelihood", "scorer_arm", "wrong_arm"),
    ("likelihood", "cohort_sha256", "0" * 64),
    ("likelihood", "pool_digest", "0" * 64),
    ("novelty", "status", "failed"),
    ("novelty", "cohort_sha256", "0" * 64),
    ("recognition", "status", "failed"),
])
def test_failed_or_mismatched_producer_receipts_stop_before_write(tmp_path, producer, field, value):
    tree = build_tree(tmp_path / "tree")
    path = {
        "likelihood": tree.likelihood["protgpt2"] / "generated_likelihood.json",
        "novelty": tree.novelty / "generated_novelty.json",
        "recognition": tree.oracle / "family_recognition.json",
    }[producer]
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(SystemExit):
        alr.run(make_args(tree, tmp_path / "out"))
    assert not (tmp_path / "out" / alr.COMPLETION).exists()


@pytest.mark.parametrize("producer", ["likelihood", "novelty", "structure", "recognition"])
def test_missing_producer_binding_is_explicitly_blocked(tmp_path, producer):
    tree = build_tree(tmp_path / "tree")
    if producer in {"likelihood", "novelty"}:
        path = (tree.likelihood["protgpt2"] / "generated_likelihood.json" if producer == "likelihood"
                else tree.novelty / "generated_novelty.json")
        receipt = json.loads(path.read_text())
        receipt.pop("cohort_sha256", None)
        receipt.pop("pool_digest", None)
        path.write_text(json.dumps(receipt))
    else:
        path = (tree.fold / "index-000-of-001.jsonl" if producer == "structure"
                else tree.oracle / "family_recognition.jsonl")
        rows = alr.read_jsonl(path)
        for row in rows:
            row.pop("sequence", None)
            row.pop("sequence_sha256", None)
            if "structure" in row:
                row["structure"].pop("sequence_sha256", None)
        write_jsonl(path, rows)
        if producer == "recognition":
            refresh_annotations_hash(tree.oracle / "family_recognition.json")
    with pytest.raises(SystemExit, match="blocked input"):
        alr.run(make_args(tree, tmp_path / "out"))


@pytest.mark.parametrize("producer", ["structure", "recognition"])
def test_same_id_wrong_sequence_is_not_a_join(tmp_path, producer):
    tree = build_tree(tmp_path / "tree")
    path = (tree.fold / "index-000-of-001.jsonl" if producer == "structure"
            else tree.oracle / "family_recognition.jsonl")
    rows = alr.read_jsonl(path)
    rows[0]["sequence_sha256"] = "0" * 64
    write_jsonl(path, rows)
    if producer == "recognition":
        refresh_annotations_hash(tree.oracle / "family_recognition.json")
    with pytest.raises(SystemExit, match="sequence binding mismatch"):
        alr.run(make_args(tree, tmp_path / "out"))


@pytest.mark.parametrize("producer", ["structure", "recognition", "likelihood", "novelty"])
def test_wrong_arm_row_is_rejected_even_with_a_matching_receipt(tmp_path, producer):
    tree = build_tree(tmp_path / "tree")
    path = {
        "structure": tree.fold / "index-000-of-001.jsonl",
        "recognition": tree.oracle / "family_recognition.jsonl",
        "likelihood": tree.likelihood["protgpt2"] / "generated_likelihood.jsonl",
        "novelty": tree.novelty / "generated_novelty.jsonl",
    }[producer]
    rows = alr.read_jsonl(path)
    rows[0]["arm"] = "wrong_arm"
    write_jsonl(path, rows)
    if producer != "structure":
        refresh_annotations_hash(path.with_suffix(".json"))
    with pytest.raises(SystemExit, match="arm mismatch"):
        alr.run(make_args(tree, tmp_path / "out"))


def test_extra_novelty_records_are_legal_and_empty_endpoint_does_not_abort(tmp_path):
    tree = build_tree(tmp_path / "tree")
    path = tree.novelty / "generated_novelty.jsonl"
    rows = alr.read_jsonl(path)
    rows.append({"id": "extra_reference", "nearest_corpus_identity": 20.})
    write_jsonl(path, rows)
    refresh_annotations_hash(tree.novelty / "generated_novelty.json")
    path = tree.fold / "index-000-of-001.jsonl"
    rows = alr.read_jsonl(path)
    for row in rows:
        row["structure"].pop("ptm")
    write_jsonl(path, rows)
    result = alr.run(make_args(tree, tmp_path / "out"))
    for arm in result["arms"].values():
        empty = arm["endpoints"][lr.SECONDARY_ENDPOINT]
        assert empty["status"] == "unavailable" and empty["reason"] == "no_support"
        assert empty["n_evaluable"] == 0 and empty["n_dropped"] == arm["n_pool"]
        assert empty["dropped_reasons"] == {"ptm_absent": arm["n_pool"]}
        assert arm["endpoints"][lr.PRIMARY_ENDPOINT]["whole_pool_spearman"]["status"] == "measured"
    json.dumps(result, allow_nan=False)


def test_a_cohort_sequence_digest_mismatch_stops_the_run(tmp_path: Path):
    tree = build_tree(tmp_path / "tree")
    rows = [
        json.loads(line)
        for line in (tree.cohort / "evaluation_cohort.jsonl").read_text().splitlines()
    ]
    rows[0]["sequence_sha256"] = "f" * 64
    write_jsonl(tree.cohort / "evaluation_cohort.jsonl", rows)
    with pytest.raises(SystemExit):
        alr.run(make_args(tree, tmp_path / "out"))
