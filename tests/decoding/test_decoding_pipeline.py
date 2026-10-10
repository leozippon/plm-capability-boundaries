"""The sweep's two analysis stages end to end, on a synthetic sweep.

These exercise the joins that would otherwise only be discovered after the GPU
hours are spent: a sweep that silently lost a cell, an in-band product that was
never folded, folds that came from two different measurement contracts, and the
assembled result table's own shape.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.capability.decoding import decoding_sweep as ds

ROOT = Path(__file__).resolve().parents[2]
ARM = "protgpt2"


def _stage(name: str):
    path = ROOT / "scripts/capability/decoding" / name
    spec = importlib.util.spec_from_file_location(f"_decoding_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sequence(rng: np.random.Generator, length: int) -> str:
    return "M" + "".join(rng.choice(list(ds.gp.AA20), size=max(0, length - 1)))


def _sweep_ledger(seed: int = 5) -> list[dict]:
    """A complete sweep for one arm: every declared configuration and draw.

    Confidence rises with length on purpose, so the length controls have
    something real to remove, and the low-temperature configurations emit
    near-duplicates so the degeneracy check has something real to find.
    """

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    collapsed = _sequence(rng, 180)
    for setting in ds.GRID:
        for cluster, _ in ds.clusters_for(ARM, None):
            for index in range(setting.draws_per_cluster):
                if setting.temperature <= 0.6:
                    sequence = collapsed
                elif not setting.allow_terminator:
                    sequence = _sequence(rng, 460)
                else:
                    sequence = _sequence(rng, int(rng.integers(80, 360)))
                tokens = max(1, len(sequence) // 2)
                rows.append(
                    {
                        "id": ds.candidate_id(
                            arm_name=ARM, config_key=setting.key, cluster=cluster, draw_index=index
                        ),
                        "schema_version": ds.SCHEMA_VERSION,
                        "campaign": ds.CAMPAIGN,
                        "role": "generated",
                        "arm": ARM,
                        "config_key": setting.key,
                        "config_axis": setting.axis,
                        "temperature": setting.temperature,
                        "top_p": setting.top_p,
                        "top_k": setting.top_k,
                        "max_new_tokens": setting.max_new_tokens,
                        "min_new_tokens": setting.min_new_tokens,
                        "allow_terminator": setting.allow_terminator,
                        "cluster": cluster,
                        "prompt_label": None,
                        "draw_index": index,
                        "cell_seed": ds.cell_seed(
                            arm_name=ARM, config_key=setting.key, cluster=cluster
                        ),
                        "batch_index": index // 8,
                        "batch_size": 8,
                        "sequence": sequence,
                        "sequence_sha256": ds.hashlib.sha256(sequence.encode()).hexdigest(),
                        "length": len(sequence),
                        "n_new_tokens": tokens,
                        "terminated_natively": setting.allow_terminator,
                        "stop_token_id": 0 if setting.allow_terminator else None,
                        "model_logprob_total": -1.5 * tokens,
                        "model_logprob_per_token": -1.5 + float(rng.normal(0, 0.2)),
                        "in_band": ds.in_band(sequence),
                        "out_of_band_reason": ds.out_of_band_reason(sequence),
                        "raw_continuation_length_chars": len(sequence) + 1,
                    }
                )
    return rows


def _confidence(length: int, rng: np.random.Generator) -> float:
    return float(np.clip(0.22 * length + rng.normal(0, 3.0), 5.0, 98.0))


def _fold_index(rows, path: Path, *, signature: str = "sig", seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for row in rows:
        value = _confidence(int(row["length"]), rng)
        payload.append(
            {
                "id": row["id"],
                "length": int(row["length"]),
                "sequence": row["sequence"],
                "role": row.get("role"),
                "arm": row.get("arm"),
                "structure": {
                    "status": "ok",
                    "mean_ca_plddt": value,
                    "fraction_ca_plddt_ge70": float(np.clip(value / 100.0, 0.0, 1.0)),
                    "ptm": float(np.clip(value / 100.0, 0.0, 1.0)),
                    "mean_pae_angstrom": float(30.0 - 0.25 * value),
                    "predicted_confidence_event": bool(value >= 70.0),
                    "object_directory": f"objects/{row['sequence_sha256'][:16]}",
                    "diffusion_sample_index": 0,
                    "evaluation_signature": signature,
                },
            }
        )
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in payload))


def _natural_index(path: Path, *, seed: int = 13) -> None:
    """A folded natural pool covering the band, plus one arm's existing results."""

    rng = np.random.default_rng(seed)
    rows = []
    for index, length in enumerate(list(range(50, 401, 2)) * 6):
        rows.append(
            {
                "id": f"nat_{index:05d}",
                "length": int(length),
                "sequence": _sequence(rng, int(length)),
                "role": "natural",
                "arm": None,
                "sequence_sha256": f"{index:064d}",
            }
        )
    for index, length in enumerate(list(range(60, 380, 4))):
        # Half under each role the reference cohort gives an arm's own products:
        # the conditioned arm's results carry only ``pool``, so an analysis that
        # read ``generated`` alone would leave that arm with no band.
        rows.append(
            {
                "id": f"old_{index:05d}",
                "length": int(length),
                "sequence": _sequence(rng, int(length)),
                "role": "generated" if index % 2 else "pool",
                "arm": ARM,
                "condition": "unconditioned",
                "sequence_sha256": f"{index + 900000:064d}",
            }
        )
    _fold_index(rows, path, seed=seed + 1)


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    directory = tmp_path_factory.mktemp("decoding")
    rows = _sweep_ledger()
    ledger = directory / "cell" / "attempts.jsonl"
    ledger.parent.mkdir(parents=True)
    ds.write_jsonl(ledger, rows)
    return {"dir": directory, "rows": rows, "ledger": ledger}


# ------------------------------------------------------------ cohort builder


def test_the_builder_accepts_a_complete_sweep_and_keeps_every_attempt(sweep):
    builder = _stage("build_decoding_cohort.py")
    out = sweep["dir"] / "cohort"
    record = builder.run(
        argparse.Namespace(
            ledgers=[sweep["ledger"]], queue=builder.DEFAULT_QUEUE, device="cpu", out=out
        )
    )
    assert record["status"] == "complete"
    assert record["arms"] == [ARM]
    assert record["n_attempts"] == len(sweep["rows"]) == sum(row.draws for row in ds.GRID)
    # The full ledger keeps the out-of-band and empty attempts so that every
    # census denominator stays the number of attempts actually made.
    assert record["n_in_band"] < record["n_attempts"]
    assert len(ds.read_jsonl(out / "sweep_ledger.jsonl")) == record["n_attempts"]
    assert len(ds.read_jsonl(out / "fold_cohort.jsonl")) == record["n_in_band"]
    # The run-on configuration produced nothing inside the band, which is its
    # result and must be visible as such rather than as a missing cell.
    run_on = record["census"][ARM]["per_configuration"]["t1.00_p0.95_k0_noterm400"]
    assert run_on["n_in_band"] == 0
    assert run_on["band_eligibility"]["above_band"] == run_on["n_draws"]
    sweep["cohort"] = out


def test_the_builder_refuses_an_incomplete_sweep(sweep, tmp_path):
    builder = _stage("build_decoding_cohort.py")
    short = tmp_path / "short.jsonl"
    ds.write_jsonl(short, sweep["rows"][:-1])
    with pytest.raises(SystemExit) as error:
        builder.run(
            argparse.Namespace(
                ledgers=[short], queue=builder.DEFAULT_QUEUE, device="cpu", out=tmp_path / "a"
            )
        )
    assert "not complete" in str(error.value)


def test_the_builder_refuses_a_cell_sampled_twice(sweep, tmp_path):
    builder = _stage("build_decoding_cohort.py")
    with pytest.raises(SystemExit) as error:
        builder.run(
            argparse.Namespace(
                ledgers=[sweep["ledger"], sweep["ledger"]],
                queue=builder.DEFAULT_QUEUE,
                device="cpu",
                out=tmp_path / "b",
            )
        )
    assert "more than once" in str(error.value)


# ---------------------------------------------------------------- analysis


@pytest.fixture(scope="module")
def analysed(sweep, tmp_path_factory):
    directory = sweep["dir"]
    rows = sweep["rows"]
    ledger = directory / "frozen.jsonl"
    ds.write_jsonl(ledger, rows)
    folds = directory / "folds"
    _fold_index([row for row in rows if row["in_band"]], folds / "index-000-of-001.jsonl")
    natural = directory / "natural" / "index-000-of-001.jsonl"
    _natural_index(natural)
    analyse = _stage("analyse_decoding_sweep.py")
    report = analyse.run(
        argparse.Namespace(
            ledger=ledger,
            structure=[folds],
            natural_structure=[natural],
            novelty=None,
            seed=3,
            resamples=200,
            device="cpu",
            out=directory / "analysis",
        )
    )
    return {"report": report, "analyse": analyse, "dirs": directory, "folds": folds, "natural": natural}


def test_the_report_covers_every_configuration_with_its_draw_count(analysed):
    report = analysed["report"]
    block = report["arms"][ARM]
    assert set(block["census"]) == set(ds.CONFIG_KEYS)
    for key in ds.CONFIG_KEYS:
        assert block["census"][key]["n_draws"] == ds.config(key).draws
        assert block["census"][key]["n_clusters"] == ds.CLUSTERS_PER_CONFIG
        # Every configuration-level block declares how many draws stood behind it.
        assert "length_distribution_all_products" in block["census"][key]
    assert report["reference_configuration"] == ds.REFERENCE_KEY
    assert report["evaluation_band"] == list(ds.EVALUATION_BAND)


def test_the_reference_band_and_the_existing_results_are_read_not_recomputed(analysed):
    block = analysed["report"]["arms"][ARM]
    assert block["natural_pool"]["n_records"] > 500
    assert block["existing_generation_results"]["n_records"] > 0
    assert set(block["existing_generation_results"]["n_by_role"]) == {"generated", "pool"}
    # The existing generation results pass through the same length-matched
    # arithmetic, so the sweep can be placed beside them.
    gap = block["existing_generation_results"]["length_matched_gap"][ds.PRIMARY_EVALUATOR]
    assert gap is not None and "interval" in gap


def test_every_configuration_has_a_length_matched_counterpart_or_a_named_refusal(analysed):
    block = analysed["report"]["arms"][ARM]
    for key in ds.CONFIG_KEYS:
        band = block["natural_band"][key][ds.PRIMARY_EVALUATOR]
        if "standardised" not in band:
            # Too few in-band products to evaluate at all, stated as such.
            assert band["resolved"] is False and band["unresolved_reason"]
            continue
        standardised = band["standardised"]
        assert standardised["resolved"] in (True, False)
        if standardised["resolved"]:
            assert standardised["gap_interval"] is not None
            assert standardised["unsupported_length_mass"] <= ds.MAX_UNSUPPORTED_MASS
        else:
            # Unresolved always says why, and the point estimate is withheld only
            # when the length control itself is unavailable -- never merely
            # because the interval could not be formed.
            assert standardised["unresolved_reason"]
            assert standardised["gap_interval"] is None
            if standardised["unsupported_length_mass"] > ds.MAX_UNSUPPORTED_MASS:
                assert standardised["gap"] is None
        paired = band["paired_whole_natural"]
        assert paired["resolved"] in (True, False)
        if not paired["resolved"]:
            assert paired["contrast"] is None


def test_a_length_only_difference_does_not_survive_standardisation(analysed):
    block = analysed["report"]["arms"][ARM]
    standardised = block["length_standardised"][ds.PRIMARY_EVALUATOR]
    resolved = {
        key: value
        for key, value in standardised["configurations"].items()
        if value["standardised_mean"] is not None
    }
    assert resolved, "no configuration covered the pooled length range"
    # Confidence in this fixture is a function of length alone, so once every
    # configuration is read at the same length mixture they must coincide.
    spread = max(value["standardised_mean"] for value in resolved.values()) - min(
        value["standardised_mean"] for value in resolved.values()
    )
    raw = max(value["raw_mean"] for value in resolved.values()) - min(
        value["raw_mean"] for value in resolved.values()
    )
    assert spread < raw


def test_the_matched_compute_table_is_reported_beside_the_per_candidate_view(analysed):
    report = analysed["report"]
    block = report["arms"][ARM]
    table = block["matched_compute"][ds.PRIMARY_EVALUATOR]
    assert len(table) == 3
    budgets = [row["generation_tokens_per_selected_candidate"] for row in table]
    assert budgets[0] < budgets[1] < budgets[2]
    for row in table:
        for entry in row["entries"]:
            # Inside one budget every configuration is within tolerance of the
            # same token cost, which is what makes the row a comparison.
            assert abs(entry["realised_over_budget"] - 1.0) <= row["tolerance_fraction"] + 1e-9
    # The deep-budget configurations are the ones whose grid of k reaches the
    # sixteen-fold budget at a comparable token cost; a shallow configuration
    # whose individual draws are expensive may also reach it, which is the point
    # of charging tokens rather than candidates.
    deepest = table[-1]
    reachable = {entry["configuration"] for entry in deepest["entries"]}
    assert set(ds.DEEP_BUDGET_KEYS) <= reachable
    assert set(ds.CONFIG_KEYS) - reachable == set(deepest["configurations_absent"])
    # A configuration that spent the budget and kept nothing is listed with no
    # mean rather than dropped, so its compute is still charged to it.
    run_on = [
        entry for entry in deepest["entries"] if entry["configuration"].endswith("noterm400")
    ]
    if run_on:
        assert run_on[0]["mean"] is None
        assert run_on[0]["block_yield"] == 0.0
    assert deepest["best_configuration"] in ds.CONFIG_KEYS
    curve = block["selection"][ds.REFERENCE_KEY]["curves"][ds.PRIMARY_EVALUATOR]
    assert set(curve) == {
        "model_logprob_per_token",
        "model_logprob_total",
        "esmfold2_confidence_oracle",
    }
    assert [point["k"] for point in curve["model_logprob_per_token"]] == [1, 2, 4, 8, 16, 32]


def test_the_verdict_answers_the_question_simultaneously(analysed):
    verdict = analysed["report"]["verdict"][ARM]
    assert verdict["primary_evaluator"] == ds.PRIMARY_EVALUATOR
    assert verdict["reference_configuration_gap"] is not None
    assert verdict["best_configuration"] in ds.CONFIG_KEYS
    assert verdict["any_configuration_reaches_natural_band"] in (True, False)
    assert verdict["simultaneous_upper_bound_on_best_gap"] is not None
    # In this fixture confidence is a pure function of length, so the arm sits on
    # the natural band and no configuration can beat it by much; the simultaneous
    # bound is what that claim is read off.
    marginals = [
        value["interval"][1]
        for value in analysed["report"]["arms"][ARM]["simultaneous"][ds.PRIMARY_EVALUATOR][
            "marginal"
        ].values()
        if value["interval"] is not None
    ]
    assert verdict["simultaneous_upper_bound_on_best_gap"] >= max(marginals) - 1e-6


def test_degeneracy_is_reported_and_the_collapsed_configuration_is_caught(analysed):
    block = analysed["report"]["arms"][ARM]
    collapsed = block["degeneracy"]["t0.60_p0.95_k0_eos400"]
    assert collapsed["profile"]["duplicate_fraction"] > 0.9
    assert collapsed["verdict"]["degenerate"]
    assert "duplicate_fraction" in collapsed["verdict"]["axes_flagged"]
    # Every configuration is priced against the reference even when it comes
    # earlier in grid order, so a gain cannot be lost to the reading order.
    assert all(
        value.get("gain_over_reference_configuration") is not None
        for key, value in block["degeneracy"].items()
        if key != ds.REFERENCE_KEY and "profile" in value
    )
    healthy = block["degeneracy"]["t1.20_p0.95_k0_eos400"]
    assert not healthy["verdict"]["degenerate"]
    assert block["novelty"] == {}, "no homology search was supplied, so none is claimed"


def test_an_unfolded_in_band_product_is_refused(analysed, tmp_path):
    analyse = analysed["analyse"]
    rows = ds.read_jsonl(analysed["dirs"] / "frozen.jsonl")
    partial = tmp_path / "partial"
    in_band = [row for row in rows if row["in_band"]]
    _fold_index(in_band[:-5], partial / "index-000-of-001.jsonl")
    with pytest.raises(SystemExit) as error:
        analyse.run(
            argparse.Namespace(
                ledger=analysed["dirs"] / "frozen.jsonl",
                structure=[partial],
                natural_structure=[analysed["natural"]],
                novelty=None,
                seed=3,
                resamples=50,
                device="cpu",
                out=tmp_path / "out",
            )
        )
    assert "have no fold" in str(error.value)


def test_a_reference_band_from_another_contract_is_refused(analysed, tmp_path):
    analyse = analysed["analyse"]
    rows = ds.read_jsonl(analysed["dirs"] / "frozen.jsonl")
    mine = tmp_path / "mine"
    _fold_index(
        [row for row in rows if row["in_band"]],
        mine / "index-000-of-001.jsonl",
        signature="a-different-contract",
    )
    with pytest.raises(SystemExit) as error:
        analyse.run(
            argparse.Namespace(
                ledger=analysed["dirs"] / "frozen.jsonl",
                structure=[mine],
                natural_structure=[analysed["natural"]],
                novelty=None,
                seed=3,
                resamples=50,
                device="cpu",
                out=tmp_path / "out3",
            )
        )
    assert "two measurement contracts" in str(error.value)


def test_folds_from_two_measurement_contracts_are_refused(analysed, tmp_path):
    analyse = analysed["analyse"]
    rows = ds.read_jsonl(analysed["dirs"] / "frozen.jsonl")
    in_band = [row for row in rows if row["in_band"]]
    mixed = tmp_path / "mixed"
    half = len(in_band) // 2
    _fold_index(in_band[:half], mixed / "index-000-of-002.jsonl", signature="one")
    _fold_index(in_band[half:], mixed / "index-001-of-002.jsonl", signature="two")
    with pytest.raises(SystemExit) as error:
        analyse.run(
            argparse.Namespace(
                ledger=analysed["dirs"] / "frozen.jsonl",
                structure=[mixed],
                natural_structure=[analysed["natural"]],
                novelty=None,
                seed=3,
                resamples=50,
                device="cpu",
                out=tmp_path / "out2",
            )
        )
    assert "evaluation signatures" in str(error.value)
