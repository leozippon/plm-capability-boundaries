"""Conditions the stage cross-scoring 2x2 analysis must always satisfy.

The cases decide whether the downstream comparison means anything: that a known
better scorer is recovered in both generators' pools, that swapping the scorer
labels flips the paired gain, that the selector is used only through its ordering
so a constant shift or a positive rescale of every NLL changes nothing, that the
campaign stream -- not the candidate identifier -- is the replication unit and the
interval is withheld below the unit floor, and that a pool digest, identifier set,
scorer binding, cohort hash, structure coverage or novelty coverage mismatch stops
the run instead of producing a partial artefact.

The correlations are pinned to the high-likelihood orientation: an ideal
ascending-NLL selector has association +1, and the ideal second scorer's paired
improvement is positive (negative after swapping scorer labels).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.capability.evaluation import analyse_stage_cross_scoring as acs
from scripts.capability.evaluation import prepare_stage_cross_scoring as pscs
from scripts.capability.evaluation import score_generated_likelihood as sjl
from src.capability.evaluation import generated_phenotype as gp
from src.capability.evaluation import likelihood_ranking as lr
from src.capability.generation import stage_pair as sp
from src.capability.generation import stage_pair_structures as sps

STAGE_ARMS = dict(sp.STAGES)
STREAMS = ("campaign_a", "campaign_b", "campaign_c")
PER_STREAM = 12
AA = gp.AA20


# ------------------------------------------------------------------- fixtures


def _sequence(salt: int, index: int) -> str:
    """A canonical sequence unique to ``(salt, index)``, with a varying length.

    The length varies so the length rung of the control ladder is not collinear with
    the intercept, which it would be on a fixture of one fixed length.
    """

    length = 50 + (int(index) % 8) * 5
    code = int(salt) * 1000 + int(index)
    digits = []
    for _ in range(5):
        digits.append(AA[code % 20])
        code //= 20
    body = [AA[(position * 7 + int(salt)) % 20] for position in range(length - 7)]
    sequence = "M" + "".join(digits) + "".join(body) + "K"
    assert len(sequence) == length and set(sequence) <= set(AA)
    return sequence


def _write_json(path: Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=1, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frozen_records() -> list[dict]:
    records = []
    for stage_index, stage in enumerate(sorted(sp.STAGES)):
        for stream_index, stream in enumerate(STREAMS):
            for index in range(PER_STREAM):
                sequence = _sequence(stage_index * 10 + stream_index, index)
                records.append(
                    {
                        "id": f"ge_{stage}_{stream}_{index}",
                        "stage": stage,
                        "arm": sp.STAGES[stage],
                        "stream": stream,
                        "sequence": sequence,
                        "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
                        "length": len(sequence),
                        "properties": {
                            "length": len(sequence),
                            "composition_entropy_nats": 2.0,
                            "distinct_residues": len(set(sequence)),
                        },
                    }
                )
    return records


def _write_stage_pair(root: Path) -> tuple[Path, Path, list[dict]]:
    records = _frozen_records()
    sequences = root / "stage_pair_sequences.jsonl"
    _write_jsonl(sequences, records)
    freeze = {
        "schema_version": sp.SCHEMA_VERSION,
        "status": "complete",
        "created_utc": "2026-10-08T00:00:00Z",
        "records_file": str(sequences),
        "records_sha256": _sha256(sequences),
        "n_sequences": len(records),
        "per_stream_target": PER_STREAM,
        "per_stage": {
            stage: {"arm": sp.STAGES[stage], "n_sequences": PER_STREAM * len(STREAMS)}
            for stage in sp.STAGES
        },
        "per_stream": [
            {
                "stage": stage,
                "stream": stream,
                "arm": sp.STAGES[stage],
                "n_selected": PER_STREAM,
                "selection_seed": 20261008,
                "length_band_residues": [40, 400],
            }
            for stream in STREAMS
            for stage in sp.STAGES
        ],
        "decoding": {
            "arms": dict(sp.STAGES),
            "symmetric": True,
            "manifest_sha256": "0" * 64,
            "declared": {"condition": "unconditioned", "prompt": "Seq=<", "temperature": 0.85},
        },
        "inputs": [],
        "no_new_sampling": "the retained cells were reused; none was sampled",
        "ceiling": {"conditional_comparison_is_impossible": "stage 1 has no instruction interface"},
    }
    freeze_path = root / "stage_pair_freeze.json"
    _write_json(freeze_path, freeze)
    return freeze_path, sequences, records


def _cohort_rows(cohort_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (cohort_dir / pscs.COHORT).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_likelihood_cell(
    directory: Path,
    *,
    generator: str,
    scorer: str,
    rows: list[dict],
    manifest: dict,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    sidecar = directory / "generated_likelihood.jsonl"
    scored = [{**row, "scorer_arm": scorer} for row in rows]
    _write_jsonl(sidecar, scored)
    values = [float(row["mean_nll_per_token_nats"]) for row in scored]
    receipt = {
        "schema_version": sjl.SCHEMA_VERSION,
        "status": "complete",
        "arm": scorer,
        "pool_arm": generator,
        "generator_arm": generator,
        "scorer_arm": scorer,
        "cross_arm": generator != scorer,
        "n_scored": len(scored),
        "n_pool_members": len(scored),
        "cohort": manifest["cohort_jsonl"],
        "cohort_sha256": manifest["cohort_sha256"],
        "pool_digest": manifest["pools"][generator]["pool_digest"],
        "annotations_jsonl": str(sidecar),
        "annotations_sha256": _sha256(sidecar),
        "rendering": {"input_format": "bare_seq_block"},
        "mean_nll_per_token_nats": {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        },
    }
    _write_json(directory / sjl.COMPLETION, receipt)


def _write_structures(
    directory: Path, cohort_rows: list[dict], endpoints: dict[str, dict[str, float]], *, drop_id: str | None
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": row["id"],
            "sequence_sha256": row["sequence_sha256"],
            "length": row["length"],
            "mean_ca_plddt": endpoints[row["id"]][lr.PRIMARY_ENDPOINT],
            "ptm": endpoints[row["id"]][lr.SECONDARY_ENDPOINT],
        }
        for row in cohort_rows
        if row["id"] != drop_id
    ]
    index = directory / acs.STRUCTURE_INDEX
    _write_jsonl(index, rows)
    _write_json(
        directory / acs.STRUCTURE_COMPLETION,
        {
            "schema_version": sps.SCHEMA_VERSION,
            "status": "complete",
            "index_file": str(index),
            "index_sha256": _sha256(index),
            "per_stage_counts": {},
        },
    )


def _write_recognition(
    directory: Path, cohort_rows: list[dict], *, drop_id: str | None, drop_key_id: str | None
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / acs.RECOGNITION_CONTROL, {"passed": True, "n_positive": 2, "n_negative": 2})
    rows = []
    for position, row in enumerate(cohort_rows):
        if row["id"] == drop_id:
            continue
        record = {
            "id": row["id"],
            "sequence_sha256": row["sequence_sha256"],
            "arm": row["arm"],
            "profile_search_status": "searched",
            "any_profile_hit": position % 5 != 0,
        }
        if row["id"] != drop_key_id:
            record["pfam_families"] = [f"PF{position % 4:05d}"] if position % 5 != 0 else []
        rows.append(record)
    sidecar = directory / acs.RECOGNITION_ROWS
    _write_jsonl(sidecar, rows)
    _write_json(
        directory / acs.RECOGNITION_COMPLETION,
        {
            "schema_version": "d1_family_recognition_v1",
            "status": "complete",
            "oracle_control": {"passed": True, "n_positive": 2, "n_negative": 2},
            "annotations_sha256": _sha256(sidecar),
        },
    )


def _write_novelty(
    directory: Path, cohort_rows: list[dict], *, cohort_sha256: str, drop_id: str | None
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": row["id"],
            "arm": row["arm"],
            "length": row["length"],
            "searched": True,
            "nearest_corpus_identity": 40.0 + 0.5 * position,
        }
        for position, row in enumerate(cohort_rows)
        if row["id"] != drop_id
    ]
    sidecar = directory / acs.NOVELTY_ROWS
    _write_jsonl(sidecar, rows)
    _write_json(
        directory / acs.NOVELTY_COMPLETION,
        {
            "schema_version": "d1_generated_novelty_v1",
            "status": "complete",
            "cohort_sha256": cohort_sha256,
            "annotations_sha256": _sha256(sidecar),
        },
    )


def build_tree(
    root: Path,
    *,
    swap_scorers: bool = False,
    constant_endpoint: bool = False,
    drop_structure_id: str | None = None,
    drop_recognition_id: str | None = None,
    drop_recognition_key_id: str | None = None,
    drop_novelty_id: str | None = None,
) -> SimpleNamespace:
    """A complete 2x2 fixture: cohort, four likelihood cells, structures, family, novelty."""

    root.mkdir(parents=True, exist_ok=True)
    freeze, sequences, _ = _write_stage_pair(root)
    cohort_dir = root / "cohort"
    manifest = pscs.run(argparse.Namespace(freeze=freeze, sequences=sequences, out=cohort_dir, device="cpu"))
    cohort_rows = _cohort_rows(cohort_dir)

    quality: dict[str, float] = {}
    by_arm: dict[str, list[dict]] = {}
    for row in cohort_rows:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    for arm, members in by_arm.items():
        ordered = sorted(members, key=lambda row: str(row["id"]))
        values = np.linspace(0.0, 1.0, len(ordered))
        for row, value in zip(ordered, values):
            quality[str(row["id"])] = float(value)

    endpoints = {
        row["id"]: {
            lr.PRIMARY_ENDPOINT: 60.0 if constant_endpoint else 50.0 + 30.0 * quality[row["id"]],
            lr.SECONDARY_ENDPOINT: 0.5 if constant_endpoint else 0.4 + 0.5 * quality[row["id"]],
        }
        for row in cohort_rows
    }

    cells: dict[tuple[str, str], Path] = {}
    for arm_index, (arm, members) in enumerate(sorted(by_arm.items())):
        ordered = sorted(members, key=lambda row: str(row["id"]))
        identifiers = [str(row["id"]) for row in ordered]
        rng = np.random.default_rng(9000 + arm_index)
        ideal = np.asarray([-quality[identifier] for identifier in identifiers], dtype=float)
        burned = ideal + rng.normal(scale=1.5, size=len(identifiers))
        for scorer in acs.SCORER_ARMS:
            # Without the swap the second scorer is the ideal quality ranking and the
            # first is the noisy one; the swap exchanges the two rankings.
            if scorer == acs.SCORER_1:
                values = ideal if swap_scorers else burned
            else:
                values = burned if swap_scorers else ideal
            rows = [
                {
                    "id": str(member["id"]),
                    "arm": arm,
                    "pool_arm": arm,
                    "length": int(member["length"]),
                    "scored_tokens": int(member["length"]),
                    "nll_sum_nats": float(value) * int(member["length"]),
                    "mean_nll_per_token_nats": float(value),
                    "mean_nll_per_residue_nats": float(value),
                }
                for member, value in zip(ordered, values)
            ]
            directory = root / "likelihood" / f"{arm}__{scorer}"
            _write_likelihood_cell(directory, generator=arm, scorer=scorer, rows=rows, manifest=manifest)
            cells[(arm, scorer)] = directory

    structures = root / "structures"
    _write_structures(
        structures, cohort_rows, endpoints, drop_id=drop_structure_id
    )
    recognition = root / "recognition"
    _write_recognition(
        recognition,
        cohort_rows,
        drop_id=drop_recognition_id,
        drop_key_id=drop_recognition_key_id,
    )
    novelty = root / "novelty"
    _write_novelty(
        novelty, cohort_rows, cohort_sha256=manifest["cohort_sha256"], drop_id=drop_novelty_id
    )
    return SimpleNamespace(
        root=root,
        cohort_dir=cohort_dir,
        manifest=manifest,
        cohort_rows=cohort_rows,
        likelihood=[cells[key] for key in sorted(cells)],
        cells=cells,
        structures=structures,
        recognition=recognition,
        novelty=novelty,
        quality=quality,
        endpoints=endpoints,
    )


def analysis_args(tree: SimpleNamespace, out: Path) -> argparse.Namespace:
    return argparse.Namespace(
        cohort_dir=tree.cohort_dir,
        likelihood=tree.likelihood,
        structures=tree.structures,
        recognition=tree.recognition,
        novelty=tree.novelty,
        device="cpu",
        out=out,
    )


def run_analysis(tree: SimpleNamespace, out: Path) -> dict:
    return acs.run(analysis_args(tree, out))


def _cell_dir(tree: SimpleNamespace, generator: str, scorer: str) -> Path:
    return tree.cells[(generator, scorer)]


def _repoint_likelihood(tree: SimpleNamespace, generator: str, scorer: str, rows: list[dict]) -> None:
    directory = _cell_dir(tree, generator, scorer)
    sidecar = directory / "generated_likelihood.jsonl"
    _write_jsonl(sidecar, rows)
    receipt_path = directory / sjl.COMPLETION
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["annotations_sha256"] = _sha256(sidecar)
    _write_json(receipt_path, receipt)


def _receipt(tree: SimpleNamespace, generator: str, scorer: str) -> dict:
    return json.loads((_cell_dir(tree, generator, scorer) / sjl.COMPLETION).read_text(encoding="utf-8"))


def _sidecar_rows(tree: SimpleNamespace, generator: str, scorer: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (_cell_dir(tree, generator, scorer) / "generated_likelihood.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------- the 2x2


def test_every_generator_pool_recovers_the_known_better_scorer(tmp_path):
    tree = build_tree(tmp_path)
    record = run_analysis(tree, tmp_path / "out")

    assert record["status"] == "complete"
    assert record["cohort"]["n_records"] == record["cohort"]["n_declared_records"] == 2 * PER_STREAM * len(STREAMS)
    assert set(record["pools"]) == set(sp.STAGES.values())
    assert record["convention"]["paired_difference"] == f"{acs.SCORER_2} minus {acs.SCORER_1}"

    for generator in sorted(record["pools"]):
        pool = record["pools"][generator]
        assert pool["pool_digest"] == tree.manifest["pools"][generator]["pool_digest"]
        assert pool["n_candidates"] == pool["n_declared_candidates"] == PER_STREAM * len(STREAMS)
        for endpoint in lr.ENDPOINTS:
            block = pool["endpoints"][endpoint]
            assert set(block["scorers"]) == set(acs.SCORER_ARMS)
            first = block["scorers"][acs.SCORER_1]
            second = block["scorers"][acs.SCORER_2]
            # The second scorer is the exact-quality ranking, so no fraction can beat
            # it and at the wide budgets it must strictly beat the noisy ranking.
            for key in ("0.25", "0.50"):
                assert (
                    second["selection"][key]["selected_mean"]
                    > first["selection"][key]["selected_mean"]
                )
            assert (
                block["paired"]["selection"]["0.50"]["selected_mean_difference"] > 0.0
            )
            assert (
                block["paired"]["selection"]["0.25"]["selected_mean_difference"] > 0.0
            )
            assert second["whole_pool"]["high_likelihood_association"] == pytest.approx(1.0)
            assert second["control_ladder"]["length"]["high_likelihood_association"] == pytest.approx(1.0)
            assert block["paired"]["whole_pool_high_likelihood_association_difference"] > 0.0
            assert (
                block["paired"]["control_ladder_high_likelihood_association_difference"]
                ["length"]["high_likelihood_association_difference"] > 0.0
            )


def test_each_cell_scores_the_same_frozen_ids_as_its_generator(tmp_path):
    tree = build_tree(tmp_path)
    record = run_analysis(tree, tmp_path / "out")

    for generator in sorted(record["pools"]):
        expected = {
            str(row["id"])
            for row in tree.cohort_rows
            if str(row["arm"]) == generator
        }
        for scorer in acs.SCORER_ARMS:
            rows = _sidecar_rows(tree, generator, scorer)
            assert {str(row["id"]) for row in rows} == expected
            assert {str(row["scorer_arm"]) for row in rows} == {scorer}


def test_swapping_the_scorers_flips_the_paired_gain(tmp_path):
    normal = build_tree(tmp_path / "normal")
    swapped = build_tree(tmp_path / "swapped", swap_scorers=True)
    first = run_analysis(normal, tmp_path / "out_normal")
    second = run_analysis(swapped, tmp_path / "out_swapped")

    for generator in sorted(first["pools"]):
        for endpoint in lr.ENDPOINTS:
            left = first["pools"][generator]["endpoints"][endpoint]["paired"]
            right = second["pools"][generator]["endpoints"][endpoint]["paired"]
            a = left["selection"]["0.50"]["selected_mean_difference"]
            b = right["selection"]["0.50"]["selected_mean_difference"]
            assert a > 0.0 > b
            assert left["whole_pool_high_likelihood_association_difference"] > 0.0
            assert right["whole_pool_high_likelihood_association_difference"] == pytest.approx(
                -left["whole_pool_high_likelihood_association_difference"]
            )
            assert right["whole_pool_high_likelihood_association_difference"] < 0.0


# ------------------------------------------------------------- rank invariance


def _synthetic_inputs(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    n = PER_STREAM * len(STREAMS)
    identifiers = [f"syn_{index:03d}" for index in range(n)]
    streams = [STREAMS[index % len(STREAMS)] for index in range(n)]
    quality = np.linspace(0.0, 1.0, n)
    sequences = [_sequence(77, index) for index in range(n)]
    families = [[f"PF{index % 3:05d}"] if index % 4 else [] for index in range(n)]
    identities = [40.0 + 30.0 * value for value in quality]
    features = [
        lr.make_row_features(
            sequences[index],
            len(sequences[index]),
            nearest_corpus_identity=identities[index],
            families=families[index],
        )
        for index in range(n)
    ]
    nll = {
        acs.SCORER_1: -quality + rng.normal(scale=1.5, size=n),
        acs.SCORER_2: -quality,
    }
    return {
        "generator_arm": STAGE_ARMS["stage_1"],
        "scorer_arms": acs.SCORER_ARMS,
        "ids": identifiers,
        "streams": streams,
        "nll": nll,
        "features": features,
        "sequences": sequences,
        "families": families,
        "identities": identities,
        "endpoints": {
            lr.PRIMARY_ENDPOINT: 50.0 + 30.0 * quality,
            lr.SECONDARY_ENDPOINT: 0.4 + 0.5 * quality,
        },
    }


def test_constant_shift_and_positive_rescale_leave_every_rank_quantity_unchanged():
    inputs = _synthetic_inputs()
    base = acs.analyse_pool(**inputs)
    shifted_inputs = dict(inputs)
    shifted_inputs["nll"] = {scorer: vector + 7.0 for scorer, vector in inputs["nll"].items()}
    scaled_inputs = dict(inputs)
    scaled_inputs["nll"] = {scorer: vector * 3.0 for scorer, vector in inputs["nll"].items()}
    shifted = acs.analyse_pool(**shifted_inputs)
    scaled = acs.analyse_pool(**scaled_inputs)

    for endpoint in lr.ENDPOINTS:
        for scorer in acs.SCORER_ARMS:
            for key in base["endpoints"][endpoint]["scorers"][scorer]["selection"]:
                reference = base["endpoints"][endpoint]["scorers"][scorer]["selection"][key]
                for other in (shifted, scaled):
                    candidate = other["endpoints"][endpoint]["scorers"][scorer]["selection"][key]
                    assert candidate["selected_mean"] == reference["selected_mean"]
                    assert candidate["oracle_bound"] == reference["oracle_bound"]
                    assert candidate["gain_over_random"] == reference["gain_over_random"]
            reference = base["endpoints"][endpoint]["scorers"][scorer]
            for other in (shifted, scaled):
                candidate = other["endpoints"][endpoint]["scorers"][scorer]
                assert (
                    candidate["whole_pool"]["high_likelihood_association"]
                    == reference["whole_pool"]["high_likelihood_association"]
                )
                assert (
                    candidate["control_ladder"]["length"]["high_likelihood_association"]
                    == reference["control_ladder"]["length"]["high_likelihood_association"]
                )
        for other in (shifted, scaled):
            assert (
                other["endpoints"][endpoint]["paired"][
                    "whole_pool_high_likelihood_association_difference"
                ]
                == base["endpoints"][endpoint]["paired"][
                    "whole_pool_high_likelihood_association_difference"
                ]
            )
    # The descriptive NLL level is allowed to move; the ranks and their consequences
    # are the only thing the analysis reads.
    assert (
        base["endpoints"][lr.PRIMARY_ENDPOINT]["scorers"][acs.SCORER_2]["nll_descriptive"]["mean"]
        != shifted["endpoints"][lr.PRIMARY_ENDPOINT]["scorers"][acs.SCORER_2]["nll_descriptive"]["mean"]
    )


def test_constant_outcome_is_explicitly_unavailable_and_json_stays_strict(tmp_path):
    tree = build_tree(tmp_path, constant_endpoint=True)
    record = run_analysis(tree, tmp_path / "out")

    for generator in sorted(record["pools"]):
        for endpoint in lr.ENDPOINTS:
            for scorer in acs.SCORER_ARMS:
                whole = record["pools"][generator]["endpoints"][endpoint]["scorers"][scorer][
                    "whole_pool"
                ]
                assert whole["status"] == "unavailable"
                assert whole["reason"] == "endpoint_constant"
                assert whole["high_likelihood_association"] is None

    text = (tmp_path / "out" / acs.COMPLETION).read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["status"] == "complete"


# --------------------------------------------------------------- independence


def test_three_streams_are_reported_without_a_population_interval(tmp_path):
    tree = build_tree(tmp_path)
    record = run_analysis(tree, tmp_path / "out")

    for generator in sorted(record["pools"]):
        pool = record["pools"][generator]
        assert pool["independence"]["unit"] == "stream"
        assert pool["independence"]["n_streams"] == 3
        assert pool["independence"]["n_rows"] == PER_STREAM * len(STREAMS)
        assert pool["independence"]["unit_floor"]["degenerate"] is True
        for endpoint in lr.ENDPOINTS:
            summary = pool["endpoints"][endpoint]["paired"]["stream_summary"]
            assert summary["n_streams"] == 3
            assert len(summary["per_stream"]) == 3
            assert summary["interval"] is None
            assert summary["interval_reason"]
            assert summary["mean_difference"] is not None
            assert summary["direction_replicated"] is not None
            for key, value in summary["per_stream"].items():
                assert value is not None


# ----------------------------------------------------------------- refusals


def test_pool_digest_cohort_hash_and_scorer_identity_mismatches_are_refused(tmp_path):
    generator = STAGE_ARMS["stage_1"]
    scorer = acs.SCORER_1

    digests = build_tree(tmp_path / "digest")
    receipt_path = _cell_dir(digests, generator, scorer) / sjl.COMPLETION
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["pool_digest"] = "0" * 64
    _write_json(receipt_path, receipt)
    with pytest.raises(SystemExit, match="pool digest"):
        run_analysis(digests, tmp_path / "digest_out")

    cohort = build_tree(tmp_path / "cohort")
    receipt_path = _cell_dir(cohort, generator, scorer) / sjl.COMPLETION
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["cohort_sha256"] = "1" * 64
    _write_json(receipt_path, receipt)
    with pytest.raises(SystemExit, match="cohort sha256"):
        run_analysis(cohort, tmp_path / "cohort_out")

    bound = build_tree(tmp_path / "bound")
    rows = _sidecar_rows(bound, generator, scorer)
    rows[0]["scorer_arm"] = acs.SCORER_2
    _repoint_likelihood(bound, generator, scorer, rows)
    with pytest.raises(SystemExit, match="names scorer"):
        run_analysis(bound, tmp_path / "bound_out")


def test_missing_and_extra_candidate_ids_and_duplicate_rows_within_a_cell_are_refused(tmp_path):
    generator = STAGE_ARMS["stage_1"]
    scorer = acs.SCORER_1

    missing = build_tree(tmp_path / "missing")
    rows = _sidecar_rows(missing, generator, scorer)[:-1]
    _repoint_likelihood(missing, generator, scorer, rows)
    with pytest.raises(SystemExit, match="does not cover"):
        run_analysis(missing, tmp_path / "missing_out")

    extra = build_tree(tmp_path / "extra")
    rows = _sidecar_rows(extra, generator, scorer)
    rows.append({**rows[0], "id": "ge_not_in_the_pool"})
    _repoint_likelihood(extra, generator, scorer, rows)
    with pytest.raises(SystemExit, match="does not cover"):
        run_analysis(extra, tmp_path / "extra_out")

    duplicated = build_tree(tmp_path / "duplicated")
    rows = _sidecar_rows(duplicated, generator, scorer)
    rows.append(dict(rows[0]))
    _repoint_likelihood(duplicated, generator, scorer, rows)
    with pytest.raises(SystemExit, match="appears twice"):
        run_analysis(duplicated, tmp_path / "duplicated_out")


def test_missing_structure_recognition_or_novelty_is_refused(tmp_path):
    cohort_rows = _frozen_records()
    first_id = cohort_rows[0]["id"]

    structures = build_tree(tmp_path / "structures", drop_structure_id=first_id)
    with pytest.raises(SystemExit, match="carry no structure"):
        run_analysis(structures, tmp_path / "structures_out")

    recognition = build_tree(tmp_path / "recognition", drop_recognition_id=first_id)
    with pytest.raises(SystemExit, match="carry no recognition row"):
        run_analysis(recognition, tmp_path / "recognition_out")

    novelty = build_tree(tmp_path / "novelty", drop_novelty_id=first_id)
    with pytest.raises(SystemExit, match="carry no novelty row"):
        run_analysis(novelty, tmp_path / "novelty_out")


def test_an_absent_family_key_is_refused_but_an_empty_family_set_is_not(tmp_path):
    cohort_rows = _frozen_records()
    first_id = cohort_rows[0]["id"]

    tree = build_tree(tmp_path / "key", drop_recognition_key_id=first_id)
    with pytest.raises(SystemExit, match="pfam_families"):
        run_analysis(tree, tmp_path / "key_out")

    # An explicit empty list is a valid "nothing recognised" and the run completes.
    valid = build_tree(tmp_path / "valid")
    record = run_analysis(valid, tmp_path / "valid_out")
    assert record["status"] == "complete"


def test_a_conditioned_row_is_refused_rather_than_analysed(tmp_path):
    tree = build_tree(tmp_path / "conditioned")
    cohort_path = tree.cohort_dir / pscs.COHORT
    rows = _cohort_rows(tree.cohort_dir)
    rows[0]["condition"] = "requested"
    _write_jsonl(cohort_path, rows)
    manifest_path = tree.cohort_dir / pscs.COMPLETION
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cohort_sha256"] = _sha256(cohort_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(SystemExit, match="condition"):
        run_analysis(tree, tmp_path / "conditioned_out")


def _mutate_annotations(directory: Path, rows_file: str, receipt_file: str, field: str, value) -> None:
    path = directory / rows_file
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0][field] = value
    _write_jsonl(path, rows)
    receipt_path = directory / receipt_file
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["annotations_sha256"] = _sha256(path)
    _write_json(receipt_path, receipt)


@pytest.mark.parametrize("digest", [None, "0" * 64])
def test_novelty_requires_the_exact_cohort_receipt(tmp_path, digest):
    tree = build_tree(tmp_path)
    path = tree.novelty / acs.NOVELTY_COMPLETION
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if digest is None:
        del receipt["cohort_sha256"]
    else:
        receipt["cohort_sha256"] = digest
    _write_json(path, receipt)
    with pytest.raises(SystemExit, match="novelty cohort sha256"):
        run_analysis(tree, tmp_path / "out")


@pytest.mark.parametrize("field,value,reason", [
    ("searched", False, "not searched"),
    ("searched", 1, "not searched"),
    ("arm", "wrong", "different arm"),
    ("nearest_corpus_identity", -0.1, "0..100"),
    ("nearest_corpus_identity", 100.1, "0..100"),
    ("nearest_corpus_identity", float("nan"), "finite"),
    ("nearest_corpus_identity", "bad", "unusable identity"),
])
def test_novelty_search_and_identity_are_verified(tmp_path, field, value, reason):
    tree = build_tree(tmp_path)
    _mutate_annotations(tree.novelty, acs.NOVELTY_ROWS, acs.NOVELTY_COMPLETION, field, value)
    with pytest.raises(SystemExit, match=reason):
        run_analysis(tree, tmp_path / "out")


@pytest.mark.parametrize("field,value,reason", [
    ("sequence_sha256", "0" * 64, "different sequence_sha256"),
    ("arm", "wrong", "different arm"),
    ("profile_search_status", "empty_not_searched", "profile_search_status"),
])
def test_recognition_rows_are_bound_and_searched(tmp_path, field, value, reason):
    tree = build_tree(tmp_path)
    _mutate_annotations(tree.recognition, acs.RECOGNITION_ROWS, acs.RECOGNITION_COMPLETION, field, value)
    with pytest.raises(SystemExit, match=reason):
        run_analysis(tree, tmp_path / "out")


@pytest.mark.parametrize("file,field,value", [
    (acs.RECOGNITION_CONTROL, "passed", False),
    (acs.RECOGNITION_CONTROL, "passed", 1),
    (acs.RECOGNITION_COMPLETION, "passed", False),
    (acs.RECOGNITION_COMPLETION, "passed", 1),
    (acs.RECOGNITION_CONTROL, "n_positive", 3),
    (acs.RECOGNITION_COMPLETION, "n_negative", 3),
])
def test_recognition_control_summary_cannot_be_detached(tmp_path, file, field, value):
    tree = build_tree(tmp_path)
    path = tree.recognition / file
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["oracle_control"] if file == acs.RECOGNITION_COMPLETION else payload
    summary[field] = value
    _write_json(path, payload)
    with pytest.raises(SystemExit, match="control"):
        run_analysis(tree, tmp_path / "out")


def test_cohort_cannot_be_redeclared_as_a_smaller_self_consistent_pool(tmp_path):
    tree = build_tree(tmp_path)
    rows = tree.cohort_rows[1:]
    path = tree.cohort_dir / pscs.COHORT
    _write_jsonl(path, rows)
    manifest = dict(tree.manifest)
    manifest["cohort_sha256"] = _sha256(path)
    manifest["n_records"] = len(rows)
    for arm, block in manifest["pools"].items():
        members = sorted((row for row in rows if row["arm"] == arm), key=lambda row: row["id"])
        block["n_members"] = len(members)
        block["pool_digest"] = gp.cohort_digest(members)
    _write_json(tree.cohort_dir / pscs.COMPLETION, manifest)
    with pytest.raises(SystemExit, match="exact declared frozen records"):
        acs.load_cohort(tree.cohort_dir)


@pytest.mark.parametrize("role", ["freeze", "sequences"])
def test_cohort_freeze_provenance_must_match_source_bytes(tmp_path, role):
    tree = build_tree(tmp_path)
    path = Path(tree.manifest["provenance"][role]["path"])
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=f"{role} provenance sha256"):
        acs.load_cohort(tree.cohort_dir)


@pytest.mark.parametrize("field", ["n_by_stream", "n_streams", "stage"])
def test_pool_stream_declarations_must_match_the_freeze(tmp_path, field):
    tree = build_tree(tmp_path)
    manifest = tree.manifest
    block = manifest["pools"][STAGE_ARMS["stage_1"]]
    block[field] = {} if field == "n_by_stream" else "wrong"
    _write_json(tree.cohort_dir / pscs.COMPLETION, manifest)
    with pytest.raises(SystemExit, match="declared frozen stream counts"):
        acs.load_cohort(tree.cohort_dir)


@pytest.mark.parametrize("role", ["freeze", "sequences"])
def test_missing_freeze_provenance_is_refused(tmp_path, role):
    tree = build_tree(tmp_path)
    manifest = tree.manifest
    del manifest["provenance"][role]
    _write_json(tree.cohort_dir / pscs.COMPLETION, manifest)
    with pytest.raises(SystemExit, match=f"{role} provenance receipt"):
        acs.load_cohort(tree.cohort_dir)


@pytest.mark.parametrize("n,constant_length,status,reason", [
    (36, True, "unavailable", "rank_deficient"),
    (2, False, "unavailable", "saturated"),
    (3, False, "unavailable", "saturated"),
    (36, False, "measured", None),
])
def test_design_status_reflects_rank_and_residual_support(n, constant_length, status, reason):
    inputs = _synthetic_inputs()
    for key in ("ids", "streams", "features", "sequences", "families", "identities"):
        inputs[key] = inputs[key][:n]
    inputs["nll"] = {key: values[:n] for key, values in inputs["nll"].items()}
    inputs["endpoints"] = {key: values[:n] for key, values in inputs["endpoints"].items()}
    if constant_length:
        inputs["features"] = [replace(feature, length=50.0) for feature in inputs["features"]]
    designs, _, _ = acs._build_designs(inputs["features"])
    design = designs["length"]
    assert design["status"] == status
    assert design["reason"] == reason
    diagnostics = design["diagnostics"]
    nested = lr.partial_spearman(
        inputs["nll"][acs.SCORER_2], inputs["endpoints"][lr.PRIMARY_ENDPOINT], design["matrix"]
    )
    assert nested["status"] == status
    assert nested.get("reason") == reason
    record = acs.analyse_pool(**inputs)
    assert record["control_ladder"]["length"]["status"] == status
    assert record["control_ladder"]["length"]["design"] == diagnostics
    assert record["control_ladder"]["length"]["n_support"] == n
    for endpoint in lr.ENDPOINTS:
        for scorer in acs.SCORER_ARMS:
            block = record["endpoints"][endpoint]["scorers"][scorer]
            assert block["whole_pool"]["status"] == "measured"
            assert block["selection"]["0.50"]["selected_mean"] is not None
            rung = block["control_ladder"]["length"]
            assert rung["status"] == status
            assert rung["reason"] == reason
            assert rung["design"] == diagnostics
            if status == "unavailable":
                assert rung["high_likelihood_association"] is None
                assert rung["adjusted_selection"] == {}


# --------------------------------------------------------------- small surface


def _keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def test_the_output_never_carries_per_variant_sequences(tmp_path):
    tree = build_tree(tmp_path)
    record = run_analysis(tree, tmp_path / "out")
    text = (tmp_path / "out" / acs.COMPLETION).read_text(encoding="utf-8")
    for row in tree.cohort_rows:
        assert str(row["sequence"]) not in text
    keys = set(_keys(record))
    assert "sequence" not in keys
