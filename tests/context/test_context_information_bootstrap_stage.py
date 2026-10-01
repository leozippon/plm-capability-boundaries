"""What the context-information re-analysis stage must do, including its refusals.

The fixtures here are real sidecars: synthetic protein records are tokenised,
reduced to the same per-record sufficient statistics ``cohort_power.py``
produces, and written through ``budget.write_power_records`` itself. Nothing is
mocked, so a test that passes is evidence about the file format the stage will
actually meet on disk rather than about a second format written in the test.

The load-bearing assertions are the negative ones. A re-analysis that quietly
drops a refused arm, quietly pairs two cohorts, or quietly resamples records
instead of near-duplicate groups still produces a complete-looking artefact full
of plausible intervals, so each of those is asserted against directly.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from src.capability.core.arms import Cohort
from src.capability.core.budget import (
    IDENTIFICATION_CRITERION,
    SCREENING_CONTEXT_INFORMATION_NATS,
    RecordStatistics,
    SparseCounts,
    write_power_records,
)
from src.capability.context.information_bootstrap import bootstrap_arms, unpaired_contrast

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_PATH = REPO_ROOT / "scripts" / "capability" / "stages/context_information_bootstrap.py"

RESIDUES = "ACDEFGHIKLMNPQRSTVWY"

#: Draws: 0.025 * 400 = 10 in each tail, which is the module's floor exactly.
DRAWS = 400


def _load_stage():
    spec = importlib.util.spec_from_file_location("_stage_41", STAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage = _load_stage()


# --------------------------------------------------------------------------- #
# Fixtures: synthetic cohorts written through the real sidecar writer
# --------------------------------------------------------------------------- #


def synthetic_records(
    seed: int, *, n_families: int, per_family: int, length: int, mutations: int
) -> list[str]:
    """Records with a declared near-duplicate structure.

    One family is one near-duplicate group: a founder and ``per_family - 1``
    point-mutated copies of it, which single-linkage 5-mer containment joins.
    Distinct families share nothing at 5-mer scale, because two independent draws
    of a 120-residue string share essentially none of their 3.2 million possible
    5-mers.
    """

    rng = np.random.default_rng(seed)
    records: list[str] = []
    for _ in range(n_families):
        founder = "".join(rng.choice(list(RESIDUES), size=length))
        records.append(founder)
        for _ in range(per_family - 1):
            letters = list(founder)
            for position in rng.choice(length, size=mutations, replace=False):
                letters[int(position)] = str(rng.choice(list(RESIDUES)))
            records.append("".join(letters))
    return records


def tokenise(record: str, *, unit: int, vocab_size: int) -> np.ndarray:
    """Residue ids, or paired-residue ids, inside the declared vocabulary."""

    indices = [RESIDUES.index(symbol) for symbol in record]
    if unit == 1:
        ids = indices
    else:
        ids = [a * len(RESIDUES) + b for a, b in zip(indices[::2], indices[1::2])]
    assert max(ids) < vocab_size
    return np.asarray(ids, dtype=np.int64)


def arm_statistics(
    name: str,
    cohort_records: list[str],
    reference_records: list[str],
    *,
    unit: int,
    vocab_size: int,
    information: float,
    smoothing: float = 1.0,
    empty_reference_rows: int = 0,
) -> RecordStatistics:
    """Per-record statistics whose context information is ``information`` by design.

    The model term is the record's cross-entropy under the smoothed reference
    unigram less ``information`` per token, so ``I`` recovers ``information`` up
    to the reference's own estimation error and the smoothing constant.
    """

    reference_blocks = [
        tokenise(record, unit=unit, vocab_size=vocab_size) for record in reference_records
    ]
    reference_blocks += [np.zeros(0, dtype=np.int64)] * empty_reference_rows
    counts = np.zeros(vocab_size, dtype=np.float64)
    for block in reference_blocks:
        counts += np.bincount(block, minlength=vocab_size)
    probabilities = (counts + smoothing) / (counts.sum() + smoothing * vocab_size)
    surprisal = -np.log(probabilities)

    cohort_blocks = [
        tokenise(record, unit=unit, vocab_size=vocab_size) for record in cohort_records
    ]
    return RecordStatistics(
        arm=name,
        vocab_size=vocab_size,
        record_index=np.arange(len(cohort_blocks), dtype=np.int64),
        clean_nll_sum=np.asarray(
            [float(surprisal[block].sum()) - information * block.size for block in cohort_blocks],
            dtype=np.float64,
        ),
        token_count=np.asarray([block.size for block in cohort_blocks], dtype=np.int64),
        n_symbols=np.asarray(
            [len(record) for record in cohort_records], dtype=np.int64
        ),
        target_counts=SparseCounts.from_records(cohort_blocks),
        reference_counts=SparseCounts.from_records(reference_blocks),
    )


def write_block(
    directory: Path,
    name: str,
    cohort_records: list[str],
    reference_records: list[str],
    *,
    arms: dict[str, dict],
    smoothing: float = 1.0,
    write_cohort_json: bool = True,
    write_reference_json: bool = False,
    declare_reference_digest: bool = True,
) -> dict[str, Path]:
    """One sidecar and its companions, named the way stage 01 names them."""

    cohort = Cohort(
        name=name,
        kind="protein",
        records=list(cohort_records),
        min_symbols=min(len(r) for r in cohort_records),
        max_symbols=max(len(r) for r in cohort_records),
    )
    reference = Cohort(
        name=f"{name}_reference",
        kind="protein",
        records=list(reference_records),
        min_symbols=min(len(r) for r in reference_records),
        max_symbols=max(len(r) for r in reference_records),
    )
    digest = cohort.digest
    directory.mkdir(parents=True, exist_ok=True)
    statistics = {
        arm: arm_statistics(
            arm, cohort_records, reference_records, smoothing=smoothing, **options
        )
        for arm, options in arms.items()
    }
    sidecar = directory / f"power_{name}_{digest[:12]}.records.npz"
    write_power_records(
        sidecar,
        statistics,
        cohort_digest=digest,
        reference_digest=reference.digest if declare_reference_digest else None,
        smoothing=smoothing,
        seeds={"cohort_draw": 20260101},
        max_len=384,
    )
    paths = {"sidecar": sidecar}
    if write_cohort_json:
        cohort_json = directory / f"cohort_{name}_{digest[:12]}.json"
        cohort_json.write_text(
            json.dumps(
                {
                    "schema_version": "r2_transfer_cohort_power_v1",
                    "artifact": "frozen_cohort",
                    "cohort_digest": digest,
                    "cohort_name": name,
                    "cohort_kind": "protein",
                    "min_symbols": cohort.min_symbols,
                    "max_symbols": cohort.max_symbols,
                    "n_records": len(cohort_records),
                    "records": list(cohort_records),
                    "metadata": {},
                }
            ),
            encoding="utf-8",
        )
        paths["cohort_json"] = cohort_json
    if write_reference_json:
        reference_json = directory / f"reference_{name}_{digest[:12]}.json"
        reference_json.write_text(
            json.dumps({"records": list(reference_records)}), encoding="utf-8"
        )
        paths["reference_json"] = reference_json
    return paths


TWO_ARMS = {
    "progen2-base": {"unit": 1, "vocab_size": 32, "information": 0.55},
    "progen2-medium": {"unit": 1, "vocab_size": 32, "information": 0.50},
}


def standard_block(directory: Path, name: str, seed: int, **kwargs) -> dict[str, Path]:
    cohort = synthetic_records(seed, n_families=12, per_family=2, length=120, mutations=6)
    reference = synthetic_records(
        seed + 1000, n_families=40, per_family=2, length=200, mutations=10
    )
    return write_block(directory, name, cohort, reference, arms=dict(TWO_ARMS), **kwargs)


def run_stage(argv: list[str]) -> dict:
    stage.main(argv)
    out = Path(argv[argv.index("--out") + 1])
    name = argv[argv.index("--report-name") + 1]
    return json.loads((out / name).read_text(encoding="utf-8"))


def base_argv(paths: dict[str, Path], out: Path, **extra) -> list[str]:
    argv = [
        "--sidecar",
        str(paths["sidecar"]),
        "--n-bootstrap",
        str(DRAWS),
        "--seed",
        "31337",
        "--out",
        str(out),
        "--report-name",
        "report.json",
    ]
    if "reference_json" in paths:
        argv += ["--reference-json", str(paths["reference_json"])]
    for flag, value in extra.items():
        argv += [f"--{flag.replace('_', '-')}", *[str(v) for v in value]]
    return argv


def blocks_argv(paths: list[dict[str, Path]], out: Path, *extra: str) -> list[str]:
    references = [entry["reference_json"] for entry in paths if "reference_json" in entry]
    assert len(references) in (0, len(paths)), "--reference-json takes all or none"
    return [
        "--sidecar",
        *[str(entry["sidecar"]) for entry in paths],
        *(["--reference-json", *[str(path) for path in references]] if references else []),
        "--n-bootstrap",
        str(DRAWS),
        "--seed",
        "4242",
        "--alpha-sweep",
        "1.0",
        "--out",
        str(out),
        "--report-name",
        "report.json",
        *extra,
    ]


def row_for(payload: dict, arm: str, block_id: str = "b0") -> dict:
    matches = [
        row
        for row in payload["arm_results"]
        if row["arm"] == arm and row["block_id"] == block_id
    ]
    assert len(matches) == 1, f"{arm} appears {len(matches)} times in {block_id}"
    return matches[0]


# --------------------------------------------------------------------------- #
# 1. The happy path exists, and says what it measured
# --------------------------------------------------------------------------- #


def test_a_well_grouped_cohort_produces_a_group_level_interval(tmp_path: Path) -> None:
    paths = standard_block(tmp_path / "block", "swissprot", seed=1)
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    row = row_for(payload, "progen2-base")
    assert row["refused"] is False
    assert row["resampling_unit"] == "group"
    assert row["cohort_draw_shared_between_terms"] is True
    assert row["reference_resampled"] is True
    assert row["context_information_nats"] == pytest.approx(0.55, abs=0.12)
    low, high = row["bootstrap_ci_95"]
    assert low < row["context_information_nats"] < high
    # The grouping is the near-duplicate one, not one record per unit.
    assert payload["blocks"][0]["cohort_grouping"]["available"] is True
    assert row["n_groups"] < row["n_scored_records"]
    for field in (
        "bootstrap_fraction_nonpositive",
        "bootstrap_bias",
        "median_bias_z0",
        "interval_mc_se",
        "relative_information",
        "relative_information_ci_95",
        "information_bits_per_symbol",
        "information_bits_per_symbol_ci_95",
        "n_scored_records",
        "n_scored_tokens",
        "n_reference_records",
        "n_reference_tokens",
        "n_effective_groups",
        "largest_group_token_share",
        "n_singleton_groups",
        "top10_record_token_share",
        "cohort_token_share_unseen_in_reference",
        "seed",
        "n_bootstrap",
    ):
        assert field in row, field
        assert row[field] is not None, field


def test_the_operative_verdict_is_the_criterion_and_the_floor_is_a_column(
    tmp_path: Path,
) -> None:
    """The row's verdict comes from ``budget.context_identification``, not a floor.

    Both halves matter. If the row took its verdict from the floor, the reading
    the floor refuses on margin would still be refused; if the floor were dropped
    rather than demoted, every verdict recorded under it would stop being
    readable. So the criterion decides and the floor is reported beside it,
    marked as deciding nothing.
    """

    paths = standard_block(tmp_path / "block", "swissprot", seed=2)
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    row = row_for(payload, "progen2-base")
    identification = row["context_identification"]
    assert identification["criterion"] == IDENTIFICATION_CRITERION
    assert row["per_arm_identification_status"] == identification["verdict"]
    assert identification["identified"] is (
        row["displacement_corrected_ci_95"][0] > 0.0
    )
    assert identification["displacement_corrected_lower_bound_nats"] == pytest.approx(
        row["displacement_corrected_ci_95"][0]
    )

    # The demoted constant travels, and says it decides nothing.
    assert row["screening_status_is_operative"] is False
    assert "decides nothing here" in row["screening_status_note"]
    assert identification["legacy_screening_floor_nats"] == pytest.approx(
        SCREENING_CONTEXT_INFORMATION_NATS
    )
    assert identification["legacy_minimum_context_information_nats"] == pytest.approx(0.30)
    assert "context_identification" in payload["summary"]["operative_gate"]


def test_the_sign_status_is_marked_non_evidential(tmp_path: Path) -> None:
    """It is expected to pass on every arm, so it must never read as a gate."""

    paths = standard_block(tmp_path / "block", "swissprot", seed=2)
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    row = row_for(payload, "progen2-base")
    assert row["sign_status"] == "PASS"
    assert row["sign_status_is_evidential"] is False
    assert "NON-EVIDENTIAL" in row["sign_status_note"]
    assert payload["summary"]["conclusion"]["sign_rule_is_adopted"] is False


# --------------------------------------------------------------------------- #
# 2. Refusals propagate; they are never dropped
# --------------------------------------------------------------------------- #


def test_one_dominant_near_duplicate_group_refuses_the_interval(tmp_path: Path) -> None:
    """A cohort that chains into one component has one effective unit, not many."""

    cohort = synthetic_records(3, n_families=1, per_family=24, length=120, mutations=4)
    reference = synthetic_records(
        4, n_families=40, per_family=2, length=200, mutations=10
    )
    paths = write_block(
        tmp_path / "block", "swissprot", cohort, reference, arms=dict(TWO_ARMS)
    )
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    row = row_for(payload, "progen2-base")
    assert row["refused"] is True
    assert row["n_groups"] == 1
    assert row["n_effective_groups"] < 8
    assert "below the 8-unit floor" in row["refusal_reason"]
    assert row["context_information_nats"] is None
    assert row["screening_status"] == "REFUSED"
    # No interval, so the criterion has nothing to read and no floor is
    # substituted for it.
    assert row["per_arm_identification_status"] == "REFUSED"
    assert row["context_identification"] is None
    assert row["displacement_corrected_ci_95"] is None
    # The arm is still in the table, and the refusal reaches the contrast.
    assert row["arm"] in {entry["arm"] for entry in payload["summary"]["arms"]}
    assert "progen2-base" in payload["summary"]["arms_with_a_refused_interval"]
    contrasts = payload["contrasts"]["within_cohort_paired"]
    assert contrasts and all(entry["refused"] for entry in contrasts)
    assert all(entry["statistics"] is None for entry in contrasts)


def test_an_arm_without_reference_counts_is_refused_rather_than_skipped(
    tmp_path: Path,
) -> None:
    cohort = synthetic_records(5, n_families=12, per_family=2, length=120, mutations=6)
    reference = synthetic_records(
        6, n_families=40, per_family=2, length=200, mutations=10
    )
    statistics = arm_statistics(
        "progen2-base", cohort, reference, unit=1, vocab_size=32, information=0.5
    )
    directory = tmp_path / "block"
    directory.mkdir(parents=True)
    frozen = Cohort(
        name="swissprot",
        kind="protein",
        records=cohort,
        min_symbols=min(len(r) for r in cohort),
        max_symbols=max(len(r) for r in cohort),
    )
    sidecar = directory / f"power_swissprot_{frozen.digest[:12]}.records.npz"
    write_power_records(
        sidecar,
        # reference_counts stripped: the held-out baseline never entered this run
        {"progen2-base": RecordStatistics(
            arm=statistics.arm,
            vocab_size=statistics.vocab_size,
            record_index=statistics.record_index,
            clean_nll_sum=statistics.clean_nll_sum,
            token_count=statistics.token_count,
            n_symbols=statistics.n_symbols,
            target_counts=statistics.target_counts,
        )},
        cohort_digest=frozen.digest,
        reference_digest=None,
        smoothing=None,
        seeds={"cohort_draw": 1},
        max_len=384,
    )
    payload = run_stage(
        base_argv({"sidecar": sidecar}, tmp_path / "out")
    )

    row = row_for(payload, "progen2-base")
    assert row["refused"] is True
    assert "no reference counts" in row["refusal_reason"]
    assert payload["blocks"][0]["arms_refused_at_load"]["progen2-base"]


def test_a_missing_cohort_file_declares_the_singleton_fallback(tmp_path: Path) -> None:
    """Falling back to record-level resampling is allowed only out loud."""

    paths = standard_block(
        tmp_path / "block", "swissprot", seed=7, write_cohort_json=False
    )
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    grouping = payload["blocks"][0]["cohort_grouping"]
    assert grouping["available"] is False
    assert "DECLARED LIMITATION" in grouping["declared_limitation"]
    assert grouping["fallback"].startswith("singleton groups")
    row = row_for(payload, "progen2-base")
    assert row["cohort_grouping"]["available"] is False
    assert row["n_groups"] == row["n_scored_records"]


# --------------------------------------------------------------------------- #
# 3. Pairing is defined within a cohort and nowhere else
# --------------------------------------------------------------------------- #


def test_arms_on_different_cohorts_are_never_paired(tmp_path: Path) -> None:
    first = standard_block(tmp_path / "one", "swissprot", seed=11)
    second = standard_block(tmp_path / "two", "zymctrl_ec", seed=99)
    payload = run_stage(
        [
            "--sidecar",
            str(first["sidecar"]),
            str(second["sidecar"]),
            "--n-bootstrap",
            str(DRAWS),
            "--seed",
            "31337",
            "--out",
            str(tmp_path / "out"),
            "--report-name",
            "report.json",
        ]
    )

    assert {block["cohort_digest"] for block in payload["blocks"]} == {
        payload["blocks"][0]["cohort_digest"],
        payload["blocks"][1]["cohort_digest"],
    }
    assert payload["blocks"][0]["cohort_digest"] != payload["blocks"][1]["cohort_digest"]

    # Every paired contrast lives inside one block.
    for entry in payload["contrasts"]["within_cohort_paired"]:
        assert entry["paired"] is True
        assert entry["common_resample_indices"] is True
        assert entry["block_id"] in {"b0", "b1"}

    across = payload["contrasts"]["cross_cohort_unpaired"]
    assert across, "the same arm on two cohort draws must be contrasted"
    for entry in across:
        assert entry["paired"] is False
        assert entry["common_resample_indices"] is False
        assert entry["left_block_id"] != entry["right_block_id"]
        assert "independence_note" in entry
    assert any(entry["same_arm_across_draws"] for entry in across)

    # The two panels must not share a seed, or the unpaired difference would be
    # neither independent nor paired.
    seeds = [panel["seed"] for block in payload["blocks"] for panel in block["panels"]]
    assert len(set(seeds)) == len(seeds)


def one_draw_two_items(directory: Path, seed: int = 401) -> list[dict[str, Path]]:
    """Two cohort items scoring one draw, the way a dtype-forced split produces it.

    The same records and the same reference under two item names. ``Cohort.digest``
    hashes the records and never the name, so both sidecars carry one cohort digest
    and one reference digest between them.
    """

    cohort = synthetic_records(seed, n_families=12, per_family=2, length=120, mutations=6)
    reference = synthetic_records(
        seed + 1000, n_families=40, per_family=2, length=200, mutations=10
    )
    return [
        write_block(directory, name, cohort, reference, arms={arm: TWO_ARMS[arm]})
        for name, arm in (
            ("swissprot_default_dtype", "progen2-base"),
            ("swissprot_progen2_f32", "progen2-medium"),
        )
    ]


def declared_reading(payload: dict) -> dict:
    """The one ProGen2-base / ProGen2-medium reading, paired or not."""

    entries = [
        entry
        for entry in payload["contrasts"]["within_cohort_paired"]
        + payload["contrasts"]["cross_cohort_unpaired"]
        if {entry["left"], entry["right"]} == {"progen2-base", "progen2-medium"}
    ]
    assert len(entries) == 1, entries
    return entries[0]


def test_two_items_scoring_one_draw_are_one_block_and_are_paired(tmp_path: Path) -> None:
    """The split that puts them in two files is a fact about the producer alone.

    ``--dtype float32`` is process-global, so an arm needing it is declared as its
    own cohort item and scored by its own invocation -- against the same records
    and the same reference as the item beside it. The digests say so, so the two
    arms share a resample index and their contrast is formed inside the iteration.
    """

    first, second = one_draw_two_items(tmp_path / "block")
    payload = run_stage(blocks_argv([first, second], tmp_path / "out"))

    assert len(payload["blocks"]) == 1
    block = payload["blocks"][0]
    assert [entry["cohort_name"] for entry in block["sidecars"]] == [
        "swissprot_default_dtype",
        "swissprot_progen2_f32",
    ]
    assert sorted(block["arms_in_block"]) == ["progen2-base", "progen2-medium"]
    assert len(block["panels"]) == 1

    entry = declared_reading(payload)
    assert entry["paired"] is True
    assert entry["common_resample_indices"] is True
    assert entry["block_id"] == "b0"
    assert not payload["contrasts"]["cross_cohort_unpaired"]
    assert not payload["contrasts"]["same_cohort_unpairable"]

    # One block is not one provenance: each arm still names the file and the
    # cohort item it came from, and both are read under one resample index.
    base = row_for(payload, "progen2-base")
    medium = row_for(payload, "progen2-medium")
    assert base["sidecar"] == str(first["sidecar"])
    assert medium["sidecar"] == str(second["sidecar"])
    assert base["cohort_name"] == "swissprot_default_dtype"
    assert medium["cohort_name"] == "swissprot_progen2_f32"
    assert base["cohort_digest"] == medium["cohort_digest"]
    assert base["panel_id"] == medium["panel_id"]
    assert base["seed"] == medium["seed"]


def test_the_paired_contrast_is_narrower_than_the_unpaired_one(tmp_path: Path) -> None:
    """Two arms of one draw are positively correlated, so pairing must narrow it.

    The unpaired interval carries the sum of the two variances; the paired one
    carries the variance of their difference, which is smaller by twice the
    covariance. These two arms share their cohort draw and their reference and
    differ by a constant per token, so the shared terms cancel and the paired
    difference is exact: everything the unpaired reading reports is variance the
    estimand does not have. The comparison is against the real
    ``unpaired_contrast`` on the same two arms rather than an algebraic
    stand-in for it.
    """

    first, second = one_draw_two_items(tmp_path / "block", seed=411)
    payload = run_stage(blocks_argv([first, second], tmp_path / "out"))
    paired = declared_reading(payload)["statistics"]["information_nats_per_token"]

    block = stage.load_block(
        0,
        [(first["sidecar"], None, None), (second["sidecar"], None, None)],
        requested_arms=None,
        containment=stage.NEAR_DUPLICATE_CONTAINMENT,
        shingle=None,
    )
    left, right = (entry.statistics for entry in block.arms)
    apart = [
        bootstrap_arms([arm], seed=seed, n_bootstrap=DRAWS, contrasts=())
        for arm, seed in ((left, 101), (right, 202))
    ]
    unpaired = unpaired_contrast(apart[0].arms[left.name], apart[1].arms[right.name])
    assert unpaired["paired"] is False
    unpaired_statistics = unpaired["statistics"]["information_nats_per_token"]

    def width(entry: dict) -> float:
        low, high = entry["interval"]
        return high - low

    assert paired["bootstrap_se"] < unpaired_statistics["bootstrap_se"]
    assert width(paired) < width(unpaired_statistics)


def test_one_cohort_scored_against_two_references_is_not_paired(tmp_path: Path) -> None:
    """Identical records are not identical resampling units: the baseline is drawn too."""

    cohort = synthetic_records(421, n_families=12, per_family=2, length=120, mutations=6)
    paths = [
        write_block(
            tmp_path / "block",
            name,
            cohort,
            synthetic_records(seed, n_families=40, per_family=2, length=200, mutations=10),
            arms={arm: TWO_ARMS[arm]},
        )
        for name, arm, seed in (
            ("swissprot_default_dtype", "progen2-base", 1421),
            ("swissprot_progen2_f32", "progen2-medium", 1521),
        )
    ]
    payload = run_stage(blocks_argv(paths, tmp_path / "out"))

    assert len(payload["blocks"]) == 2
    left, right = payload["blocks"]
    assert left["cohort_digest"] == right["cohort_digest"]
    assert left["reference_digest"] != right["reference_digest"]
    assert not payload["contrasts"]["within_cohort_paired"]

    entry = declared_reading(payload)
    assert entry["paired"] is False
    assert entry["common_resample_indices"] is False
    assert "independence_note" in entry


def test_a_sidecar_that_names_no_reference_is_never_merged(tmp_path: Path) -> None:
    """An unverifiable reference is not a shared one, whatever the cohort digest says.

    These two sidecars do hold the same reference records. Neither says so, and a
    pairing that rested on that would rest on nothing, so they stay two blocks.
    """

    cohort = synthetic_records(431, n_families=12, per_family=2, length=120, mutations=6)
    reference = synthetic_records(
        1431, n_families=40, per_family=2, length=200, mutations=10
    )
    paths = [
        write_block(
            tmp_path / "block",
            name,
            cohort,
            reference,
            arms={arm: TWO_ARMS[arm]},
            declare_reference_digest=False,
        )
        for name, arm in (
            ("swissprot_default_dtype", "progen2-base"),
            ("swissprot_progen2_f32", "progen2-medium"),
        )
    ]
    payload = run_stage(blocks_argv(paths, tmp_path / "out"))

    assert len(payload["blocks"]) == 2
    assert payload["blocks"][0]["cohort_digest"] == payload["blocks"][1]["cohort_digest"]
    assert all(block["reference_digest"] is None for block in payload["blocks"])
    assert declared_reading(payload)["paired"] is False


def test_one_arm_scored_twice_on_one_draw_is_refused(tmp_path: Path) -> None:
    """Two readings of one arm on one draw are two measurements, not one panel arm."""

    cohort = synthetic_records(441, n_families=12, per_family=2, length=120, mutations=6)
    reference = synthetic_records(
        1441, n_families=40, per_family=2, length=200, mutations=10
    )
    paths = [
        write_block(tmp_path / "block", name, cohort, reference, arms=dict(TWO_ARMS))
        for name in ("swissprot_default_dtype", "swissprot_progen2_f32")
    ]
    with pytest.raises(ValueError, match="two measurements"):
        run_stage(blocks_argv(paths, tmp_path / "out"))


def test_the_null_control_of_a_merged_block_keeps_its_arm_s_provenance(
    tmp_path: Path,
) -> None:
    """A control is built per arm, so it carries the item that declared that arm.

    Two draws, each scored by two cohort items, is the shape the panel actually
    has. The control borrows the other draw's reference, joins its parent's
    panel, and reports the cohort item its parent came from rather than one name
    chosen for the whole block.
    """

    paths = [
        entry
        for index, seed in enumerate((461, 471))
        for entry in one_draw_two_items(tmp_path / f"draw{index}", seed=seed)
    ]
    payload = run_stage(
        blocks_argv(paths, tmp_path / "out", "--unigram-null-control")
    )

    assert len(payload["blocks"]) == 2
    provenance = control_provenance(payload, "b0", "progen2-medium")
    assert provenance["available"] is True
    assert provenance["independent"] is True
    assert provenance["cohort_name"] == "swissprot_progen2_f32"
    assert provenance["baseline_block_id"] == "b0"
    assert provenance["control_block_id"] == "b1"

    control = row_for(payload, "progen2-medium::unigram-null", "b0")
    parent = row_for(payload, "progen2-medium", "b0")
    assert control["cohort_name"] == parent["cohort_name"] == "swissprot_progen2_f32"
    assert control["panel_id"] == parent["panel_id"]
    assert control["seed"] == parent["seed"]
    assert payload["unigram_null_control"]["n_controls_measured"] == 4


def test_a_merged_block_reproduces_bit_for_bit_at_a_fixed_seed(tmp_path: Path) -> None:
    first, second = one_draw_two_items(tmp_path / "block", seed=451)
    argv = blocks_argv([first, second], tmp_path / "out")

    assert json.dumps(_without_timestamps(run_stage(argv)), sort_keys=True) == json.dumps(
        _without_timestamps(run_stage(argv)), sort_keys=True
    )


def test_the_declared_progen2_contrast_is_named_and_its_sign_tracked(
    tmp_path: Path,
) -> None:
    first = standard_block(tmp_path / "one", "swissprot", seed=13)
    second = standard_block(tmp_path / "two", "swissprot_skip", seed=77)
    payload = run_stage(
        [
            "--sidecar",
            str(first["sidecar"]),
            str(second["sidecar"]),
            "--n-bootstrap",
            str(DRAWS),
            "--seed",
            "4242",
            "--out",
            str(tmp_path / "out"),
            "--report-name",
            "report.json",
        ]
    )

    declared = payload["contrasts"]["declared"]
    assert len(declared) == 1
    entry = declared[0]
    assert {entry["left"], entry["right"]} == {"progen2-base", "progen2-medium"}
    assert entry["ordering_may_be_reported"] is False
    assert "NOT SUPPORTED" in entry["preregistration_note"]
    # One paired reading per block, plus the unpaired cross-block pair.
    assert entry["n_readings"] >= 3
    assert any(reading["paired"] is False for reading in entry["readings"])
    assert entry["sign_is_stable_across_readings"] in (True, False)


# --------------------------------------------------------------------------- #
# 4. Determinism
# --------------------------------------------------------------------------- #


def _without_timestamps(payload: dict) -> dict:
    trimmed = json.loads(json.dumps(payload))
    trimmed["metadata"].pop("started_at_utc")
    trimmed["metadata"].pop("finished_at_utc")
    return trimmed


def test_a_fixed_seed_reproduces_the_report_bit_for_bit(tmp_path: Path) -> None:
    """Same inputs, same seed, same command: the same bytes but for the clock."""

    paths = standard_block(tmp_path / "block", "swissprot", seed=17)
    argv = base_argv(paths, tmp_path / "out")
    first = run_stage(argv)
    second = run_stage(argv)

    assert json.dumps(_without_timestamps(first), sort_keys=True) == json.dumps(
        _without_timestamps(second), sort_keys=True
    )
    assert first["metadata"]["runner_sha256"] == second["metadata"]["runner_sha256"]


# --------------------------------------------------------------------------- #
# 5. E4, the smoothing sweep
# --------------------------------------------------------------------------- #


def test_the_alpha_sweep_stays_inside_the_analytic_bound(tmp_path: Path) -> None:
    """``I(a) - I(0) <= log(1 + aV/R)`` exactly, and the bound grows with V.

    The bound is what the sweep is checked against rather than a monotonicity
    claim: a token whose reference count is below ``R/V`` has its surprisal
    *reduced* by smoothing, so ``I`` is not monotone in ``a`` in general. The
    inequality is, and it is the whole content of the withdrawn figures.
    """

    cohort = synthetic_records(19, n_families=12, per_family=2, length=120, mutations=6)
    reference = synthetic_records(
        20, n_families=40, per_family=2, length=200, mutations=10
    )
    paths = write_block(
        tmp_path / "block",
        "swissprot",
        cohort,
        reference,
        arms={
            "progen2-base": {"unit": 1, "vocab_size": 32, "information": 0.5},
            "protgpt2": {"unit": 2, "vocab_size": 512, "information": 0.5},
        },
    )
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    records = {entry["arm"]: entry for entry in payload["alpha_sensitivity"]["records"]}
    assert set(records) == {"progen2-base", "protgpt2"}
    for arm, entry in records.items():
        assert entry["unsmoothed_information_nats"] is not None, arm
        assert entry["unsmoothed_unavailable_reason"] is None, arm
        for reading in entry["by_alpha"]:
            assert reading["within_analytic_bound"] is True, (arm, reading)
            assert reading["excess_over_unsmoothed_nats"] <= reading["analytic_bound_nats"]
        ascending = sorted(entry["by_alpha"], key=lambda r: r["alpha"])
        bounds = [reading["analytic_bound_nats"] for reading in ascending]
        assert bounds == sorted(bounds)
        assert bounds[0] < bounds[-1]
        for reading in ascending:
            expected = math.log1p(
                reading["alpha"] * entry["vocab_size"] / entry["reference_tokens"]
            )
            assert reading["analytic_bound_nats"] == pytest.approx(expected)

    # The bias is vocabulary-dependent: the wider inventory carries the larger
    # bound at every rung of the ladder, which is why it falls unequally.
    small = {r["alpha"]: r for r in records["progen2-base"]["by_alpha"]}
    large = {r["alpha"]: r for r in records["protgpt2"]["by_alpha"]}
    for alpha in small:
        assert large[alpha]["analytic_bound_nats"] > small[alpha]["analytic_bound_nats"]
    assert (
        records["protgpt2"]["information_range_nats"]
        > records["progen2-base"]["information_range_nats"]
    )


# --------------------------------------------------------------------------- #
# 6. E5, the leakage-removed sensitivity
# --------------------------------------------------------------------------- #


def test_the_leakage_sensitivity_is_unavailable_without_reference_records(
    tmp_path: Path,
) -> None:
    paths = standard_block(tmp_path / "block", "swissprot", seed=23)
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    records = payload["leakage_removed_sensitivity"]["records"]
    assert records
    for entry in records:
        assert entry["available"] is False
        # The reason must name the input that would fix it, not merely say
        # the screen is off -- an operator reads this line to act on it.
        assert "--reference-json" in entry["reason"]
        assert entry["information_nats"] is None


def test_the_leakage_sensitivity_drops_the_leaked_reference_records(
    tmp_path: Path,
) -> None:
    cohort = synthetic_records(29, n_families=12, per_family=2, length=120, mutations=6)
    clean = synthetic_records(30, n_families=40, per_family=2, length=200, mutations=10)
    # Ten reference records that are outright copies of cohort records, which
    # exact-content disjointness would catch but 30-mer leakage is defined to
    # include, plus the clean block.
    reference = clean + [record + record for record in cohort[:10]]
    paths = write_block(
        tmp_path / "block",
        "swissprot",
        cohort,
        reference,
        arms=dict(TWO_ARMS),
        write_reference_json=True,
    )
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    records = {entry["arm"]: entry for entry in payload["leakage_removed_sensitivity"]["records"]}
    entry = records["progen2-base"]
    assert entry["available"] is True
    assert entry["n_reference_records_dropped"] >= 10
    assert entry["information_nats"] is not None
    assert entry["delta_vs_headline_nats"] is not None
    # The reference grouping is the near-duplicate one once the records exist.
    assert payload["blocks"][0]["reference_grouping"]["available"] is True


# --------------------------------------------------------------------------- #
# 6b. The null control, where the true information is zero
# --------------------------------------------------------------------------- #


def control_provenance(payload: dict, block_id: str, arm: str) -> dict:
    matches = [
        entry
        for entry in payload["unigram_null_control"]["provenance"]
        if entry["block_id"] == block_id and entry["arm"] == arm
    ]
    assert len(matches) == 1, (block_id, arm)
    return matches[0]


def test_a_single_block_cannot_build_a_null_control_and_says_so(tmp_path: Path) -> None:
    """One block holds one reference, and one reference cannot be two samples."""

    paths = standard_block(tmp_path / "block", "swissprot", seed=51)
    payload = run_stage(blocks_argv([paths], tmp_path / "out", "--unigram-null-control"))

    entry = control_provenance(payload, "b0", "progen2-base")
    assert entry["available"] is False
    assert entry["independent"] is False
    assert "no other supplied block carries progen2-base" in entry["refusal_reason"]
    assert payload["unigram_null_control"]["n_controls_measured"] == 0
    assert not [row for row in payload["arm_results"] if row["is_unigram_null_control"]]


def test_two_blocks_sharing_a_reference_refuse_the_null_control(tmp_path: Path) -> None:
    """A control on the baseline's own reference is a tautology, and is refused.

    The two blocks here hold different cohorts and the *same* held-out
    reference, which is the case the control has to reject: fitted on it, the
    control's unigram is the baseline's unigram and its ``I`` is zero by algebra
    rather than by measurement. Nothing is emitted for it, and the reason names
    the shared reference.
    """

    reference = synthetic_records(
        900, n_families=40, per_family=2, length=200, mutations=10
    )
    paths = [
        write_block(
            tmp_path / directory,
            "swissprot",
            synthetic_records(seed, n_families=12, per_family=2, length=120, mutations=6),
            reference,
            arms=dict(TWO_ARMS),
        )
        for directory, seed in (("one", 52), ("two", 53))
    ]
    payload = run_stage(blocks_argv(paths, tmp_path / "out", "--unigram-null-control"))

    assert (
        payload["blocks"][0]["reference_digest"]
        == payload["blocks"][1]["reference_digest"]
    )
    entry = control_provenance(payload, "b0", "progen2-base")
    assert entry["available"] is False
    assert "same held-out reference" in entry["refusal_reason"]
    assert "zero identically rather than by measurement" in entry["refusal_reason"]
    assert payload["unigram_null_control"]["n_controls_measured"] == 0
    assert payload["unigram_null_control"]["controls_without_an_independent_source"]


def test_cohort_copies_are_dropped_from_the_borrowed_reference(tmp_path: Path) -> None:
    """A control fitted on the records it scores is a memoriser, not a null.

    The source block's held-out reference here contains this block's whole
    scored cohort verbatim, and its digest still differs from the baseline's, so
    only the exact-content screen sees it. Those records are dropped before the
    control's unigram is fitted -- the same disjointness the producing stage
    asserts within a block -- and the count is reported rather than absorbed.
    """

    cohort = synthetic_records(81, n_families=12, per_family=2, length=120, mutations=6)
    other = synthetic_records(82, n_families=12, per_family=2, length=120, mutations=6)
    reference = synthetic_records(
        83, n_families=40, per_family=2, length=200, mutations=10
    )
    borrowed = synthetic_records(
        87, n_families=40, per_family=2, length=200, mutations=10
    )
    paths = [
        write_block(
            tmp_path / "one",
            "swissprot_a",
            cohort,
            reference,
            arms=dict(TWO_ARMS),
            write_reference_json=True,
        ),
        write_block(
            tmp_path / "two",
            "swissprot_b",
            other,
            borrowed + cohort,
            arms=dict(TWO_ARMS),
            write_reference_json=True,
        ),
    ]
    payload = run_stage(blocks_argv(paths, tmp_path / "out", "--unigram-null-control"))

    screened = control_provenance(payload, "b0", "progen2-base")
    assert screened["available"] is True
    assert screened["cohort_copies_dropped_from_control_reference"] == len(cohort)
    assert screened["control_reference_records_before_screen"] == len(borrowed) + len(
        cohort
    )
    assert screened["control_reference_records"] == len(borrowed)
    # The other direction shares no record, so nothing is dropped there.
    clean = control_provenance(payload, "b1", "progen2-base")
    assert clean["available"] is True
    assert clean["cohort_copies_dropped_from_control_reference"] == 0
    assert clean["exact_overlap_checked"] is True


def test_a_borrowed_reference_that_is_all_cohort_is_refused(tmp_path: Path) -> None:
    """Nothing survives the screen, so there is no independent sample to fit."""

    cohort = synthetic_records(84, n_families=12, per_family=2, length=120, mutations=6)
    reference = synthetic_records(
        85, n_families=40, per_family=2, length=200, mutations=10
    )
    paths = [
        write_block(
            tmp_path / "one",
            "swissprot_a",
            cohort,
            reference,
            arms=dict(TWO_ARMS),
            write_reference_json=True,
        ),
        write_block(
            tmp_path / "two",
            "swissprot_b",
            synthetic_records(86, n_families=12, per_family=2, length=120, mutations=6),
            cohort,
            arms=dict(TWO_ARMS),
            write_reference_json=True,
        ),
    ]
    payload = run_stage(blocks_argv(paths, tmp_path / "out", "--unigram-null-control"))

    entry = control_provenance(payload, "b0", "progen2-base")
    assert entry["available"] is False
    assert "disjoint sample survives" in entry["refusal_reason"]
    assert entry["cohort_copies_dropped_from_control_reference"] == len(cohort)


def test_the_null_control_measures_a_known_zero_inside_the_same_panel(
    tmp_path: Path,
) -> None:
    """The criteria are watched at a value that is known rather than estimated.

    The control joins the panel of the arm it is built from, so it is measured
    by the same call under the same resample indices; adding it must leave every
    real arm's statistics untouched. At a true zero the floor behaves correctly
    by refusing, and whatever the sign rule does is counted rather than
    described.
    """

    # The reference records are written too, so the leakage screen runs and the
    # control travels the whole stage rather than only the part it was added to.
    paths = [
        standard_block(tmp_path / "one", "swissprot_a", seed=61, write_reference_json=True),
        standard_block(tmp_path / "two", "swissprot_b", seed=62, write_reference_json=True),
    ]
    plain = run_stage(blocks_argv(paths, tmp_path / "plain"))
    payload = run_stage(
        blocks_argv(paths, tmp_path / "control", "--unigram-null-control")
    )

    entry = control_provenance(payload, "b0", "progen2-base")
    assert entry["available"] is True
    assert entry["independent"] is True
    assert entry["baseline_block_id"] == "b0"
    assert entry["control_block_id"] == "b1"
    assert entry["baseline_reference_digest"] != entry["control_reference_digest"]
    assert entry["true_information_nats"] == 0.0

    control_row = row_for(payload, "progen2-base::unigram-null", "b0")
    arm_row_ = row_for(payload, "progen2-base", "b0")
    assert control_row["is_unigram_null_control"] is True
    assert arm_row_["is_unigram_null_control"] is False
    for field in ("panel_id", "seed", "n_bootstrap", "confidence", "resampling_unit"):
        assert control_row[field] == arm_row_[field], field
    assert control_row["unit_floor"] == arm_row_["unit_floor"]

    # A known zero: the floor refuses it, and the departure is far below both the
    # floor and the arm the control was built beside.
    assert control_row["screening_status"] == "FAIL"
    assert abs(control_row["context_information_nats"]) < 0.05
    assert arm_row_["context_information_nats"] > 0.4

    summary = payload["unigram_null_control"]
    reading = next(
        r
        for r in summary["readings"]
        if r["block_id"] == "b0" and r["arm"] == "progen2-base"
    )
    assert reading["true_information_nats"] == 0.0
    assert reading["interval_covers_zero"] is True
    assert reading["cohort_copies_dropped_from_control_reference"] == 0
    assert reading["baseline_control_reference_exact_overlap"] == 0
    assert summary["independence_check"]["n_readings_that_dropped_cohort_copies"] == 0
    assert summary["independence_check"]["max_cohort_copies_dropped"] == 0
    assert reading["measured_information_nats"] == control_row["context_information_nats"]
    criteria = summary["criteria_at_a_known_zero"]
    assert criteria["n_floor_pass_readings"] == 0
    assert criteria["floor_behaves"] is True
    assert criteria["sign_rule_is_adopted"] is False
    # Counted over readings and not over arm names: one arm carries one reading
    # per block, and collapsing them reports a rate as many times too small as
    # there are blocks.
    assert criteria["n_readings"] == len(summary["readings"]) == 4
    assert criteria["n_sign_pass_readings"] == sum(
        reading["sign_status"] == "PASS" for reading in summary["readings"]
    )
    assert criteria["observed_sign_pass_rate"] == pytest.approx(
        criteria["n_sign_pass_readings"] / len(summary["readings"])
    )
    assert criteria["expected_false_pass_rate_at_a_true_zero"] == pytest.approx(0.025)
    # The mechanism the sign rule would read at a zero is carried on the record.
    assert reading["bootstrap_bias_nats"] == control_row["bootstrap_bias"]
    assert reading["interval_lies_entirely_above_the_point"] == (
        control_row["bootstrap_ci_95"][0] > control_row["context_information_nats"]
    )

    # Joining the panel must not move the panel.
    def statistics_of(report: dict) -> dict:
        return {
            (row["block_id"], row["arm"]): row["statistics"]
            for row in report["arm_results"]
            if not row["is_unigram_null_control"]
        }

    assert statistics_of(plain) == statistics_of(payload)

    # A designed zero is not a panel verdict, so it stays out of the panel table's
    # conclusion and is named as what it is.
    assert "progen2-base::unigram-null" in payload["summary"]["unigram_null_control_arms"]
    assert (
        "progen2-base::unigram-null"
        not in payload["summary"]["arms_where_the_sign_rule_disagrees_with_the_floor"]
    )
    assert plain["unigram_null_control"]["requested"] is False

    # The leakage screen reaches the control and says which reference it screened.
    leakage = next(
        record
        for record in payload["leakage_removed_sensitivity"]["records"]
        if record["arm"] == "progen2-base::unigram-null" and record["block_id"] == "b0"
    )
    assert leakage["available"] is True
    assert leakage["is_unigram_null_control"] is True
    assert "not screened here" in leakage["screened_reference"]


def test_the_null_control_reproduces_bit_for_bit_at_a_fixed_seed(
    tmp_path: Path,
) -> None:
    paths = [
        standard_block(tmp_path / "one", "swissprot_a", seed=71),
        standard_block(tmp_path / "two", "swissprot_b", seed=72),
    ]
    argv = blocks_argv(paths, tmp_path / "out", "--unigram-null-control")
    first = run_stage(argv)
    second = run_stage(argv)

    assert json.dumps(_without_timestamps(first), sort_keys=True) == json.dumps(
        _without_timestamps(second), sort_keys=True
    )
    assert first["unigram_null_control"]["n_controls_measured"] == 4


# --------------------------------------------------------------------------- #
# 7. Between-block variance
# --------------------------------------------------------------------------- #


def test_two_blocks_report_a_spread_and_refuse_an_interval(tmp_path: Path) -> None:
    first = standard_block(tmp_path / "one", "swissprot", seed=31)
    second = standard_block(tmp_path / "two", "swissprot_skip", seed=131)
    payload = run_stage(
        [
            "--sidecar",
            str(first["sidecar"]),
            str(second["sidecar"]),
            "--n-bootstrap",
            str(DRAWS),
            "--seed",
            "555",
            "--out",
            str(tmp_path / "out"),
            "--report-name",
            "report.json",
        ]
    )

    records = {entry["arm"]: entry for entry in payload["between_block_variance"]}
    entry = records["progen2-base"]
    assert entry["n_blocks"] == 2
    assert entry["degrees_of_freedom"] == 1
    assert entry["between_block_sd_nats"] is not None
    assert entry["between_block_interval"] is None
    assert "not an interval" in entry["between_block_interval_refusal"]
    assert entry["folded_into_the_bootstrap_interval"] is False
    assert entry["blocks_disjoint"]["checked"] is True
    assert entry["blocks_disjoint"]["all_disjoint"] is True


# --------------------------------------------------------------------------- #
# 8. The summary table names what is absent
# --------------------------------------------------------------------------- #


def test_an_arm_with_no_record_is_named_absent_rather_than_omitted(
    tmp_path: Path,
) -> None:
    paths = standard_block(tmp_path / "block", "swissprot", seed=37)
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    entries = {entry["arm"]: entry for entry in payload["summary"]["arms"]}
    # Every declared panel arm is accounted for, measured or not.
    assert {"bygpt5-small-en", "bygpt5-base-en", "gpt2-large"} <= set(entries)
    absent = entries["bygpt5-small-en"]
    assert absent["present"] is False
    assert absent["status"] == "ABSENT"
    assert "not a pass" in absent["absent_reason"]
    assert "bygpt5-small-en" in payload["summary"]["arms_with_no_record"]
    assert entries["progen2-base"]["present"] is True
    assert entries["progen2-base"]["status"] == "PASS"
    conclusion = payload["summary"]["conclusion"]
    assert conclusion["sign_rule_is_adopted"] is False
    assert isinstance(conclusion["interval_alters_the_unmeasurable_set"], bool)
    assert isinstance(conclusion["interval_statement"], str)


def test_eight_blocks_earn_a_between_block_interval_that_is_still_separate(
    tmp_path: Path,
) -> None:
    """K = 8 is where the component stops being a one-degree-of-freedom quantity.

    It is still reported beside the bootstrap interval and never inside it: a
    variance component estimated from K blocks does not belong in a percentile
    interval computed from one.
    """

    blocks = [
        standard_block(tmp_path / f"block{index}", f"swissprot_{index}", seed=200 + index)
        for index in range(8)
    ]
    argv = ["--sidecar", *[str(block["sidecar"]) for block in blocks]]
    argv += [
        "--n-bootstrap",
        str(DRAWS),
        "--seed",
        "909",
        "--alpha-sweep",
        "1.0",
        "--out",
        str(tmp_path / "out"),
        "--report-name",
        "report.json",
    ]
    payload = run_stage(argv)

    entry = {row["arm"]: row for row in payload["between_block_variance"]}["progen2-base"]
    assert entry["n_blocks"] == 8
    assert entry["degrees_of_freedom"] == 7
    assert entry["between_block_interval_refusal"] is None
    low, high = entry["between_block_interval"]
    assert low < entry["mean_nats"] < high
    assert entry["folded_into_the_bootstrap_interval"] is False
    assert entry["blocks_disjoint"]["all_disjoint"] is True


# --------------------------------------------------------------------------- #
# 9. The held-out validation of the identification floor
# --------------------------------------------------------------------------- #


def null_reading(
    block_id: str,
    arm: str,
    vocab_size: int,
    value: float,
    *,
    control_block_id: str | None = None,
    half_width: float = 0.002,
    se: float = 0.001,
    displacement: float = 0.0015,
) -> dict:
    """One control reading in the shape ``null_control_summary`` publishes.

    ``displacement`` is the L34 lift of the interval off the point estimate, so
    the corrected interval is the same interval shifted down by it and the
    reference-held-fixed interval is a *narrower* one -- the three objects the
    held-out validation has to keep apart.
    """

    return {
        "arm": arm,
        "block_id": block_id,
        "vocab_size": vocab_size,
        "control_block_id": control_block_id or block_id,
        "measured_information_nats": value,
        "bootstrap_ci_95": [value - half_width + displacement, value + half_width + displacement],
        "displacement_corrected_ci_95": [value - half_width, value + half_width],
        "reference_held_fixed_ci_95": [value - 0.6 * half_width, value + 0.6 * half_width],
        "reference_resampling_displacement": displacement,
        "cohort_only_bootstrap_se": 0.6 * se,
        "bootstrap_se": se,
    }


def real_reading(
    block_id: str,
    arm: str,
    value: float,
    *,
    se: float = 0.004,
    half_width: float = 0.008,
) -> dict:
    return {
        "arm": arm,
        "block_id": block_id,
        "is_unigram_null_control": False,
        "context_information_nats": value,
        "bootstrap_se": se,
        "displacement_corrected_ci_95": [value - half_width, value + half_width],
        "per_arm_identification_status": "PASS" if value - half_width > 0.0 else "FAIL",
    }


#: Three quiet blocks and one noisy one. The noisy block is what the panel-wide
#: threshold is chosen against, so a fold that holds it out must choose a
#: different threshold -- which is the whole thing a held-out validation exists
#: to expose.
QUIET_BLOCKS = ("b0", "b1", "b2")
NOISY_BLOCK = "b3"


def one_noisy_block() -> list[dict]:
    # The standard errors are the panel's own order of magnitude, so the Fieller
    # denominator condition is exercised where it actually sits rather than
    # against a precision no bootstrap on 200 records produces.
    readings = [
        null_reading(block, arm, vocab, 0.001, control_block_id="b0", se=0.004)
        for block in QUIET_BLOCKS
        for arm, vocab in (("small-vocab-arm", 10), ("large-vocab-arm", 20))
    ]
    readings += [
        null_reading(
            NOISY_BLOCK, "small-vocab-arm", 10, 0.030, control_block_id="b0", se=0.012
        ),
        null_reading(
            NOISY_BLOCK, "large-vocab-arm", 20, 0.032, control_block_id="b0", se=0.012
        ),
    ]
    return readings


def scheme_of(validation: dict, name: str) -> dict:
    matches = [entry for entry in validation["schemes"] if entry["scheme"] == name]
    assert len(matches) == 1, name
    return matches[0]


def fold_of(scheme: dict, unit: str) -> dict:
    matches = [fold for fold in scheme["folds"] if fold["held_out_unit"] == unit]
    assert len(matches) == 1, unit
    return matches[0]


def test_the_recalibrated_threshold_is_derived_without_the_unit_it_is_scored_on(
    tmp_path: Path,
) -> None:
    """The load-bearing assertion: cross-fitting must actually withhold.

    A validation that leaves the extreme unit in the derivation reports the
    in-sample answer under a different name, and would do so silently. Here the
    panel-wide threshold is set by one noisy block; the fold that holds that
    block out must therefore derive a *lower* threshold and must then fail on the
    readings it never saw.
    """

    readings = one_noisy_block()
    panel_tau = stage._smallest_admissible_tau(
        [reading["measured_information_nats"] for reading in readings],
        stage.HELD_OUT_TARGET_FALSE_POSITIVE_RATE,
    )
    assert panel_tau == pytest.approx(0.035)

    validation = stage.held_out_threshold_validation(
        readings, [], threshold=0.05, confidence=0.95
    )
    scheme = scheme_of(validation, "leave_one_cohort_draw_out")
    fold = fold_of(scheme, NOISY_BLOCK)
    assert fold["recalibrated_tau_nats"]["point_rule"] == pytest.approx(0.005)
    assert fold["recalibrated_tau_nats"]["point_rule"] < panel_tau
    assert fold["held_out_false_positive_rate"]["recalibrated_point_rule"] == 1.0
    assert fold["n_calibration_readings"] == 6
    assert fold["n_held_out_readings"] == 2

    pooled = scheme["pooled_held_out_false_positive_rate"]
    assert pooled["recalibrated_point_rule"]["n_false_positives"] == 2
    assert pooled["recalibrated_point_rule"]["rate_over_readings"] == pytest.approx(0.25)
    assert pooled["recalibrated_point_rule"]["meets_the_target_over_readings"] is False
    assert (
        validation["conclusion"][
            "recalibrated_threshold_meets_its_own_target_out_of_sample"
        ]
        is False
    )


def test_the_shipped_floor_is_scored_on_the_very_readings_the_fold_threshold_is(
    tmp_path: Path,
) -> None:
    """One set of held-out readings, two rules -- never two sets, one rule each."""

    validation = stage.held_out_threshold_validation(
        one_noisy_block(), [], threshold=0.05, confidence=0.95
    )
    for name in (
        "leave_one_cohort_draw_out",
        "leave_one_tokenisation_class_out",
    ):
        pooled = scheme_of(validation, name)["pooled_held_out_false_positive_rate"]
        denominators = {entry["n_held_out_readings"] for entry in pooled.values()}
        assert denominators == {8}, name
        assert pooled["shipped_point_rule"]["n_false_positives"] == 0
        assert pooled["shipped_interval_rule"]["n_false_positives"] == 0
        assert (
            pooled["shipped_fieller_denominator_admissibility"]["n_false_positives"]
            == 0
        )
    assert validation["conclusion"]["shipped_floor_admits_no_held_out_null"] is True


def test_a_zero_held_out_rate_is_reported_with_the_bound_it_actually_carries() -> None:
    """No false positive is not a false-positive rate of zero.

    A rate of 0 over a few dozen units is consistent with a true rate an order of
    magnitude above the target, and reporting the point estimate alone would
    repeat, one level up, the mistake this validation exists to correct.
    """

    validation = stage.held_out_threshold_validation(
        one_noisy_block(), [], threshold=0.05, confidence=0.95
    )
    pooled = scheme_of(validation, "leave_one_cohort_draw_out")[
        "pooled_held_out_false_positive_rate"
    ]["shipped_point_rule"]
    assert pooled["rate_over_readings"] == 0.0
    assert pooled["exact_upper_95_over_readings"] > 0.0
    # Here every reading is its own measurement, so the two bounds coincide; the
    # distinct one is never the narrower of the pair.
    assert pooled["exact_upper_95_over_distinct_measurements"] == pytest.approx(
        pooled["exact_upper_95_over_readings"]
    )
    assert stage._exact_upper_bound(0, 8, 0.95) == pytest.approx(
        1.0 - 0.05 ** (1.0 / 8.0)
    )
    assert stage._exact_upper_bound(0, 0, 0.95) is None


def test_controls_sharing_a_block_and_a_vocabulary_are_one_measurement() -> None:
    """The unit count is measured, not assumed, and duplicates collapse into it."""

    readings = [
        null_reading("b0", "gpt2", 50257, 0.001),
        null_reading("b0", "gpt2-large", 50257, 0.001),
        null_reading("b0", "llama", 128256, 0.002),
        null_reading("b1", "gpt2", 50257, 0.003),
        null_reading("b1", "gpt2-large", 50257, 0.003),
        null_reading("b1", "llama", 128256, 0.004),
    ]
    equivalence = stage.null_reading_equivalence(readings)
    assert equivalence["n_readings"] == 6
    assert equivalence["n_distinct_measurements"] == 4
    assert equivalence["n_cohort_draws"] == 2
    labels = {entry["label"]: entry["arms"] for entry in equivalence["tokenisation_classes"]}
    assert labels == {"gpt2+gpt2-large": ["gpt2", "gpt2-large"], "llama": ["llama"]}
    assert equivalence["arm_to_tokenisation_class"]["gpt2"] == "gpt2+gpt2-large"

    # And the collapse is why leaving one *arm* out would measure nothing: gpt2
    # and gpt2-large are one fold unit, not two.
    validation = stage.held_out_threshold_validation(
        readings, [], threshold=0.05, confidence=0.95
    )
    scheme = scheme_of(validation, "leave_one_tokenisation_class_out")
    assert scheme["n_folds"] == 2

    # Duplicated readings must not buy resolution they do not carry: the bound
    # read over distinct measurements is the wider of the two.
    pooled = scheme["pooled_held_out_false_positive_rate"]["shipped_point_rule"]
    assert pooled["n_held_out_distinct_measurements"] == 4
    assert pooled["n_held_out_readings"] == 6
    assert (
        pooled["exact_upper_95_over_distinct_measurements"]
        > pooled["exact_upper_95_over_readings"]
    )


def test_controls_of_one_block_and_vocabulary_that_disagree_are_refused() -> None:
    """The premise the unit accounting rests on is checked rather than trusted."""

    readings = [
        null_reading("b0", "gpt2", 50257, 0.001),
        null_reading("b0", "gpt2-large", 50257, 0.009),
    ]
    with pytest.raises(SystemExit) as raised:
        stage.null_reading_equivalence(readings)
    assert "do not carry the same reading" in str(raised.value)


def test_a_scheme_with_one_unit_is_refused_rather_than_scored() -> None:
    """A unit cannot be held out of its own derivation, and saying so is the point.

    A fold built on a single unit would compare a threshold with itself and
    report a held-out rate that is nothing of the kind.
    """

    readings = [
        null_reading("b0", "small-vocab-arm", 10, 0.001),
        null_reading("b0", "large-vocab-arm", 20, 0.002),
    ]
    validation = stage.held_out_threshold_validation(
        readings, [], threshold=0.05, confidence=0.95
    )
    draws = scheme_of(validation, "leave_one_cohort_draw_out")
    assert draws["refused"] is True
    assert "at least two units" in draws["refusal_reason"]
    assert "folds" not in draws
    classes = scheme_of(validation, "leave_one_tokenisation_class_out")
    assert classes["refused"] is False
    assert classes["n_folds"] == 2


def test_too_few_measured_controls_refuse_the_whole_validation() -> None:
    validation = stage.held_out_threshold_validation(
        [null_reading("b0", "one-arm", 10, 0.001)], [], threshold=0.05, confidence=0.95
    )
    assert validation["refused"] is True
    assert "at least two units" in validation["refusal_reason"]
    assert "schemes" not in validation


def test_a_real_arm_between_the_two_thresholds_is_named_as_changing_verdict() -> None:
    """The agreement statement must be capable of being false.

    On the panel no arm sits between a fold's threshold and the shipped floor, so
    the reported agreement is worth exactly as much as the check's ability to
    detect a disagreement; this puts one there.
    """

    readings = one_noisy_block()
    rows = [
        real_reading(NOISY_BLOCK, "borderline-arm", 0.020),
        real_reading(NOISY_BLOCK, "clearly-measurable-arm", 1.400),
        real_reading("b0", "clearly-measurable-arm", 1.400),
    ]
    validation = stage.held_out_threshold_validation(
        readings, rows, threshold=0.05, confidence=0.95
    )
    scheme = scheme_of(validation, "leave_one_cohort_draw_out")
    fold = fold_of(scheme, NOISY_BLOCK)
    changed = fold["real_arm_verdicts"]["readings_that_change_verdict"]
    assert [entry["arm"] for entry in changed] == ["borderline-arm"]
    assert changed[0]["verdict_under_the_recalibrated_threshold"] == "PASS"
    assert changed[0]["verdict_under_the_shipped_floor"] == "FAIL"
    assert scheme["real_arm_verdict_agreement"]["agrees_everywhere"] is False
    assert validation["conclusion"]["real_arm_verdicts_are_unchanged"] is False

    # A real arm on a unit no fold holds out is named rather than dropped.
    classes = scheme_of(validation, "leave_one_tokenisation_class_out")
    assert classes["real_arm_verdict_agreement"]["n_readings_on_no_held_out_unit"] == 3


def test_the_buffered_scheme_withholds_the_block_the_control_borrowed_from() -> None:
    """The rotation couples a held-out reading to a calibration block's reference."""

    readings = [
        null_reading(block, "one-arm", 10, 0.001, control_block_id=source)
        for block, source in (("b0", "b1"), ("b1", "b2"), ("b2", "b3"), ("b3", "b0"))
    ]
    validation = stage.held_out_threshold_validation(
        readings, [], threshold=0.05, confidence=0.95
    )
    plain = fold_of(scheme_of(validation, "leave_one_cohort_draw_out"), "b0")
    assert plain["calibration_units_withheld_as_a_buffer"] == []
    assert plain["n_calibration_readings"] == 3

    buffered = fold_of(
        scheme_of(
            validation,
            "leave_one_cohort_draw_out_buffered_against_the_control_rotation",
        ),
        "b0",
    )
    assert buffered["calibration_units_withheld_as_a_buffer"] == ["b1", "b3"]
    assert buffered["n_calibration_readings"] == 1


def test_the_validation_needs_the_null_control_and_says_so() -> None:
    with pytest.raises(SystemExit) as raised:
        stage.parse_args(
            ["--sidecar", "x.records.npz", "--held-out-threshold-validation"]
        )
    assert "--unigram-null-control" in str(raised.value)


def test_the_stage_carries_the_validation_end_to_end(tmp_path: Path) -> None:
    """Through the real stage on real sidecars, not over hand-written readings.

    Two blocks of two arms sharing one vocabulary: the cohort-draw scheme has the
    two units it needs, and the tokenisation-class scheme has one and must refuse
    rather than report a rate.
    """

    paths = [
        standard_block(tmp_path / "one", "swissprot_a", seed=71, write_reference_json=True),
        standard_block(tmp_path / "two", "swissprot_b", seed=72, write_reference_json=True),
    ]
    payload = run_stage(
        blocks_argv(
            paths,
            tmp_path / "out",
            "--unigram-null-control",
            "--held-out-threshold-validation",
        )
    )
    validation = payload["held_out_threshold_validation"]
    assert validation["requested"] is True
    assert validation["refused"] is False
    assert validation["shipped_floor_nats"] == pytest.approx(
        payload["summary"]["threshold_nats"]
    )

    accounting = validation["unit_accounting"]
    assert accounting["n_cohort_draws"] == 2
    assert accounting["n_distinct_measurements"] == 2
    assert accounting["n_readings"] == 4

    draws = scheme_of(validation, "leave_one_cohort_draw_out")
    assert draws["refused"] is False
    assert draws["n_folds"] == 2
    pooled = draws["pooled_held_out_false_positive_rate"]
    assert pooled["shipped_point_rule"]["n_false_positives"] == 0
    assert pooled["shipped_point_rule"]["exact_upper_95_over_readings"] > 0.0
    assert draws["real_arm_verdict_agreement"]["n_held_out_readings"] == 4

    classes = scheme_of(validation, "leave_one_tokenisation_class_out")
    assert classes["refused"] is True

    assert validation["conclusion"]["shipped_floor_admits_no_held_out_null"] is True
    assert "0.05 nats/token floor" in validation["conclusion"]["statement"]


def test_the_validation_is_absent_unless_it_is_asked_for(tmp_path: Path) -> None:
    paths = [
        standard_block(tmp_path / "one", "swissprot_a", seed=81, write_reference_json=True),
        standard_block(tmp_path / "two", "swissprot_b", seed=82, write_reference_json=True),
    ]
    payload = run_stage(
        blocks_argv(paths, tmp_path / "out", "--unigram-null-control")
    )
    assert payload["held_out_threshold_validation"]["requested"] is False
