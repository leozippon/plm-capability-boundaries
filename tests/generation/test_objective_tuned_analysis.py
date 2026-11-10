"""The objective-tuned downstream analysis, exercised on this repository's own measurements.

The fixture campaign is built from real triples -- a generated sequence, its real
ESMFold2 mean CA-pLDDT and its real identity to a natural reference -- taken from
the completed EXP-R2-233 evidence run, and written through the free-length lane's
own ``generate``/``write_attempt_ledger`` so the ledgers, the stop statuses and
the denominator rule are the production ones rather than a hand-made shape. The
attempt count is reduced for the test; nothing else about the lane is.

Only the campaign *layout* is synthetic. Every pLDDT, every identity and every
sequence the analysis reads here was measured by this repository.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS = REPO_ROOT / "scripts/capability/generation"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyse_objective_tuned as aot  # noqa: E402

from src.capability.context.homology import assign_stratum  # noqa: E402
from src.capability.core.io import sha256_file  # noqa: E402
from src.capability.extensions import bidirectional as bd  # noqa: E402
from src.capability.generation import objective_tuned_analysis as ota  # noqa: E402
from src.capability.generation import unconditional_generation as ug  # noqa: E402
from src.capability.generation.structure_evidence import sequence_digest  # noqa: E402

EVIDENCE = REPO_ROOT / "results/R6/generation_evidence_20260905"
PREDICTIONS = EVIDENCE / "structure/esmfold2_20260919155754/main/predictions.jsonl"
SUBSET = EVIDENCE / "main_subset.jsonl"
CONDITIONED = REPO_ROOT / "results/R6/conditioned_generation_20260826"
#: The frozen identity-over-query search of this pair's own conditioned cells,
#: produced for the 2026-11-10 amendment by
#: scripts/capability/generation/build_within_arm_novelty_cohort.py and
#: scripts/capability/evaluation/search_generated_novelty.py.
WITHIN_ARM_NOVELTY = REPO_ROOT / "results/R6/objective_tuned_within_arm_novelty_20261110"
#: A search of a different campaign, which overlaps these cells in sixteen
#: sequences. It is the partial-coverage case, not this pair's instrument.
OTHER_CAMPAIGN_NOVELTY = REPO_ROOT / "results/R6/generation_novelty_20261008"
KMERS = REPO_ROOT / "data/kmer_background/uniref50_high_order"

#: Sequences per cell, by stratum, and the empty attempts beside them. The
#: near-duplicate allocation is deliberately below the resampling floor on a
#: single-cell tuned stream and above it on a four-cell base stream, so one run
#: exercises both dispositions.
PER_CELL = {ota.STRATA[0]: 30, ota.STRATA[1]: 20, ota.STRATA[2]: 18, ota.STRATA[3]: 6}
EMPTIES = 6
ATTEMPTS = sum(PER_CELL.values()) + EMPTIES


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def triples() -> dict[tuple[str, str], list[tuple[str, float, float]]]:
    """Real (sequence, mean CA-pLDDT, identity) triples, by identity stratum and half.

    Within each stratum the real triples are split at the median measured pLDDT,
    and the base arms draw from the lower half while the tuned arms draw from the
    upper. That is a test construction, not a finding: it gives the fixture a
    tuned-minus-base gain of a known sign so the estimator's resolved branches
    are exercised, while every pLDDT, identity and sequence stays a measurement
    this repository made.
    """

    identity: dict[str, float] = {}
    for row in read_jsonl(SUBSET):
        status = row.get("reference_search_status")
        if status == "aligned_coverage_unavailable" and row.get("reference_identity") is not None:
            identity[row["id"]] = float(row["reference_identity"])
        elif status == "no_reported_alignment":
            identity[row["id"]] = 0.0
    grouped: dict[str, list[tuple[str, float, float]]] = {name: [] for name in ota.STRATA}
    for row in read_jsonl(PREDICTIONS):
        structure = row.get("structure") or {}
        if structure.get("status") != "ok" or row["id"] not in identity:
            continue
        value = identity[row["id"]]
        grouped[assign_stratum(value)].append(
            (str(row["sequence"]), float(structure["mean_ca_plddt"]), value))
    halves: dict[tuple[str, str], list[tuple[str, float, float]]] = {}
    for name, rows in grouped.items():
        assert len(rows) >= 2 * PER_CELL[name], f"{name} has too few real triples: {len(rows)}"
        rows.sort(key=lambda row: (row[1], row[0]))
        middle = len(rows) // 2
        halves[(name, "lower")] = sorted(rows[:middle])
        halves[(name, "upper")] = sorted(rows[middle:])
    return halves


def build_campaign(root: Path, triples, monkeypatch, *, arms=None, drop_plddt: bool = False,
                   cohort_shape: str = "sequence_sha256", corrupt_digest: bool = False):
    """Write a free-length campaign through the lane's own writers, and its joins."""

    monkeypatch.setitem(ug.POLICY, "attempts", ATTEMPTS)
    cursor = dict.fromkeys(triples, 0)
    used: dict[str, tuple[str, float, float]] = {}
    seed = 20264011000

    def take(stratum: str, half: str, count: int) -> list[tuple[str, float, float]]:
        rows = triples[(stratum, half)]
        start = cursor[(stratum, half)]
        cursor[(stratum, half)] = (start + count) % len(rows)
        return [rows[(start + offset) % len(rows)] for offset in range(count)]

    campaign = root / "campaign"
    campaign.mkdir(parents=True)
    wanted = arms
    # Every free-length pair, which is the sixty-cell campaign: the three
    # matched pairs and the declared narrowing control beside them.
    for pair in ota.PRIMARY_PAIRS + ota.SECONDARY_PAIRS:
        for arm in pair.arms:
            if wanted is not None and arm not in wanted:
                continue
            role = "base" if arm == pair.base else "tuned"
            for stream in range(1, aot.builder.STREAMS + 1):
                for replicate in range(aot.builder.CELLS_PER_STREAM[role]):
                    half = "lower" if role == "base" else "upper"
                    chosen = [row for stratum, count in PER_CELL.items()
                              for row in take(stratum, half, count)]
                    for row in chosen:
                        used[sequence_digest(row[0])] = row
                    close = ug.close_token_for(arm)
                    raws = [sequence + ("" if index % 10 == 0 else close)
                            for index, (sequence, _, _) in enumerate(chosen)]
                    raws.extend([""] * EMPTIES)
                    seed += aot.builder.SEED_STEP
                    cell = campaign / f"{aot.builder.KEY}_{arm}_s{stream}_r{replicate}"
                    ug.generate(arm, n=len(raws), seed=seed, output_dir=cell,
                                generate_fn=lambda n, **kwargs: list(raws))

    structure = root / "predictions.jsonl"
    with structure.open("w", encoding="utf-8") as handle:
        for digest, (sequence, plddt, _) in sorted(used.items()):
            block = ({"status": "ok"} if drop_plddt
                     else {"status": "ok", "mean_ca_plddt": plddt})
            handle.write(json.dumps({"sequence_sha256": digest, "sequence": sequence,
                                     "role": "generation", "paired_id": None,
                                     "structure": block}, sort_keys=True) + "\n")

    novelty = root / "novelty"
    novelty.mkdir()
    cohort = novelty / "cohort.jsonl"
    rows, cohort_rows = [], []
    for index, (digest, (sequence, _, value)) in enumerate(sorted(used.items())):
        # Two real cohort shapes exist in this repository: the evaluation cohort
        # carries a sequence_sha256 column, and the campaign's own fold cohort
        # keys its rows by the sequence digest and carries no such column.
        identifier = digest if cohort_shape == "digest_id" else f"nov_{index:05d}"
        row = {"id": identifier, "sequence": sequence, "length": len(sequence),
               "role": "generation"}
        if cohort_shape == "sequence_sha256":
            row["sequence_sha256"] = "0" * 64 if corrupt_digest and index == 0 else digest
        cohort_rows.append(row)
        rows.append({"id": identifier, "nearest_corpus_identity": value,
                     "identity_stratum": assign_stratum(value), "searched": True,
                     "had_any_hit": value > 0.0})
    cohort.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in cohort_rows),
                      encoding="utf-8")
    (novelty / "generated_novelty.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (novelty / "generated_novelty.json").write_text(
        json.dumps({"cohort": str(cohort), "cohort_sha256": sha256_file(cohort),
                    "n_searched": len(rows)}, sort_keys=True), encoding="utf-8")
    return campaign, structure, novelty


def invoke(root: Path, campaign: Path, structure: Path, novelty: Path, *extra: str):
    return aot.main(["--campaign-root", str(campaign), "--structure", str(structure),
                     "--novelty", str(novelty), "--kmer-background", str(KMERS),
                     "--out", str(root / "out"), *extra])


@pytest.fixture(scope="module")
def built(triples, tmp_path_factory):
    root = tmp_path_factory.mktemp("objective_tuned")
    with pytest.MonkeyPatch.context() as monkeypatch:
        campaign, structure, novelty = build_campaign(root, triples, monkeypatch)
        record = invoke(root, campaign, structure, novelty)
    return {"root": root, "campaign": campaign, "structure": structure, "novelty": novelty,
            "record": record}


@pytest.fixture(scope="module")
def analysis(built):
    return built["record"]


# ------------------------------------------------------- the declared families

def test_primary_family_is_declared_and_asserted():
    assert ota.PRIMARY_FAMILY_SIZE == 12
    assert len(ota.PRIMARY_PAIRS) == 3 and len(ota.STRATA) == 4
    assert [pair.tuned for pair in ota.PRIMARY_PAIRS] == [
        "protgpt3-112m-dpo", "protgpt3-1.3b-dpo", "protgpt3-10b-dpo"]
    assert ota.WITHIN_ARM_PAIRS[0].tuned == "zymctrl"
    # The two identity endpoints are different units and are never one family.
    assert ota.IDENTITY_MEAN_FAMILY != ota.IDENTITY_FRACTION_FAMILY
    assert ota.FAMILY_SIZES[ota.IDENTITY_MEAN_FAMILY] == len(ota.PRIMARY_PAIRS)


def test_a_realised_family_that_is_not_the_declared_one_refuses():
    short = [{"key": f"k{index}", "base": {}, "tuned": {}} for index in range(8)]
    with pytest.raises(ValueError, match="declares 12 columns and realises 8"):
        ota.contrast_bands(short, family=ota.PRIMARY_FAMILY)
    repeated = [{"key": "k", "base": {}, "tuned": {}} for _ in range(12)]
    with pytest.raises(ValueError, match="repeats a column key"):
        ota.contrast_bands(repeated, family=ota.PRIMARY_FAMILY)


def test_a_pool_below_the_bootstrap_floor_is_unresolved_not_an_estimate():
    rng = np.random.default_rng(0)
    columns = []
    for index in range(ota.PRIMARY_FAMILY_SIZE):
        size = ota.MIN_UNITS_PER_POOL - 1 if index == 0 else 40
        columns.append({"key": f"k{index}",
                        "base": {1: rng.normal(60, 5, size).tolist(),
                                 2: rng.normal(60, 5, 40).tolist()},
                        "tuned": {1: rng.normal(62, 5, 40).tolist(),
                                  2: rng.normal(62, 5, 40).tolist()}})
    record = ota.contrast_bands(columns, family=ota.PRIMARY_FAMILY)
    assert record["declared_family_size"] == 12 and record["resolved_family_size"] == 11
    thin = record["columns"]["k0"]
    assert thin["status"] == ota.UNRESOLVED and thin["point"] is None and thin["interval"] is None
    assert str(ota.MIN_UNITS_PER_POOL) in thin["reason"]
    good = record["columns"]["k1"]
    assert good["interval"] == [good["point"] - record["critical_value"],
                                good["point"] + record["critical_value"]]
    assert record["standardized"]["critical_value"] is not None


def test_the_margin_and_the_decision_rule_are_the_declared_ones():
    assert ota.margin_for(50.0)["margin"] == pytest.approx(5.0)
    assert ota.margin_for(68.0)["margin"] == pytest.approx(ota.MARGIN_FLOOR)
    # The margin is one-sided: a base arm already above the criterion has no
    # headroom left and takes the floor, rather than the laxest margin of all.
    above = ota.margin_for(90.0)
    assert above["margin"] == pytest.approx(ota.MARGIN_FLOOR)
    assert above["headroom_to_criterion"] == pytest.approx(0.0)
    assert ota.margin_for(70.0)["margin"] == pytest.approx(ota.MARGIN_FLOOR)
    assert ota.margin_finding([3.0, 6.0], 2.0)["finding"] == ota.SUPPORTED
    assert ota.margin_finding([-1.0, 1.5], 2.0)["finding"] == ota.BOUNDED_NEGATIVE
    assert ota.margin_finding([-5.0, 9.0], 2.0)["finding"] == ota.UNRESOLVED
    # A resolved positive smaller than the margin is unresolved, never a negative.
    assert ota.margin_finding([0.5, 4.0], 2.0)["finding"] == ota.UNRESOLVED


# ---------------------------------------------------------------- end to end

def test_primary_contrast_on_real_measurements(analysis):
    band = analysis["primary"]["bands"]
    assert band["declared_family_size"] == 12
    assert band["draws"] == bd.DRAWS and band["seed"] == bd.SEED
    assert band["confidence"] == bd.CONFIDENCE
    assert band["family"] == ota.PRIMARY_FAMILY
    near_duplicate = [key for key, entry in band["columns"].items()
                      if entry["status"] == ota.UNRESOLVED]
    assert sorted(near_duplicate) == sorted(f"{pair.tuned}|{ota.STRATA[3]}"
                                            for pair in ota.PRIMARY_PAIRS)
    assert band["resolved_family_size"] == 9
    for key, entry in band["columns"].items():
        if entry["status"] != "resolved":
            continue
        assert entry["interval"] == [entry["point"] - band["critical_value"],
                                     entry["point"] + band["critical_value"]]
        assert entry["point"] == pytest.approx(entry["tuned_mean"] - entry["base_mean"])
        assert 0.0 <= entry["base_mean"] <= 100.0
        arm, stratum = key.split("|")
        cell = analysis["primary"]["per_pair"][arm]["per_stratum"][stratum]
        assert cell["margin"]["margin"] == pytest.approx(
            max(ota.MARGIN_FLOOR, 0.25 * max(0.0, 70.0 - entry["base_mean"])))
        assert cell["finding"] in ota.FINDING_STATES
        assert len(cell["streams"]) == aot.builder.STREAMS
    for pair in ota.PRIMARY_PAIRS:
        decision = analysis["primary"]["per_pair"][pair.tuned]["pair_decision"]
        assert decision["design_relevant_bias"]["finding"] in ota.FINDING_STATES
        assert decision["narrowing_confined_to_record_strata"]["finding"] in ota.FINDING_STATES
        thin = analysis["primary"]["per_pair"][pair.tuned]["per_stratum"][ota.STRATA[3]]
        assert thin["finding"] == ota.UNRESOLVED and thin["margin"] is None


def test_the_denominator_keeps_empty_and_censored_attempts(analysis):
    for key, cell in analysis["cells"].items():
        factor = aot.builder.CELLS_PER_STREAM[cell["role"]]
        assert cell["attempts"] == ATTEMPTS * factor
        assert cell["empty_or_noncanonical"] == EMPTIES * factor
        assert cell["budget_censored"] > 0, key
        assert cell["non_empty"] == cell["attempts"] - cell["empty_or_noncanonical"]
        assert cell["folded"] == cell["non_empty"] and cell["searched"] == cell["non_empty"]
    fraction = analysis["identity_shift"]["fraction_below_30"]
    pair = ota.PRIMARY_PAIRS[0]
    entry = fraction["columns"][pair.tuned]
    assert entry["status"] == "resolved"
    base_cell = analysis["cells"][f"{pair.base}|s1"]
    # The rate's denominator is every retained attempt, so an empty continuation
    # lowers the fraction below 30% rather than leaving the pool: the realised
    # rate is 30/80 and not the 30/74 a non-empty denominator would give.
    assert entry["units"]["base"]["1"] == base_cell["attempts"]
    assert entry["base_mean"] == pytest.approx(PER_CELL[ota.STRATA[0]] / ATTEMPTS)
    assert entry["base_mean"] != pytest.approx(
        PER_CELL[ota.STRATA[0]] / (ATTEMPTS - EMPTIES))


def test_the_explicit_statistics_control_reproduces_a_stated_share(analysis):
    control = analysis["explicit_statistics_control"]
    assert sorted(control["rules"]) == ["composition_order0", "low_complexity_residues",
                                        "markov_order2"]
    assert control["top_fraction"] == 0.25
    resolved_cells = 0
    for name, block in control["rules"].items():
        band = block["bands"]
        # One declared family per rule, never merged with another rule's or with
        # the primary family, and each reads its own seed block.
        assert band["family"] == ota.CONTROL_FAMILIES[name]
        assert band["declared_family_size"] == ota.PRIMARY_FAMILY_SIZE == 12
        assert band["draws"] == bd.DRAWS and band["seed"] == bd.SEED
        assert band["common_draw_discard_fraction"] <= ota.MAX_INVALID_DRAW_FRACTION
        for stream, selection in block["selection"][ota.PRIMARY_PAIRS[0].tuned].items():
            assert selection["n_retained"] == int(selection["n_scored"] * 0.25)
            assert selection["n_retained"] == selection["tuned_folded_pool"], (name, stream)
            assert selection["n_unscorable"] == 0
        for key, cell in band["columns"].items():
            arm, stratum = key.split("|")
            if cell["status"] != "resolved":
                assert cell["reason"] and cell["interval"] is None
                continue
            resolved_cells += 1
            realised = cell["realised"]
            gain = realised["tuned_realised"] - realised["base_realised"]
            reproduced = realised["filtered_realised"] - realised["base_realised"]
            assert gain > 0
            assert cell["gains"]["tuned_minus_base_gain"] == pytest.approx(gain)
            assert cell["gains"]["filter_minus_base_gain"] == pytest.approx(reproduced)
            assert cell["point"] == pytest.approx(reproduced / gain)
            assert cell["interval"] == [cell["point"] - band["critical_value"],
                                        cell["point"] + band["critical_value"]]
            assert cell["invalid_draw_fraction"] <= ota.MAX_INVALID_DRAW_FRACTION
            assert cell["standardized_sensitivity"]["standard_error"] > 0
            assert block["identity_axis"][arm][stratum]["filtered_mean_identity"] is not None
        # The near-duplicate stratum is below the control's own floor on a
        # single-cell tuned stream, so it carries counts and no estimate.
        for pair in ota.PRIMARY_PAIRS:
            thin = band["columns"][f"{pair.tuned}|{ota.STRATA[3]}"]
            assert thin["status"] == ota.UNRESOLVED and thin["interval"] is None
    assert resolved_cells >= 3 * len(ota.PRIMARY_PAIRS)


def test_the_nested_resample_shares_one_draw_between_the_two_means():
    rng = np.random.default_rng(3)
    members = [{"key": f"m{index:03d}", "plddt": float(60 + 20 * rng.random()),
                "stratum": ota.STRATA[index % 2], "score": float(rng.random())}
               for index in range(120)]
    kwargs = dict(retain="highest", family=ota.CONTROL_FAMILIES["markov_order2"],
                  column=0, stream_index=0, draws=500)
    first = ota.nested_base_draws(members, **kwargs)
    second = ota.nested_base_draws(members, **kwargs)
    for stratum in ota.STRATA[:2]:
        assert np.array_equal(first[stratum]["base"], second[stratum]["base"])
        assert np.array_equal(first[stratum]["filtered"], second[stratum]["filtered"])
        assert first[stratum]["valid"].all()
        # The filter keeps the top quarter by score and the scores here are
        # independent of the values, so the two means differ yet move together,
        # which is only possible if they are read off the same draw.
        assert not np.array_equal(first[stratum]["base"], first[stratum]["filtered"])
        assert np.corrcoef(first[stratum]["base"], first[stratum]["filtered"])[0, 1] > 0.0
    # A stratum with no member is invalid on every draw rather than zero-valued.
    assert not first[ota.STRATA[2]]["valid"].any()
    assert np.isnan(first[ota.STRATA[2]]["base"]).all()


def test_a_share_of_a_gain_that_is_not_resolved_carries_no_band():
    """A denominator that can reach zero gives a width, not an uncertainty."""

    rng = np.random.default_rng(5)
    columns = []
    for index in range(ota.PRIMARY_FAMILY_SIZE):
        members = [{"key": f"m{position:03d}", "plddt": float(60 + 5 * rng.standard_normal()),
                    "stratum": ota.STRATA[0], "score": float(rng.random())}
                   for position in range(80)]
        draws = ota.nested_base_draws(members, retain="highest",
                                      family=ota.CONTROL_FAMILIES["composition_order0"],
                                      column=index, stream_index=0, draws=400)
        # Every column but the first has a tuned pool that overlaps its base
        # pool, so some resample shows no gain for it to be a share of.
        offset = 20.0 if index == 0 else 0.0
        mean = float(np.mean([row["plddt"] for row in members]))
        columns.append({
            "key": f"k{index}",
            # The primary endpoint resolves a gain in every column here, so each
            # share is a real quantity and only the ratio's own band is at issue.
            "primary": {"status": "resolved", "zero_verdict": "positive",
                        "point": 1.0, "interval": [0.5, 1.5]},
            "base": {1: draws[ota.STRATA[0]]["base"]},
            "filtered": {1: draws[ota.STRATA[0]]["filtered"]},
            "valid": {1: draws[ota.STRATA[0]]["valid"]},
            "tuned": {1: [row["plddt"] + offset for row in members]},
            "base_realised": mean, "filtered_realised": mean + 1.0,
            "tuned_realised": mean + offset + 0.05,
        })
    record = ota.reproduced_fraction_bands(
        columns, family=ota.CONTROL_FAMILIES["composition_order0"], draws=400)
    assert record["declared_family_size"] == 12
    assert record["columns"]["k0"]["status"] == "resolved"
    for index in range(1, ota.PRIMARY_FAMILY_SIZE):
        entry = record["columns"][f"k{index}"]
        assert entry["status"] == ota.UNRESOLVED and entry["interval"] is None
        assert "gain" in entry["reason"]
    assert record["resolved_family_size"] == 1


def test_the_undeclarable_endpoints_say_so_rather_than_guessing(analysis):
    assert analysis["prediction_side_effect"]["status"] == "not_measured"
    assert "scoring" in analysis["prediction_side_effect"]["reason"]
    assert analysis["recognition"]["status"] == "not_measured"
    assert analysis["recognition"]["complete_domain_coverage"] == 0.8
    selector = analysis["selector_competence"]["per_arm"]
    for arm, entry in selector.items():
        assert entry["oracle_gain"] > 0.0, arm
        assert entry["captured"]["status"] == "not_measured"
    assert analysis["within_arm"]["status"] == "not_measured"
    assert analysis["within_arm"]["structural_contrast"]["finding"] == ota.NON_IDENTIFIABLE
    shift = analysis["identity_shift"]["shift_share"][ota.PRIMARY_PAIRS[0].tuned]
    assert shift["pooled_gain"] == pytest.approx(shift["within_stratum_term"]
                                                 + shift["composition_term"])
    assert Path(analysis["campaign_root"]).name == "campaign"


# ------------------------------------------------------------------- refusals

def test_a_missing_base_pool_refuses(triples, tmp_path):
    with pytest.MonkeyPatch.context() as monkeypatch:
        keep = {arm for pair in ota.PRIMARY_PAIRS for arm in pair.arms} - {"protgpt3-10b"}
        campaign, structure, novelty = build_campaign(tmp_path, triples, monkeypatch, arms=keep)
        with pytest.raises(SystemExit, match="matched base pool protgpt3-10b is missing"):
            invoke(tmp_path, campaign, structure, novelty)


def test_an_absent_plddt_column_refuses(triples, tmp_path):
    with pytest.MonkeyPatch.context() as monkeypatch:
        one = {arm for arm in ota.PRIMARY_PAIRS[0].arms}
        campaign, structure, novelty = build_campaign(tmp_path, triples, monkeypatch, arms=one,
                                                      drop_plddt=True)
        with pytest.raises(SystemExit, match="the pLDDT column is absent"):
            invoke(tmp_path, campaign, structure, novelty)


def test_the_within_arm_pair_is_refused_in_the_free_length_root(triples, tmp_path):
    with pytest.MonkeyPatch.context() as monkeypatch:
        campaign, structure, novelty = build_campaign(
            tmp_path, triples, monkeypatch, arms={arm for arm in ota.PRIMARY_PAIRS[0].arms})
        stolen = campaign / f"{aot.builder.KEY}_zymctrl_s1_r0"
        stolen.mkdir()
        (stolen / ug.LEDGER_NAME).write_text("", encoding="utf-8")
        (stolen / ug.EXPECT_GENERATE).write_text("{}", encoding="utf-8")
        with pytest.raises(SystemExit, match="within-arm only"):
            invoke(tmp_path, campaign, structure, novelty)


def test_an_interface_cell_is_refused_as_a_measurement(triples, tmp_path):
    with pytest.MonkeyPatch.context() as monkeypatch:
        campaign, structure, novelty = build_campaign(
            tmp_path, triples, monkeypatch, arms={arm for arm in ota.PRIMARY_PAIRS[0].arms})
        cell = next(campaign.iterdir())
        summary = json.loads((cell / ug.EXPECT_GENERATE).read_text(encoding="utf-8"))
        summary["is_scientific_measurement"] = False
        (cell / ug.EXPECT_GENERATE).write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(SystemExit, match="is_scientific_measurement"):
            invoke(tmp_path, campaign, structure, novelty)


# -------------------------------------------------------- the within-arm lane

def test_within_arm_reading_on_its_own_frozen_instrument():
    """The amendment's reading, against the search of this pair's own cells."""

    record = aot.within_arm_reading(WITHIN_ARM_NOVELTY, CONDITIONED)
    assert record["status"] == "measured"
    # No matched structural contrast for this pair, at this lane, ever.
    assert record["structural_contrast"]["finding"] == ota.NON_IDENTIFIABLE
    assert "confounded" in record["structural_contrast"]["reason"]
    assert "two hundred" in record["population_note"]

    zymctrl = record["per_arm"]["zymctrl"]["conditions"]
    assert sorted(zymctrl) == ["mismatched", "requested"]
    for condition, entry in zymctrl.items():
        assert entry["status"] == "measured", condition
        assert entry["search_coverage"] == 1.0
        assert entry["searched"] == entry["distinct_sequences"]
        assert entry["generated_samples"] == 3200
        assert sum(entry["stratum_shares"].values()) == pytest.approx(1.0)
        assert entry["fraction_below_30"] == pytest.approx(
            entry["stratum_shares"][ota.STRATA[0]])
        assert 0.0 <= entry["mean_identity"] <= 100.0
    floor = record["per_arm"]["protgpt2"]["conditions"]["unconditioned_floor"]
    assert floor["status"] == "measured" and floor["distinct_sequences"] == 200
    assert floor["search_coverage"] == 1.0


def test_within_arm_reading_refuses_to_average_a_partial_search():
    """A distribution read off part of a pool is biased by whatever chose that part."""

    record = aot.within_arm_reading(OTHER_CAMPAIGN_NOVELTY, CONDITIONED)
    assert record["status"] == "measured"
    for condition, entry in record["per_arm"]["zymctrl"]["conditions"].items():
        assert entry["status"] == ota.UNRESOLVED, condition
        assert 0.0 < entry["search_coverage"] < 0.01
        assert "descriptive counts rather than an estimate" in entry["reason"]
    floor = record["per_arm"]["protgpt2"]["conditions"]["unconditioned_floor"]
    assert floor["searched"] == 0 and floor["status"] == ota.UNRESOLVED


def test_the_control_arithmetic_and_its_undefined_case():
    positive = ota.reproduced_fraction(60.0, 68.0, 66.0)
    assert positive["tuned_minus_base_gain"] == pytest.approx(8.0)
    assert positive["filter_minus_base_gain"] == pytest.approx(6.0)
    assert positive["reproduced_fraction"] == pytest.approx(0.75)
    at_one = ota.reproduced_fraction(60.0, 68.0, 68.0)
    assert at_one["reproduced_fraction"] == pytest.approx(1.0)
    flat = ota.reproduced_fraction(60.0, 60.0, 64.0)
    assert flat["reproduced_fraction"] is None and "not positive" in flat["reason"]
    split = ota.shift_share(
        {ota.STRATA[0]: {"share": 0.5, "mean": 40.0},
         ota.STRATA[2]: {"share": 0.5, "mean": 90.0}},
        {ota.STRATA[0]: {"share": 0.25, "mean": 40.0},
         ota.STRATA[2]: {"share": 0.75, "mean": 90.0}})
    assert split["within_stratum_term"] == pytest.approx(0.0)
    assert split["composition_term"] == pytest.approx(12.5)
    assert split["pooled_gain"] == pytest.approx(12.5)


def test_the_band_is_reproducible_from_the_declared_seed():
    rng = np.random.default_rng(7)
    columns = [{"key": f"k{index}",
                "base": {1: rng.normal(60, 6, 50).tolist(), 2: rng.normal(60, 6, 50).tolist()},
                "tuned": {1: rng.normal(63, 6, 50).tolist(), 2: rng.normal(63, 6, 50).tolist()}}
               for index in range(ota.PRIMARY_FAMILY_SIZE)]
    first = ota.contrast_bands(columns, family=ota.PRIMARY_FAMILY)
    second = ota.contrast_bands(columns, family=ota.PRIMARY_FAMILY)
    assert first["critical_value"] == second["critical_value"]
    assert [first["columns"][key]["interval"] for key in sorted(first["columns"])] == \
           [second["columns"][key]["interval"] for key in sorted(second["columns"])]
    # A different declared family reads its own seed block, so the two cannot
    # silently share draws.
    narrow = ota.contrast_bands(columns[:ota.IDENTITY_FAMILY_SIZE],
                                family=ota.IDENTITY_MEAN_FAMILY)
    assert narrow["critical_value"] != first["critical_value"]
    assert narrow["seed"] == bd.SEED and narrow["draws"] == bd.DRAWS


def test_a_cohort_keyed_by_the_sequence_digest_joins(triples, tmp_path):
    """The campaign's own fold cohort keys rows by the digest and has no digest column."""

    with pytest.MonkeyPatch.context() as monkeypatch:
        one = set(ota.PRIMARY_PAIRS[0].arms)
        campaign, structure, novelty = build_campaign(tmp_path, triples, monkeypatch, arms=one,
                                                      cohort_shape="digest_id")
        record = invoke(tmp_path, campaign, structure, novelty, "--dry-run")
    assert record["status"] == "dry_run"
    assert len(record["cells"]) == 2 * aot.builder.STREAMS
    for cell in record["cells"].values():
        assert cell["searched"] == cell["non_empty"] == cell["folded"]
        assert cell["identity_search_coverage"] == 1.0


def test_a_recorded_digest_that_disagrees_with_the_bytes_refuses(triples, tmp_path):
    with pytest.MonkeyPatch.context() as monkeypatch:
        one = set(ota.PRIMARY_PAIRS[0].arms)
        campaign, structure, novelty = build_campaign(tmp_path, triples, monkeypatch, arms=one,
                                                      corrupt_digest=True)
        with pytest.raises(SystemExit, match="the join key and the sequence disagree"):
            invoke(tmp_path, campaign, structure, novelty, "--dry-run")


def test_the_second_narrowing_route_is_quantified():
    """Composition term above half the pooled gain, with a resolved move toward the record."""

    toward = {ota.IDENTITY_MEAN_FAMILY: {"status": "resolved", "point": 4.0, "interval": [1.0, 7.0]},
              ota.IDENTITY_FRACTION_FAMILY: {"status": "resolved", "point": 0.0,
                                             "interval": [-0.02, 0.02]}}
    away = {ota.IDENTITY_MEAN_FAMILY: {"status": "resolved", "point": -4.0,
                                       "interval": [-7.0, -1.0]},
            ota.IDENTITY_FRACTION_FAMILY: {"status": "resolved", "point": 0.05,
                                           "interval": [0.01, 0.09]}}
    wide = {name: {"status": ota.UNRESOLVED, "reason": "spans zero", "interval": None}
            for name in (ota.IDENTITY_MEAN_FAMILY, ota.IDENTITY_FRACTION_FAMILY)}
    accounts = {"pooled_gain": 4.0, "composition_term": 3.0, "within_stratum_term": 1.0}
    small = {"pooled_gain": 4.0, "composition_term": 1.0, "within_stratum_term": 3.0}
    flat = {"pooled_gain": -1.0, "composition_term": 2.0, "within_stratum_term": -3.0}

    supported = ota.narrowing_by_composition(accounts, toward)
    assert supported["finding"] == ota.SUPPORTED
    assert supported["share_of_pooled_gain"] == pytest.approx(0.75)
    # A large composition term on a shift no band resolves is not the finding.
    assert ota.narrowing_by_composition(accounts, wide)["finding"] == ota.UNRESOLVED
    # Nor is a resolved move that cannot account for the gain.
    assert ota.narrowing_by_composition(small, toward)["finding"] == ota.UNRESOLVED
    # A move away from the record is never narrowing, however large the term.
    assert ota.narrowing_by_composition(accounts, away)["finding"] == ota.UNRESOLVED
    # With no gain there is nothing for a shift to account for: unresolved, not negative.
    no_gain = ota.narrowing_by_composition(flat, toward)
    assert no_gain["finding"] == ota.UNRESOLVED and "not positive" in no_gain["reason"]
    # The fraction below 30% falling is the other direction of the same move.
    only_fraction = ota.narrowing_by_composition(
        accounts, {ota.IDENTITY_MEAN_FAMILY: wide[ota.IDENTITY_MEAN_FAMILY],
                   ota.IDENTITY_FRACTION_FAMILY: {"status": "resolved", "point": -0.05,
                                                  "interval": [-0.09, -0.01]}})
    assert only_fraction["finding"] == ota.SUPPORTED


def test_the_pair_decision_reports_both_narrowing_routes(analysis):
    for pair in ota.PRIMARY_PAIRS:
        decision = analysis["primary"]["per_pair"][pair.tuned]["pair_decision"]
        assert set(decision) == {"design_relevant_bias", "narrowing_confined_to_record_strata",
                                 "narrowing_by_composition_shift", "narrowing"}
        for key, entry in decision.items():
            assert entry["finding"] in ota.FINDING_STATES, (pair.tuned, key)
        composition = decision["narrowing_by_composition_shift"]
        assert composition["rule"] == ota.NARROWING_COMPOSITION_RULE
        assert set(composition["identity_moves"]) == {ota.IDENTITY_MEAN_FAMILY,
                                                      ota.IDENTITY_FRACTION_FAMILY}
    # The secondary pair carries no identity-shift family of its own, so the
    # second narrowing route is unreadable for it and says so.
    secondary = ota.SECONDARY_PAIRS[0].tuned
    route = analysis["secondary_pair"]["per_pair"][secondary]["pair_decision"]
    assert route["narrowing_by_composition_shift"]["finding"] == ota.UNRESOLVED
    assert "no declared identity-shift family" in \
        route["narrowing_by_composition_shift"]["reason"]


def test_the_banded_share_says_whether_statistics_reproduce_the_whole_gain(analysis):
    """The reading a point estimate cannot give: most of the gain against all of it."""

    assert ota.reproduction_finding([1.1, 1.4])["finding"] == ota.SUPPORTED
    assert ota.reproduction_finding([0.1, 0.9])["finding"] == ota.BOUNDED_NEGATIVE
    assert ota.reproduction_finding([0.4, 1.6])["finding"] == ota.UNRESOLVED
    seen = set()
    for block in analysis["explicit_statistics_control"]["rules"].values():
        band = block["bands"]
        assert band["full_reproduction"] == 1.0
        assert sum(band["reproduction_counts"].values()) == band["resolved_family_size"]
        for cell in band["columns"].values():
            if cell["status"] != "resolved":
                continue
            entry = cell["statistics_reproduce_the_gain"]
            assert entry["finding"] in ota.FINDING_STATES
            low, high = cell["interval"]
            if entry["finding"] == ota.BOUNDED_NEGATIVE:
                assert high < 1.0
            elif entry["finding"] == ota.SUPPORTED:
                assert low > 1.0
            else:
                assert low <= 1.0 <= high
            seen.add(entry["finding"])
    # On the fixture the three rules reproduce none of the constructed gain, and
    # the band is what shows that rather than the point estimate alone.
    assert ota.BOUNDED_NEGATIVE in seen


def test_the_secondary_pair_is_read_beside_the_primary_family_not_inside_it(analysis):
    primary, secondary = analysis["primary"]["bands"], analysis["secondary_pair"]["bands"]
    assert primary["family"] == ota.PRIMARY_FAMILY
    assert secondary["family"] == ota.SECONDARY_FAMILY
    assert secondary["declared_family_size"] == ota.SECONDARY_FAMILY_SIZE == 4
    assert primary["declared_family_size"] == 12
    tuned = ota.SECONDARY_PAIRS[0].tuned
    assert not any(key.startswith(tuned) for key in primary["columns"])
    assert all(key.startswith(tuned) for key in secondary["columns"])
    # Its own seed block, so the two families never read the same draws.
    assert ota.FAMILY_SEED_OFFSETS[ota.SECONDARY_FAMILY] != \
        ota.FAMILY_SEED_OFFSETS[ota.PRIMARY_FAMILY]
    assert secondary["critical_value"] != primary["critical_value"]
    # The explicit-statistics control is declared over the matched pairs only.
    for block in analysis["explicit_statistics_control"]["rules"].values():
        assert not any(key.startswith(tuned) for key in block["bands"]["columns"])
    assert tuned in analysis["selector_competence"]["per_arm"]


def test_a_control_with_no_gain_to_recover_is_moot_not_unresolved():
    """The two readings a single `unresolved` would conflate."""

    absent = ota.primary_gain_disposition(
        {"status": "resolved", "zero_verdict": "unresolved", "point": 0.2,
         "interval": [-2.0, 2.4]})
    assert absent["gain"] == ota.GAIN_ABSENT and "no in-stratum gain" in absent["reason"]
    identified = ota.primary_gain_disposition(
        {"status": "resolved", "zero_verdict": "positive", "point": 5.0, "interval": [3.0, 7.0]})
    assert identified["gain"] == ota.GAIN_IDENTIFIED and identified["reason"] is None
    unmeasured = ota.primary_gain_disposition(
        {"status": ota.UNRESOLVED, "reason": "below the floor", "interval": None})
    assert unmeasured["gain"] == ota.GAIN_UNMEASURED and "not measured" in unmeasured["reason"]
    assert ota.primary_gain_disposition(None)["gain"] == ota.GAIN_UNMEASURED

    # A whole family with no gain anywhere is moot, and the record says so
    # rather than emitting the same word it uses for an imprecise share.
    rng = np.random.default_rng(11)
    columns = []
    for index in range(ota.PRIMARY_FAMILY_SIZE):
        members = [{"key": f"m{position:03d}", "plddt": float(60 + 5 * rng.standard_normal()),
                    "stratum": ota.STRATA[0], "score": float(rng.random())}
                   for position in range(40)]
        draws = ota.nested_base_draws(members, retain="highest",
                                      family=ota.CONTROL_FAMILIES["markov_order2"],
                                      column=index, stream_index=0, draws=200)
        mean = float(np.mean([row["plddt"] for row in members]))
        columns.append({
            "key": f"k{index}",
            "primary": {"status": "resolved", "zero_verdict": "unresolved", "point": 0.3,
                        "interval": [-1.9, 2.5]},
            "base": {1: draws[ota.STRATA[0]]["base"]},
            "filtered": {1: draws[ota.STRATA[0]]["filtered"]},
            "valid": {1: draws[ota.STRATA[0]]["valid"]},
            "tuned": {1: [row["plddt"] + 9.0 for row in members]},
            "base_realised": mean, "filtered_realised": mean + 1.0,
            "tuned_realised": mean + 9.0,
        })
    record = ota.reproduced_fraction_bands(
        columns, family=ota.CONTROL_FAMILIES["markov_order2"], draws=200)
    assert record["control_status"] == "moot"
    assert record["primary_gain_counts"][ota.GAIN_ABSENT] == ota.PRIMARY_FAMILY_SIZE
    assert record["resolved_family_size"] == 0 and record["critical_value"] is None
    assert "rests on the primary endpoint" in record["moot_note"]
    for entry in record["columns"].values():
        # Non-identifiable, not unresolved: the share is undefined, and a
        # realised gain of nine points in these pools does not rescue it.
        assert entry["status"] == ota.NON_IDENTIFIABLE
        assert entry["point"] is None and entry["interval"] is None


def test_the_control_names_the_primary_endpoint_for_every_cell(analysis):
    for block in analysis["explicit_statistics_control"]["rules"].values():
        band = block["bands"]
        assert band["control_status"] == "readable"
        assert sum(band["primary_gain_counts"].values()) == ota.PRIMARY_FAMILY_SIZE
        assert band["primary_gain_counts"][ota.GAIN_ABSENT] == 0
        for key, cell in band["columns"].items():
            assert cell["primary_gain"]["gain"] in ota.GAIN_STATES
            if key.endswith(ota.STRATA[3]):
                # The primary endpoint does not resolve the near-duplicate
                # stratum here, so the share is unmeasured rather than absent.
                assert cell["primary_gain"]["gain"] == ota.GAIN_UNMEASURED
                assert cell["status"] == ota.UNRESOLVED
            else:
                assert cell["primary_gain"]["gain"] == ota.GAIN_IDENTIFIED
