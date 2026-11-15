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

    if not any(path.exists() for path in (EVIDENCE, KMERS)):
        pytest.skip("the measured generation evidence and UniRef50 k-mer bundle are not distributed")
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


def test_an_interval_beyond_the_negative_margin_is_not_described_as_spanning_it():
    """The unresolved state is one word for two situations; the reason is not.

    The campaign's primary cell is the below-30%-identity 10B contrast, whose
    interval lies wholly past the negative margin. Reporting that as an interval
    that "spans" its margin describes the opposite of the measurement, in a
    column that feeds the write-up, so the two situations carry different
    sentences while keeping the one finding state the vocabulary has for them.
    """

    beyond = ota.margin_finding([-12.193416721734723, -4.31528250734277], 2.8896638248068243)
    straddling = ota.margin_finding([-5.0, 9.0], 2.0)
    assert beyond["finding"] == straddling["finding"] == ota.UNRESOLVED
    assert "spans" not in beyond["reason"]
    assert "at or below the -2.890-point margin" in beyond["reason"]
    assert "excluded rather than unmeasured" in beyond["reason"]
    assert "spans the 2.000-point margin" in straddling["reason"]
    # The boundary case belongs with the interval that does not reach past it.
    assert "spans" not in ota.margin_finding([-9.0, -2.0], 2.0)["reason"]
    assert "spans" in ota.margin_finding([-9.0, -1.0], 2.0)["reason"]


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


def test_an_attempt_is_keyed_by_its_digest_rank_and_not_its_ledger_position(built):
    """The key breaks a filter rule's ties, so it may not carry generation order."""

    table = aot.assemble(aot.read_campaign(built["campaign"]),
                         aot.read_structure([built["structure"]]),
                         aot.read_novelty(aot.novelty_sources([built["novelty"]])))
    for (arm, stream), pool in table["pools"].items():
        digests = [row["digest"] for row in pool["rows"]]
        assert digests == sorted(digests), (arm, stream)
        assert [row["key"] for row in pool["rows"]] == [
            f"{arm}|s{stream}|{index}" for index in range(len(digests))], (arm, stream)


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
            assert 0 < selection["n_retained_at_cut"] <= min(selection["n_retained"],
                                                             selection["n_tied_at_cut"])
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


def test_a_readout_sees_each_chunk_of_draws_and_cannot_change_them():
    rng = np.random.default_rng(3)
    members = [{"key": f"m{index:03d}", "plddt": float(60 + 20 * rng.random()),
                "stratum": ota.STRATA[index % 2], "score": float(rng.random())}
               for index in range(120)]
    kwargs = dict(retain="highest", family=ota.CONTROL_FAMILIES["markov_order2"],
                  column=0, stream_index=0, draws=500)
    seen = []
    watched = ota.nested_base_draws(
        members, readout=lambda start, stop, blocks, top: seen.append(
            (start, stop, set(blocks), top.shape)), **kwargs)
    plain = ota.nested_base_draws(members, **kwargs)
    for stratum in ota.STRATA:
        for field in ("base", "filtered", "valid"):
            assert np.array_equal(plain[stratum][field], watched[stratum][field], equal_nan=True)
    assert [(start, stop) for start, stop, _, _ in seen] == [
        (start, start + bd.DRAW_CHUNK) for start in range(0, 500, bd.DRAW_CHUNK)]
    assert all(strata == set(ota.STRATA[:2]) and shape == (stop - start, 30)
               for start, stop, strata, shape in seen)


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

    if not any(path.exists() for path in (WITHIN_ARM_NOVELTY, CONDITIONED)):
        pytest.skip("the within-arm novelty and conditioned-generation bundle is not distributed")
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

    if not any(path.exists() for path in (OTHER_CAMPAIGN_NOVELTY, CONDITIONED)):
        pytest.skip("the partial-coverage novelty and conditioned-generation bundle is not distributed")
    record = aot.within_arm_reading(OTHER_CAMPAIGN_NOVELTY, CONDITIONED)
    assert record["status"] == "measured"
    for condition, entry in record["per_arm"]["zymctrl"]["conditions"].items():
        assert entry["status"] == ota.UNRESOLVED, condition
        assert 0.0 < entry["search_coverage"] < 0.01
        assert "descriptive counts rather than an estimate" in entry["reason"]
    floor = record["per_arm"]["protgpt2"]["conditions"]["unconditioned_floor"]
    assert floor["searched"] == 0 and floor["status"] == ota.UNRESOLVED


def test_the_control_arithmetic_and_its_undefined_case():
    assert ota.reproduction_finding([1.1, 1.4])["finding"] == ota.SUPPORTED
    assert ota.reproduction_finding([0.1, 0.9])["finding"] == ota.BOUNDED_NEGATIVE
    assert ota.reproduction_finding([0.4, 1.6])["finding"] == ota.UNRESOLVED
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


# --------------------------------------------- the base-likelihood rule (2026-11-14)

def _likelihood_fixture(tmp_path, sequences, mean_nll):
    """A score file in the scorer's own row shape, plus the table read from it."""

    path = tmp_path / "own.jsonl"
    path.write_text("".join(json.dumps({
        "sequence_sha256": sequence_digest(sequence), "n_tokens": len(sequence),
        "sum_nll": mean_nll[index] * len(sequence), "mean_nll": mean_nll[index]}) + "\n"
        for index, sequence in enumerate(sequences)), encoding="utf-8")
    return path


def test_the_likelihood_rule_ranks_by_mean_loglikelihood_per_token(tmp_path):
    sequences = ["ACDEFGHIKL", "ACDEFGHIKLMNPQRSTVWY" * 2, "MKV"]
    # Mean NLL falls as the sequence gets more probable; the longest has the
    # largest *summed* NLL and the lowest *mean*.
    table = ota.read_own_likelihood(_likelihood_fixture(tmp_path, sequences, [2.0, 1.0, 3.0]))
    rules = {rule.name: rule for rule in ota.filter_rules(_counts(), table)}
    declared, summed = rules[ota.LIKELIHOOD_RULE_NAME], rules[ota.LIKELIHOOD_SENSITIVITY_NAME]
    assert declared.retain == summed.retain == "highest"
    assert [declared.statistic(s) for s in sequences] == pytest.approx([-2.0, -1.0, -3.0])
    assert [summed.statistic(s) for s in sequences] == pytest.approx([-20.0, -40.0, -9.0])
    scored = [{"key": s, "sequence": s, "score": declared.statistic(s)} for s in sequences]
    assert ota.rank_and_retain(scored, declared, fraction=0.34)["retained_keys"] == {sequences[1]}
    scored = [{"key": s, "sequence": s, "score": summed.statistic(s)} for s in sequences]
    assert ota.rank_and_retain(scored, summed, fraction=0.34)["retained_keys"] == {"MKV"}


def _counts():
    """Small synthetic counts for rule declarations and rankings, not corpus measurements."""
    from src.capability.generation import generative_control as gc
    return {k: np.ones(len(gc.AA20) ** k, dtype=np.int64) for k in aot.KMER_ORDERS}


def test_the_three_declared_rules_are_unchanged_without_a_likelihood_table():
    assert tuple(rule.name for rule in ota.filter_rules(_counts())) == ota.FILTER_RULE_NAMES


def test_a_sequence_without_a_score_refuses_rather_than_dropping(tmp_path):
    table = ota.read_own_likelihood(_likelihood_fixture(tmp_path, ["ACDEFGHIKL"], [2.0]))
    rule = ota.filter_rules(_counts(), table)[3]
    with pytest.raises(ValueError, match="partial pool"):
        ota.score_pool([{"key": "a", "sequence": "ACDEFGHIKL"},
                        {"key": "b", "sequence": "WWWWWWWWWW"}], rule)


def test_a_score_over_the_wrong_token_count_refuses():
    digest = sequence_digest("ACDEFGHIKL")
    rule = ota.filter_rules(_counts(), {digest: (9, 18.0)})[3]
    with pytest.raises(ValueError, match="tokens"):
        rule.statistic("ACDEFGHIKL")


def test_a_malformed_score_file_refuses(tmp_path):
    good = {"sequence_sha256": "a" * 64, "n_tokens": 4, "sum_nll": 8.0, "mean_nll": 2.0}
    def write(*rows):
        path = tmp_path / "bad.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path
    with pytest.raises(ValueError, match="twice"):
        ota.read_own_likelihood(write(good, good))
    with pytest.raises(ValueError, match="tokens, sum"):
        ota.read_own_likelihood(write({**good, "sum_nll": float("inf"), "mean_nll": float("inf")}))
    with pytest.raises(ValueError, match="sum_nll / n_tokens"):
        ota.read_own_likelihood(write({**good, "mean_nll": 3.0}))
    with pytest.raises(ValueError, match="another arm"):
        ota.merge_likelihood({"x": {"a" * 64: (4, 8.0)}, "y": {"a" * 64: (4, 8.0)}})


def test_a_pool_too_small_for_a_retained_quarter_refuses():
    members = [{"key": f"m{i}", "plddt": 60.0, "stratum": ota.STRATA[0], "score": float(i)}
               for i in range(ota.MIN_UNITS_PER_POOL - 1)]
    with pytest.raises(ValueError, match="at least"):
        ota.nested_base_draws(members, retain="highest", draws=10,
                              family=ota.LIKELIHOOD_CONTROL_FAMILIES[ota.LIKELIHOOD_RULE_NAME],
                              column=0, stream_index=0)


def test_the_likelihood_families_are_declared_distinct_and_sized():
    primary = ota.LIKELIHOOD_CONTROL_FAMILIES[ota.LIKELIHOOD_RULE_NAME]
    secondary = ota.SECONDARY_LIKELIHOOD_CONTROL_FAMILIES[ota.LIKELIHOOD_RULE_NAME]
    assert ota.FAMILY_SIZES[primary] == ota.PRIMARY_FAMILY_SIZE
    assert ota.FAMILY_SIZES[secondary] == ota.SECONDARY_FAMILY_SIZE
    others = set(ota.CONTROL_FAMILIES.values()) | set(ota.SECONDARY_CONTROL_FAMILIES.values())
    assert len({primary, secondary} | others) == len(others) + 2
    assert len({ota.FAMILY_SEED_OFFSETS[f] for f in {primary, secondary} | others}) == len(others) + 2
    # The sensitivity reads the declared family's draws, so it needs no family of its own.
    assert (ota.LIKELIHOOD_CONTROL_FAMILIES[ota.LIKELIHOOD_SENSITIVITY_NAME] == primary)
    with pytest.raises(SystemExit):
        aot.explicit_statistics_control({"pools": {}}, ota.PRIMARY_PAIRS, {}, {},
                                        ota.LIKELIHOOD_CONTROL_FAMILIES)


# ------------------------------------- the selection curve and repetitiveness (2026-11-14)

def _driver(name, directory="objective_tuned_base_likelihood_20261114"):
    import importlib.util
    path = SCRIPTS / directory / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_five_mer_run_reads_the_retained_matchers_from_their_own_source():
    matchers = _driver("run_ties_and_length", "objective_tuned_five_mer_rule_20261114"
                       ).load_matchers()
    parent_length, parent_value = np.array([16.0, 17.0, 40.0]), np.array([1.0, 3.0, 5.0])
    other_length, other_value = np.array([16.0, 40.0, 100.0]), np.array([4.0, 7.0, 9.0])
    # Sixteen-residue bins: the two matched bins differ by 2 each, and the 100-residue
    # row has no parent in its bin and is dropped.
    assert matchers["bin"](parent_length, parent_value, other_length, other_value) == (2.0, 2, 3)
    # Nearest parent length with a caliper of 5: differences 3 and 2, and the 100-residue row,
    # whose nearest parent is 60 residues away, is dropped.
    assert matchers["nn"](parent_length, parent_value, other_length, other_value) == (2.5, 2)


def test_the_cut_group_subsample_is_compared_with_its_whole_group():
    represent = _driver("run_ties_and_length", "objective_tuned_five_mer_rule_20261114"
                        ).representativeness
    rows = [{"key": f"m{index}", "score": 1.0, "length": 100.0 + index, "plddt": 50.0 + index}
            for index in range(10)] + [{"key": "x", "score": 0.5, "length": 1.0, "plddt": 1.0}]
    balanced = represent(rows, {f"m{index}" for index in (0, 4, 5, 9)}, 1.0)
    assert balanced == {"z_length_retained_in_group": pytest.approx(0.0),
                        "z_plddt_retained_in_group": pytest.approx(0.0)}
    # The four lowest members of a ten-member group sit well below its mean; the lower-scoring
    # member is not part of the group at the cut.
    low = represent(rows, {f"m{index}" for index in range(4)}, 1.0)
    assert low["z_length_retained_in_group"] < -2.0
    # Nothing is subsampled where the whole group is retained.
    assert represent(rows, {f"m{index}" for index in range(10)}, 1.0) == {
        "z_length_retained_in_group": None, "z_plddt_retained_in_group": None}


def test_the_match_point_is_the_smallest_k_not_wholly_below_one():
    curve = _driver("run_selection_curve")
    cell = lambda status, finding: {"status": status, "finding": finding}  # noqa: E731
    below = ota.BOUNDED_NEGATIVE
    assert curve.match_point({2: cell("resolved", below), 4: cell("resolved", ota.UNRESOLVED),
                              8: cell("resolved", ota.SUPPORTED)}) == 4
    assert curve.match_point({2: cell("resolved", below), 4: cell("resolved", below),
                              8: cell("unresolved", None)}) is None
    # An unresolved cell is not a match, whatever its finding field holds.
    assert curve.match_point({2: cell("unresolved", ota.UNRESOLVED)}) is None


def test_the_retained_count_follows_the_selection_fraction():
    members = [{"key": f"m{i:03d}", "sequence": "A", "score": float(i)} for i in range(80)]
    rule = ota.filter_rules(_counts())[0]
    assert [ota.rank_and_retain(members, rule, fraction=f)["n_retained"]
            for f in (0.5, 0.25, 0.125)] == [40, 20, 10]


def test_the_cut_records_how_much_of_the_retained_set_the_tie_break_decided():
    rule = ota.filter_rules(_counts())[0]

    def pool(scores):
        return [{"key": f"m{index}", "sequence": "A", "score": score}
                for index, score in enumerate(scores)]

    # Ranked: one member beats the cut, and the cut group is told apart by its key.
    ranked = ota.rank_and_retain(pool([3.0, 2.0, 2.0, 1.0]), rule, fraction=0.5)
    assert (ranked["cut_score"], ranked["n_tied_at_cut"], ranked["n_retained_at_cut"]) == (2.0, 2, 1)
    assert ranked["retained_keys"] == {"m0", "m1"}
    # Saturated: the retained set lies wholly inside the group at the best score, so it is
    # the key-ordered members of the group. Keys compare as strings, which is why
    # "m10" is kept before "m2".
    members = pool([1.0] * 12 + [0.5] * 4)
    saturated = ota.rank_and_retain(members, rule, fraction=0.25)
    assert (saturated["cut_score"], saturated["n_tied_at_cut"],
            saturated["n_retained_at_cut"], saturated["n_retained"]) == (1.0, 12, 4, 4)
    assert saturated["retained_keys"] == {"m0", "m1", "m10", "m11"}
    # Nothing retained, nothing cut.
    nothing = ota.rank_and_retain(pool([1.0, 2.0]), rule, fraction=0.25)
    assert nothing["cut_score"] is None
    assert nothing["n_tied_at_cut"] == nothing["n_retained_at_cut"] == 0


def test_the_repetitiveness_statistics_and_rank_correlation():
    repeat = _driver("run_likelihood_repetitiveness")
    assert repeat.longest_run("AAKKKKLM") == 4 and repeat.longest_run("") == 0
    x = np.arange(50.0)
    assert repeat.spearman(x, x ** 3) == pytest.approx(1.0)
    assert repeat.spearman(x, -x) == pytest.approx(-1.0)
    assert np.isnan(repeat.spearman(x, np.ones(50)))


def test_the_outside_fraction_counts_both_sides_and_reads_against_the_declared_thresholds():
    support = _driver("run_support_containment")
    rng = np.random.default_rng(0)
    base = np.linspace(1.0, 2.0, 1000)
    inside = support.outside(base, np.linspace(1.2, 1.8, 400), rng)
    assert inside["n_below_range"] == inside["n_above_range"] == 0 and inside["reading"] == "inside"
    tuned = np.concatenate([np.linspace(1.2, 1.8, 300), np.full(60, 0.5), np.full(40, 2.5)])
    split = support.outside(base, tuned, rng)
    assert (split["n_below_range"], split["n_above_range"]) == (60, 40)
    assert split["fraction"] == pytest.approx(0.25) and split["reading"] == "outside"
    assert split["exchangeable_expectation"] == pytest.approx(2 / 1001)
