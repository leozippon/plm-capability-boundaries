"""Conditions the effective-guidance-strength measurement must always satisfy.

The math is checked on small synthetic vectors against closed forms; the two
drivers (the sampler and the teacher-forced replay) are checked against each other
on the fixture tokenisers and model of ``test_guided_decoding``, and the sampler is
checked to be blind to the measurement.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.context.potts_gauge import to_zero_sum  # noqa: E402
from src.capability.decoding import guided_decoding  # noqa: E402
from src.capability.decoding.guidance_strength import (  # noqa: E402
    RECORD_KEYS, StrengthMeter, step_divergence)
from src.capability.decoding.guided_decoding import (  # noqa: E402
    canonical_token_ids, generate_cell, replay_strength)
from test_guided_decoding import (  # noqa: E402
    SHORT, _decoder, _identity, _multi_residue_arm, _score_fn, _single_residue_arm,
    _synthetic_model)

SCRIPT = REPO_ROOT / "scripts/capability/decoding/measure_effective_guidance_strength.py"
_spec = importlib.util.spec_from_file_location("measure_strength_script", SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

ARMS = [_single_residue_arm, _multi_residue_arm]


# ------------------------------------------------------------------ the math


def _vectors(seed=0, size=12):
    rng = np.random.default_rng(seed)
    mask = rng.random(size) < 0.7
    mask[:2] = True
    return rng.normal(size=size) * 3, rng.normal(size=size), mask


def test_zero_bias_gives_exactly_zero():
    logits, _, mask = _vectors()
    assert step_divergence(logits, np.zeros_like(logits), mask) == (0.0, 0.0)


def test_closed_form_on_two_tokens():
    logits, bias, mask = np.zeros(2), np.array([np.log(3.0), 0.0]), np.ones(2, bool)
    kl, tv = step_divergence(logits, bias, mask)
    assert kl == pytest.approx(0.75 * np.log(1.5) + 0.25 * np.log(0.5), abs=1e-14)
    assert tv == pytest.approx(0.25, abs=1e-14)


def test_invariant_to_constants_a_softmax_cannot_see():
    logits, bias, mask = _vectors()
    base = step_divergence(logits, bias, mask)
    for shifted in (step_divergence(logits, bias + 7.5, mask),
                    step_divergence(logits - 1e4, bias, mask),
                    step_divergence(logits + 123.0, bias - 3.0, mask)):
        np.testing.assert_allclose(shifted, base, rtol=0, atol=1e-10)


def test_masked_tokens_contribute_nothing():
    logits, bias, mask = _vectors()
    base = step_divergence(logits, bias, mask)
    poisoned_logits, poisoned_bias = logits.copy(), bias.copy()
    poisoned_logits[~mask] = np.array([np.nan, np.inf, -np.inf, 1e9] * 8)[: (~mask).sum()]
    poisoned_bias[~mask] = 1e9
    assert step_divergence(poisoned_logits, poisoned_bias, mask) == base


def test_large_logits_do_not_overflow():
    kl, tv = step_divergence(np.array([5e4, 5e4 - 3, -5e4]), np.array([0.0, 4.0, 0.0]),
                             np.ones(3, bool))
    assert np.isfinite(kl) and kl >= 0.0 and 0.0 <= tv <= 1.0


def test_refuses_empty_admissible_set_and_non_finite_admissible_logit():
    with pytest.raises(ValueError, match="no admissible"):
        step_divergence(np.zeros(3), np.zeros(3), np.zeros(3, bool))
    with pytest.raises(ValueError, match="not finite"):
        step_divergence(np.array([0.0, np.nan]), np.zeros(2), np.ones(2, bool))


def test_per_residue_normalisation_on_a_multi_residue_token():
    logits, bias, mask = _vectors()
    kl, _ = step_divergence(logits, bias, mask)
    meter = StrengthMeter()
    meter.add_step(logits, bias, mask, residues=4)  # one BPE token carrying four residues
    meter.add_step(logits, np.zeros_like(bias), mask, residues=0)  # a line-break token
    record = meter.record()
    assert tuple(record) == RECORD_KEYS
    assert record["steps"] == 2 and record["residues"] == 4
    assert record["kl_total"] == pytest.approx(kl)
    assert record["kl_per_residue"] == pytest.approx(kl / 4)
    assert record["kl_mean_step"] == pytest.approx(kl / 2)


def test_meter_refuses_an_empty_or_residue_free_sequence():
    with pytest.raises(ValueError):
        StrengthMeter().record()
    meter = StrengthMeter()
    meter.add_step(np.zeros(2), np.zeros(2), np.ones(2, bool), residues=0)
    with pytest.raises(ValueError, match="per-residue"):
        meter.record()


# --------------------------------------------------------- the two drivers


def _cell(tmp_path, build_arm, arm_name, dose=1.0, count=4):
    model = _synthetic_model(tmp_path / "refs", "SYN", SHORT, seed=0)
    arm = build_arm()
    decoder = _decoder(arm, model, arm_name, dose)
    records = generate_cell(
        arm=arm, decoder=decoder, identity=_identity("fixture", "SYN", arm_name, dose, 0, 0, 1000),
        count=count, batch_size=2, score_fn=_score_fn)
    return arm, decoder, records


@pytest.mark.parametrize("build_arm", ARMS)
def test_sampling_is_blind_to_the_measurement(tmp_path, build_arm, monkeypatch):
    """The realised draws are identical with the meter replaced by one that records nothing."""

    _, _, measured = _cell(tmp_path, build_arm, "full")

    class Blind:
        def add_step(self, *args, **kwargs):
            pass

        def record(self):
            return {}

    monkeypatch.setattr(guided_decoding, "StrengthMeter", Blind)
    _, _, blind = _cell(tmp_path / "blind", build_arm, "full")
    for with_meter, without in zip(measured, blind, strict=True):
        for key in ("sequence", "token_ids", "step_entropy_nats", "native_loglik",
                    "guidance_bias_applied"):
            assert with_meter[key] == without[key], key


@pytest.mark.parametrize("build_arm", ARMS)
def test_replay_of_the_sampled_path_reproduces_the_sampling_record(tmp_path, build_arm):
    """One accumulator, two drivers: teacher forcing the drawn tokens gives the same numbers."""

    arm, decoder, records = _cell(tmp_path, build_arm, "full")
    for record in records:
        replayed = replay_strength(arm=arm, decoder=decoder, token_ids=record["token_ids"])
        assert replayed["steps"] == record["token_steps"]
        assert replayed["residues"] == len(record["sequence"])
        for key in RECORD_KEYS:
            assert replayed[key] == pytest.approx(record["guidance_strength"][key], abs=1e-6), key
    assert max(r["guidance_strength"]["kl_total"] for r in records) > 0.0


@pytest.mark.parametrize("build_arm", ARMS)
def test_the_native_arm_moves_nothing(tmp_path, build_arm):
    _, _, records = _cell(tmp_path, build_arm, "native", dose=0.0)
    for record in records:
        assert record["guidance_strength"]["kl_total"] == 0.0
        assert record["guidance_strength"]["tv_max_step"] == 0.0


def test_segmentation_flag_distinguishes_exact_from_bpe_reapproximation(tmp_path):
    for build_arm, expected in ((_single_residue_arm, "exact"),
                                (_multi_residue_arm, "canonical_reapproximated")):
        arm, decoder, records = _cell(tmp_path / expected, build_arm, "full", count=2)
        ids, flag = canonical_token_ids(arm, decoder, records[0]["sequence"])
        assert flag == expected
        replay_strength(arm=arm, decoder=decoder, token_ids=ids)


def test_replay_refuses_an_inadmissible_or_incomplete_path(tmp_path):
    arm, decoder, records = _cell(tmp_path, _single_residue_arm, "full", count=1)
    ids = records[0]["token_ids"]
    with pytest.raises(ValueError, match="covers"):
        replay_strength(arm=arm, decoder=decoder, token_ids=ids[:-1])
    with pytest.raises(ValueError, match="not a candidate|does not fit"):
        replay_strength(arm=arm, decoder=decoder, token_ids=[0] + list(ids[1:]))


@pytest.mark.parametrize("build_arm", ARMS)
def test_measure_record_flags_its_segmentation_and_cells_aggregate(tmp_path, build_arm):
    arm, decoder, records = _cell(tmp_path, build_arm, "full")
    rows = [{**r, "token_ids": None} for r in records]  # a merged record without token ids
    entries = [script.measure_record(arm, decoder, r) for r in rows]
    expected = "exact" if build_arm is _single_residue_arm else "canonical_reapproximated"
    assert {e["segmentation"] for e in entries} == {expected}
    recorded = script.measure_record(arm, decoder, records[0])
    assert recorded["segmentation"] == "recorded_token_ids"
    cells = script.aggregate_cells(entries)
    assert len(cells) == 1 and cells[0]["sequences"] == len(records)
    assert cells[0]["segmentation"] == {expected: len(records)}
    assert cells[0]["kl_per_residue"]["mean"] > 0.0


# --------------------------------------------------------- the script's contract


def _args(**overrides):
    values = dict(device="cpu", out=Path("out"), records=Path("r.jsonl"), references=Path("refs"),
                  protocol=Path("p.json"), gauge="raw", limit=None, shard=0, shards=1)
    return script.argparse.Namespace(**{**values, **overrides})


def test_a_missing_asset_is_refused_by_option_name(tmp_path):
    records, protocol = tmp_path / "r.jsonl", tmp_path / "p.json"
    records.write_text("{}\n")
    protocol.write_text("{}")
    with pytest.raises(SystemExit, match="--references"):
        script.require_assets(_args(records=records, protocol=protocol, references=tmp_path / "nope"))
    with pytest.raises(SystemExit, match="--records"):
        script.require_assets(_args(records=tmp_path / "nope", protocol=protocol, references=tmp_path))


def test_the_parser_is_flat_and_the_gauge_has_no_default():
    parser = script.build_parser()
    base = ["--device", "cpu", "--out", "o", "--records", "r", "--references", "d", "--protocol", "p"]
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--gauge", "centred"])
    assert parser.parse_args(base + ["--gauge", "zero-sum"]).gauge == "zero-sum"


def test_sharding_partitions_the_limited_records():
    rows = [{"id": f"{i:03d}"} for i in range(10)]
    parts = [script.select_shard(rows, limit=8, shard=k, shards=3) for k in range(3)]
    assert sorted(r["id"] for part in parts for r in part) == [f"{i:03d}" for i in range(8)]
    with pytest.raises(SystemExit, match="--shard"):
        script.require_assets(_args(shard=3, shards=3, records=SCRIPT, protocol=SCRIPT, references=REPO_ROOT))


def test_gauge_raw_is_identity_and_zero_sum_is_the_shared_convention(tmp_path):
    model = _synthetic_model(tmp_path, "SYN", SHORT, seed=0)
    assert script.apply_gauge(model, "raw") is model
    zero = script.apply_gauge(model, "zero-sum")
    h, J = to_zero_sum(model.h, model.J)
    np.testing.assert_array_equal(zero.h, h)
    np.testing.assert_array_equal(zero.J, J)
    np.testing.assert_allclose(zero.J.mean(axis=3), 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="gauge"):
        script.apply_gauge(model, "other")
