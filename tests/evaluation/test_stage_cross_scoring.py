"""Conditions the E18 stage cross-scoring adapter and likelihood extension must satisfy.

These cases decide whether the 2x2 means anything. The pool a cell scores must be
the frozen one, not a resample and not a fresh draw; both scorers of a generating
arm must read byte-identical members; a conditioned stage-1 pool must not be
invented to square the design; a stale or edited artefact must be refused rather
than adapted with borrowed provenance; and the count expected must come from the
freeze rather than from a fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts.capability.evaluation import prepare_stage_cross_scoring as pscs
from scripts.capability.evaluation import score_generated_likelihood as scorer
from src.capability.evaluation import generated_phenotype as gp
from src.capability.generation import stage_pair as sp

SEQUENCES = {
    "stage_1": ("MACDEFGHIKL", "MNPQRSTVWY"),
    "stage_2": ("MGGGACDEFGH", "MWWYYACDEFG"),
}
STREAMS = ("replicate_1", "replicate_2")


def frozen_record(stage: str, stream: str, index: int) -> dict:
    sequence = SEQUENCES[stage][index % len(SEQUENCES[stage])]
    return {
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


def build_pair(tmp_path: Path, *, per_stream: int = 2, records=None):
    """A tiny, internally consistent stage-pair freeze and sequences file.

    ``per_stream`` is declared into the freeze and realised in the records, so the
    declared count is genuinely derived rather than assumed by the adapter.
    """

    if records is None:
        records = [
            frozen_record(stage, stream, index)
            for stream in STREAMS
            for stage in sp.STAGES
            for index in range(per_stream)
        ]
    sequences = tmp_path / "stage_pair_sequences.jsonl"
    sequences.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    freeze = {
        "schema_version": sp.SCHEMA_VERSION,
        "status": "complete",
        "created_utc": "2026-10-08T00:00:00Z",
        "records_file": str(sequences),
        "records_sha256": hashlib.sha256(sequences.read_bytes()).hexdigest(),
        "n_sequences": len(records),
        "per_stream_target": per_stream,
        "per_stage": {
            stage: {"arm": sp.STAGES[stage], "n_sequences": per_stream * len(STREAMS)}
            for stage in sp.STAGES
        },
        "per_stream": [
            {
                "stage": stage,
                "stream": stream,
                "arm": sp.STAGES[stage],
                "n_selected": per_stream,
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
            "declared": {
                "condition": "unconditioned",
                "prompt": "Seq=<",
                "temperature": 0.85,
            },
        },
        "inputs": [
            {
                "stage": "stage_2",
                "stream": "replicate_1",
                "path": "ledger",
                "sha256": "a" * 64,
            }
        ],
        "no_new_sampling": "the retained cells were reused; none was sampled",
        "ceiling": {
            "conditional_comparison_is_impossible": (
                "stage 1 has no instruction interface, so a conditioned counterpart "
                "cannot be constructed"
            )
        },
    }
    freeze_path = tmp_path / "stage_pair_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=1, sort_keys=True), encoding="utf-8"
    )
    return freeze_path, sequences


def adapt(freeze: Path, sequences: Path, out: Path) -> dict:
    return pscs.run(
        argparse.Namespace(freeze=freeze, sequences=sequences, out=out, device="cpu")
    )


def rewrite_freeze_digest(sequences: Path, freeze: Path) -> None:
    """Re-point the freeze at edited bytes, so a later check is the one under test."""

    record = json.loads(freeze.read_text(encoding="utf-8"))
    record["records_sha256"] = hashlib.sha256(sequences.read_bytes()).hexdigest()
    freeze.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------- legacy default


def test_pool_identity_defaults_to_the_scorer_and_is_read_by_generator(tmp_path):
    """Without --pool-arm the E17 behaviour is unchanged, including the role filter."""

    assert scorer.resolve_pool_arm("protgpt2", None) == "protgpt2"
    assert scorer.resolve_pool_arm("prollama", "prollama-stage-1") == "prollama-stage-1"

    freeze, sequences = build_pair(tmp_path)
    record = adapt(freeze, sequences, tmp_path / "out")
    cohort = Path(record["cohort_jsonl"])

    from_arm = scorer.load_pool(cohort, "prollama")
    assert [row["arm"] for row in from_arm] == ["prollama"] * len(from_arm)
    assert from_arm == sorted(from_arm, key=lambda row: row["id"])
    with pytest.raises(SystemExit):
        scorer.load_pool(cohort, "protgpt2")

    # A record that is not a pool member is not read, exactly as before.
    extra = (
        cohort.read_text(encoding="utf-8")
        + json.dumps(
            {"id": "nat_1", "arm": "prollama", "roles": ["natural"], "sequence": "MAA"}
        )
        + "\n"
    )
    cohort.write_text(extra, encoding="utf-8")
    assert all("pool" in row["roles"] for row in scorer.load_pool(cohort, "prollama"))
    assert len(scorer.load_pool(cohort, "prollama")) == len(from_arm)


# ------------------------------------------------------ pools are scorer-blind


def test_both_scorer_choices_read_one_pool_per_generator(tmp_path):
    freeze, sequences = build_pair(tmp_path)
    record = adapt(freeze, sequences, tmp_path / "out")

    generators = sorted(record["pools"])
    for generator in generators:
        assert record["pools"][generator]["candidate_key"] == "id"
        assert record["pools"][generator]["independence_unit"] == "stream"
        assert record["pools"][generator]["n_streams"] == len(STREAMS)
        rows = scorer.load_pool(Path(record["cohort_jsonl"]), generator)
        assert gp.cohort_digest(rows) == record["pools"][generator]["pool_digest"]

        licenses = []
        for scoring_arm in sorted(sp.STAGES.values()):
            scorer.require_cross_arm_supported(scoring_arm, generator, rows)
            licenses.append((scoring_arm, gp.cohort_digest(rows)))
        assert len({digest for _, digest in licenses}) == 1

        cells = [cell for cell in record["cells"] if cell["generator_arm"] == generator]
        assert len(cells) == len(sp.STAGES)
        assert {cell["pool_digest"] for cell in cells} == {
            record["pools"][generator]["pool_digest"]
        }
        assert {cell["scorer_arm"] for cell in cells} == set(sp.STAGES.values())

    assert len(record["cells"]) == 4
    assert {cell["generator_arm"] for cell in record["cells"]} == set(
        sp.STAGES.values()
    )


def test_a_cross_arm_cell_outside_the_stage_pair_is_refused():
    rows = [{"id": "x", "condition": "unconditioned"}]
    with pytest.raises(SystemExit):
        scorer.require_cross_arm_supported("protgpt2", "zymctrl", rows)
    with pytest.raises(SystemExit):
        scorer.require_cross_arm_supported("prollama", "protgpt2", rows)


# ---------------------------------------------------- no conditioned stage 1


def test_a_conditioned_freeze_is_refused_rather_than_adapted(tmp_path):
    freeze, sequences = build_pair(tmp_path)
    record = json.loads(freeze.read_text(encoding="utf-8"))
    record["decoding"]["declared"]["condition"] = "requested"
    freeze.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")

    with pytest.raises(SystemExit, match="condition"):
        adapt(freeze, sequences, tmp_path / "out")


def test_a_conditioning_label_on_a_record_is_refused(tmp_path):
    records = [
        frozen_record(stage, stream, index)
        for stream in STREAMS
        for stage in sp.STAGES
        for index in range(2)
    ]
    records[0]["class_key"] = "1.1.1.-"
    freeze, sequences = build_pair(tmp_path, records=records)

    with pytest.raises(SystemExit, match="class_key"):
        adapt(freeze, sequences, tmp_path / "out")

    # The scorer refuses the same thing on an already-built pool.
    with pytest.raises(SystemExit, match="conditioning"):
        scorer.require_cross_arm_supported(
            "prollama", "prollama-stage-1", [{"id": "x", "condition": "requested"}]
        )


def test_the_adapted_cohort_never_carries_a_condition(tmp_path):
    freeze, sequences = build_pair(tmp_path)
    record = adapt(freeze, sequences, tmp_path / "out")
    rows = [
        json.loads(line)
        for line in Path(record["cohort_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert all(row["condition"] == "unconditioned" for row in rows)
    assert all(row.get("class_key") in (None, "") for row in rows)
    assert "conditional_comparison_is_impossible" in record["provenance"]["ceiling"]


# -------------------------------------------------------------- stale inputs


def test_edited_sequences_are_refused_against_the_freeze_digest(tmp_path):
    freeze, sequences = build_pair(tmp_path)
    adapt(freeze, sequences, tmp_path / "out")

    sequences.write_text(sequences.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match the digest"):
        adapt(freeze, sequences, tmp_path / "out2")


def test_an_edited_sequence_with_a_stale_digest_is_refused(tmp_path):
    """Even with the freeze re-pointed, a record whose bytes moved is not adapted."""

    freeze, sequences = build_pair(tmp_path)
    rows = [
        json.loads(line)
        for line in sequences.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["sequence"] = rows[0]["sequence"] + "A"
    rows[0]["length"] = len(rows[0]["sequence"])
    sequences.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    rewrite_freeze_digest(sequences, freeze)

    with pytest.raises(SystemExit, match="sequence_sha256"):
        adapt(freeze, sequences, tmp_path / "out")


# ------------------------------------------------------- missing/duplicate ids


def test_a_missing_member_is_refused(tmp_path):
    freeze, sequences = build_pair(tmp_path)
    kept = [
        line
        for line in sequences.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:-1]
    sequences.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    rewrite_freeze_digest(sequences, freeze)
    with pytest.raises(SystemExit, match="declared frozen set"):
        adapt(freeze, sequences, tmp_path / "out")


def test_a_repeated_identifier_is_refused(tmp_path):
    freeze, sequences = build_pair(tmp_path)
    rows = [
        json.loads(line)
        for line in sequences.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.append(dict(rows[0]))
    sequences.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    rewrite_freeze_digest(sequences, freeze)
    with pytest.raises(SystemExit, match="repeats"):
        adapt(freeze, sequences, tmp_path / "out")


def test_an_empty_pool_is_refused(tmp_path):
    freeze, sequences = build_pair(tmp_path)
    sequences.write_text("", encoding="utf-8")
    rewrite_freeze_digest(sequences, freeze)
    with pytest.raises(SystemExit, match="no record"):
        adapt(freeze, sequences, tmp_path / "out")


def test_a_stage_arm_disagreement_is_refused(tmp_path):
    freeze, sequences = build_pair(tmp_path)
    rows = [
        json.loads(line)
        for line in sequences.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["stage"] == "stage_1"
    rows[0]["arm"] = sp.STAGES["stage_2"]
    sequences.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    rewrite_freeze_digest(sequences, freeze)
    with pytest.raises(SystemExit, match="stage and arm disagree"):
        adapt(freeze, sequences, tmp_path / "out")


# ------------------------------------------------------------- conservation


def test_sequences_ids_stages_and_properties_are_preserved(tmp_path):
    freeze, sequences = build_pair(tmp_path)
    source = [
        json.loads(line)
        for line in sequences.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = adapt(freeze, sequences, tmp_path / "out")
    cohort = [
        json.loads(line)
        for line in Path(record["cohort_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert len(cohort) == len(source)
    assert [row["id"] for row in cohort] == [row["id"] for row in source]
    by_id = {row["id"]: row for row in cohort}
    for original in source:
        adapted = by_id[original["id"]]
        assert adapted["sequence"] == original["sequence"]
        assert adapted["sequence"].encode("utf-8") == original["sequence"].encode(
            "utf-8"
        )
        assert adapted["sequence_sha256"] == original["sequence_sha256"]
        assert adapted["length"] == original["length"] == len(adapted["sequence"])
        assert adapted["stage"] == original["stage"]
        assert adapted["stream"] == original["stream"]
        assert adapted["arm"] == original["arm"]
        assert adapted["properties"] == original["properties"]
        assert adapted["roles"] == ["pool"]
    assert record["n_records"] == len(source)
    assert record["conservation"]["resampling"].startswith("none")


# --------------------------------------------------- count is declared, not fixed


def test_the_expected_count_is_derived_from_the_declared_artifact(tmp_path):
    freeze, sequences = build_pair(tmp_path, per_stream=2)
    record = adapt(freeze, sequences, tmp_path / "out")
    assert record["n_records"] == 8
    assert record["n_by_stage"] == {"stage_1": 4, "stage_2": 4}
    assert record["pools"][sp.STAGES["stage_1"]]["n_streams"] == len(STREAMS)

    # A freeze whose per-stream declaration overstates what it carries is refused,
    # which is the property a hardcoded 1800/900 could not test on a fixture.
    lying = json.loads(freeze.read_text(encoding="utf-8"))
    for block in lying["per_stream"]:
        block["n_selected"] = 5
    for stage in lying["per_stage"]:
        lying["per_stage"][stage]["n_sequences"] = 10
    lying["n_sequences"] = 20
    freeze.write_text(json.dumps(lying, indent=1, sort_keys=True), encoding="utf-8")
    with pytest.raises(SystemExit, match="declared frozen set"):
        adapt(freeze, sequences, tmp_path / "out2")


# --------------------------------------------------------- executable CPU route


@pytest.fixture
def native_runtime(monkeypatch):
    """Real BareBlockScorer and CPU forward; only checkpoint/tokenisation are tiny."""

    import torch

    from src.capability.core import arms
    from src.capability.models import joint_lineage as lineage
    from src.capability.models.joint_modes import RenderedProtein, rendering

    state = SimpleNamespace(
        context=64, forwards=0, loaded=[], zero_span=False, nonfinite=False
    )

    class Tokenisation:
        declaration = rendering(lineage.RENDERING_FAMILY)

        def render(self, sequence):
            # A merged residue piece and a tail piece, behind four wrapper ids.
            pieces = (5, 6) if len(sequence) > 1 else (5,)
            block = RenderedProtein(
                rendering=self.declaration.name,
                variant="declared",
                text=self.declaration.render_protein(sequence),
                token_ids=(1, 2, 3, 4, *pieces, 7),
                scored_positions=tuple(range(4, 4 + len(pieces))),
                n_residues=len(sequence),
            )
            return replace(block, scored_positions=()) if state.zero_span else block

    class Model:
        device = "cpu"

        def __call__(self, *, input_ids, attention_mask):
            state.forwards += 1
            logits = torch.zeros((*input_ids.shape, 8))
            if state.nonfinite:
                logits.fill_(float("nan"))
            return SimpleNamespace(logits=logits)

    def load(name, *, device, dtype):
        state.loaded.append(name)
        return lineage.LoadedRung(
            rung=lineage.rung(name),
            model=Model(),
            tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=7),
            tokenisation=cast(Any, Tokenisation()),
            context=state.context,
            facts={
                "checkpoint": str(lineage.rung(name).checkpoint),
                "context": state.context,
            },
        )

    def reject_panel(*args, **kwargs):
        pytest.fail("ProLLaMA must bypass the panel loader")

    assert not set(sp.STAGES.values()) & set(arms.PANEL)
    monkeypatch.setattr(lineage, "load_rung", load)
    monkeypatch.setattr(arms, "load_arm", reject_panel)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return state


def scoring_args(tmp_path, arm, pool_arm=None, **changes):
    cohort = tmp_path / "cohort.jsonl"
    if not cohort.exists():
        cohort.write_text(
            "".join(
                json.dumps(
                    {
                        "id": f"{generator}_1",
                        "arm": generator,
                        "roles": ["pool"],
                        "sequence": "MACD",
                        "length": 4,
                        "condition": "unconditioned",
                    }
                )
                + "\n"
                for generator in sp.STAGES.values()
            ),
            encoding="utf-8",
        )
    values = dict(
        cohort=cohort,
        arm=arm,
        pool_arm=pool_arm,
        device="cuda:0",
        dtype="bfloat16",
        max_len=64,
        batch_size=2,
        out=tmp_path / f"out_{arm}_{pool_arm}",
    )
    values.update(changes)
    return argparse.Namespace(**values)


def test_real_run_routes_all_four_cells_through_native_scorer(tmp_path, native_runtime):
    digests = {}
    for generator in sp.STAGES.values():
        for arm in sp.STAGES.values():
            args = scoring_args(tmp_path, arm, generator)
            record = scorer.run(args)
            row = json.loads(Path(record["annotations_jsonl"]).read_text())
            assert record["status"] == "complete"
            assert record["arm"] == record["scorer_arm"] == arm
            assert row["arm"] == row["pool_arm"] == generator
            assert row["scorer_arm"] == arm
            assert record["loader"].endswith("joint_lineage.load_rung")
            assert record["checkpoint"] == record["loaded_facts"]["checkpoint"]
            assert record["rendering"]["target_rule"] == "spelled_sequence_run"
            assert row["native_rendering"] == "Seq=<MACD>"
            assert row["scored_target_positions"] == [4, 5]
            assert row["scored_tokens"] == 2  # wrappers contribute no terms
            assert row["nll_sum_nats"] == pytest.approx(2 * math.log(8))
            assert row["mean_nll_per_token_nats"] == pytest.approx(math.log(8))
            assert row["mean_nll_per_residue_nats"] == pytest.approx(math.log(8) / 2)
            assert record["residue_accounting"]["residues_per_scored_token"] == 2
            digests.setdefault(generator, set()).add(record["pool_digest"])
    assert all(len(values) == 1 for values in digests.values())
    assert native_runtime.loaded == [
        arm for _ in sp.STAGES for arm in sp.STAGES.values()
    ]
    assert native_runtime.forwards == 4


@pytest.mark.parametrize("generator", sp.STAGES.values())
@pytest.mark.parametrize("arm", sp.STAGES.values())
@pytest.mark.parametrize(
    "label", [{"condition": "requested"}, {"class_key": "1.1.1.-"}]
)
def test_native_self_and_cross_refuse_conditions_before_load(
    tmp_path, native_runtime, generator, arm, label
):
    args = scoring_args(tmp_path, arm, generator)
    rows = [json.loads(line) for line in args.cohort.read_text().splitlines()]
    for row in rows:
        if row["arm"] == generator:
            row.update(label)
    args.cohort.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(SystemExit, match="conditioning"):
        scorer.run(args)
    assert native_runtime.loaded == []


@pytest.mark.parametrize("context,max_len", [(64, 6), (6, 64)])
def test_native_whole_length_guard_precedes_forward(
    tmp_path, native_runtime, context, max_len
):
    native_runtime.context = context
    args = scoring_args(tmp_path, "prollama", max_len=max_len)
    with pytest.raises(SystemExit, match="whole native rendering.*effective limit 6"):
        scorer.run(args)
    assert native_runtime.forwards == 0
    assert not (args.out / scorer.COMPLETION).exists()


def test_native_exact_budget_is_not_truncation(tmp_path, native_runtime):
    args = scoring_args(tmp_path, "prollama", max_len=7)
    assert scorer.run(args)["effective_max_len"] == 7


@pytest.mark.parametrize("failure", ["zero_span", "nonfinite", "length"])
def test_native_invalid_scores_never_publish(tmp_path, native_runtime, failure):
    args = scoring_args(tmp_path, "prollama")
    if failure == "length":
        rows = [json.loads(line) for line in args.cohort.read_text().splitlines()]
        rows[-1]["length"] = 0
        args.cohort.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        setattr(native_runtime, failure, True)
    with pytest.raises((SystemExit, RuntimeError), match="target|finite|length"):
        scorer.run(args)
    assert not (args.out / scorer.COMPLETION).exists()


@pytest.mark.parametrize("invalid_result", [None, "zero_count", "nonfinite"])
def test_legacy_panel_run_defaults_pool_and_checks_actual_score_result(
    tmp_path, monkeypatch, invalid_result
):
    import torch

    from src.capability.core import arms, scoring

    calls = []
    panel = SimpleNamespace(
        spec=SimpleNamespace(input_format="raw", tokenisation="fixture"),
        device="cpu",
        model=lambda **kwargs: SimpleNamespace(logits=torch.zeros((1, 2, 8))),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        arms, "load_arm", lambda name, **kwargs: calls.append(name) or panel
    )
    monkeypatch.setattr(arms.Cohort, "input_strings", lambda self, arm: self.records)
    monkeypatch.setattr(
        arms, "conditioning_boundary_ids", lambda *args, **kwargs: (None, None)
    )
    monkeypatch.setattr(arms, "rendering_marker_ids", lambda *args: ())
    monkeypatch.setattr(
        arms,
        "tokenize_batch",
        lambda *args: (
            torch.tensor([[1, 2]]),
            torch.tensor([[1, 1]]),
        ),
    )
    if invalid_result:
        original_scores = scoring.per_sequence_scores

        def invalid_scores(*args):
            result = original_scores(*args)
            # Keep the precheck mask positive, but fault-inject the actual score
            # result to prove its count/total is validated before division.
            if invalid_result == "zero_count":
                result[0]["token_count"] = 0
            else:
                result[0]["clean_nll_sum"] = float("nan")
            return result

        monkeypatch.setattr(scoring, "per_sequence_scores", invalid_scores)
    args = scoring_args(tmp_path, "protgpt2")
    delattr(args, "pool_arm")  # programmatic E17 callers predate this optional flag
    args.cohort.write_text(
        json.dumps(
            {
                "id": "p1",
                "arm": "protgpt2",
                "roles": ["pool"],
                "sequence": "MA",
                "length": 2,
            }
        )
        + "\n"
    )
    if invalid_result:
        with pytest.raises(SystemExit, match="after causal shift|not finite"):
            scorer.run(args)
        assert not (args.out / scorer.COMPLETION).exists()
    else:
        record = scorer.run(args)
        assert record["pool_arm"] == record["scorer_arm"] == "protgpt2"
        assert record["rendering"]["input_format"] == "raw"
    assert calls == ["protgpt2"]


REAL_FREEZE = Path("results/R6/stage_pair_20261008/stage_pair_freeze.json")
REAL_SEQUENCES = Path("results/R6/stage_pair_20261008/stage_pair_sequences.jsonl")


@pytest.mark.skipif(
    not (REAL_FREEZE.is_file() and REAL_SEQUENCES.is_file()), reason="E18 set absent"
)
def test_the_real_e18_set_adapts_at_its_own_declared_size(tmp_path):
    """The expected count is read off the freeze, so the real set is checked against itself."""

    freeze = json.loads(REAL_FREEZE.read_text(encoding="utf-8"))
    declared = {
        (block["stage"], block["stream"]): block["n_selected"]
        for block in freeze["per_stream"]
    }
    expected = sum(declared.values())
    record = adapt(REAL_FREEZE, REAL_SEQUENCES, tmp_path / "out")

    assert record["n_records"] == expected == freeze["n_sequences"]
    assert record["n_by_stage"] == {
        stage: freeze["per_stage"][stage]["n_sequences"] for stage in sp.STAGES
    }
    for arm, pool in record["pools"].items():
        rows = scorer.load_pool(Path(record["cohort_jsonl"]), arm)
        assert len(rows) == pool["n_members"]
        assert gp.cohort_digest(rows) == pool["pool_digest"]
