"""EXP-R2-226's ProLLaMA lineage: the declaration, the doors, and what they refuse.

The doors these pin are the ones the campaign could most easily have got wrong.
``retrieval_bound.py`` and ``designed_referent.py`` were frozen on arms
resolved through :mod:`src.capability.core.arms`, and a joint checkpoint is not an arm;
opening a stage to a name whose per-arm content is missing is the defect class
EXP-R2-225 left its own doors shut to avoid, and it surfaces as a ``KeyError`` a
long way into a scored run rather than at the door.

So: every rung either stage admits must carry its corpus identification, and
neither identification table may be indexed with a name the other door admitted.
The rest is the all-or-stop rule made executable -- an unqualified rung, a rung
qualified at another precision and a rung whose gate failed are refusals at the
door and not discoveries in an artefact.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.mutation import designed_referent as D  # noqa: E402
from src.capability.models import joint_lineage as L  # noqa: E402
from src.capability.core.arms import PANEL, STAGED_ARMS, STAGED_SCALE_ARMS  # noqa: E402
from src.capability.models.joint_modes import TOKEN_UNIT  # noqa: E402


from scripts.capability.entrypoints import stage_path

def _load_stage(filename: str):
    path = stage_path(REPO_ROOT, filename)
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


QUAL = _load_stage("adaptation_stage_qualification.py")
STAGE20 = _load_stage("retrieval_bound.py")
STAGE29 = _load_stage("designed_referent.py")


# ------------------------------------------------------------- the declaration


def test_the_lineage_is_three_rungs_in_training_order_with_two_adjacent_pairs():
    assert L.LINEAGE_RUNGS == ("llama-2-7b", "prollama-stage-1", "prollama")
    assert L.ADJACENT_PAIRS == (
        ("llama-2-7b", "prollama-stage-1"),
        ("prollama-stage-1", "prollama"),
    )
    assert sorted(L.RUNGS) == sorted(L.LINEAGE_RUNGS)
    assert [L.rung(name).directory_name for name in L.LINEAGE_RUNGS] == [
        "Llama-2-7b-hf",
        "ProLLaMA_Stage_1",
        "ProLLaMA",
    ]


def test_the_reversal_clause_is_exempt_on_the_base_rung_and_on_no_other():
    """The declared floor is a declaration, not an outcome of a failed check."""

    assert L.rung("llama-2-7b").reversal_control_applies is False
    assert L.rung("prollama-stage-1").reversal_control_applies is True
    assert L.rung("prollama").reversal_control_applies is True


def test_no_rung_of_this_lineage_is_declared_as_an_arm():
    """``joint_mode_qualification.py``'s rule, asserted where it can be broken."""

    for name in L.LINEAGE_RUNGS:
        assert name not in PANEL, name
        assert name not in STAGED_ARMS, name
        assert name not in STAGED_SCALE_ARMS, name
    for directory in ("Llama-2-7b-hf", "ProLLaMA_Stage_1", "ProLLaMA"):
        assert directory not in STAGED_ARMS, directory


def test_an_unknown_rung_is_refused_by_name():
    with pytest.raises(KeyError, match="unknown lineage rung"):
        L.rung("prollama-stage-3")


def test_the_scorer_declares_the_token_as_its_symbol_unit():
    """The tokenisation axis is carried explicitly, not inferred from the arm."""

    assert L.BareBlockScorer.symbol_unit == TOKEN_UNIT
    assert "merged multi-residue" in L.BareBlockScorer.score_description


# ------------------------------------------------- stage 20: the door and its wall


def test_stage_20_admits_every_rung_and_declares_a_corpus_for_each():
    """The door and the content behind it, checked together and not separately."""

    for name in L.LINEAGE_RUNGS:
        assert name in STAGE20.SCOREABLE_ARMS, name
        record = STAGE20.corpus_record(name)
        assert record["identification"], name
        assert record["note"], name


def test_stage_20_admits_no_arm_whose_corpus_it_does_not_declare():
    """Whatever the admitted set becomes, none of it may be a door onto a wall."""

    missing = [
        name for name in STAGE20.SCOREABLE_ARMS
        if not STAGE20.corpus_record(name).get("identification")
    ]
    assert missing == []


def test_no_rung_of_this_lineage_calls_its_lookup_a_retrieval_bound():
    """A declared corpus family is not an evidenced identity, and says so."""

    for name in L.LINEAGE_RUNGS:
        record = STAGE20.corpus_record(name)
        assert "not a retrieval bound" in record["identification"], name
    assert "CANNOT be signed" in STAGE20.corpus_record("llama-2-7b")["note"]
    for name in ("prollama-stage-1", "prollama"):
        assert "model-favouring" in STAGE20.corpus_record(name)["note"], name


def test_the_stage_20_default_run_is_unchanged_by_this_campaign():
    assert sorted(STAGE20.ARM_CORPUS) == ["progen2-medium", "progen3-112m", "protgpt2"]
    assert set(STAGE20.ARM_CORPUS).isdisjoint(L.LINEAGE_RUNGS)


# ------------------------------------------------- stage 29: the door and its wall


def test_stage_29_declares_an_identification_for_every_rung_it_admits():
    assert sorted(D.JOINT_LINEAGE_IDENTIFICATION) == sorted(L.LINEAGE_RUNGS)
    for name in L.LINEAGE_RUNGS:
        entry = D.JOINT_LINEAGE_IDENTIFICATION[name]
        assert entry["identification"], name
        assert entry["note"], name


def test_stage_29_declares_an_identification_for_every_proteingym_rendering_door():
    from src.capability.models import galactica_fitness as G
    from src.capability.models import instructprotein_fitness as IP

    expected = set(G.GALACTICA_RUNGS) | {IP.INSTRUCTPROTEIN_ARM}
    assert set(D.JOINT_RENDERING_IDENTIFICATION) == expected
    for name in expected:
        entry = D.JOINT_RENDERING_IDENTIFICATION[name]
        assert entry["identification"] in D.IDENTIFICATION_CLASSES, name
        assert entry["note"], name


def test_the_two_identification_tables_are_disjoint():
    """Neither door may be indexed with a name the other one admitted."""

    assert set(D.ARM_IDENTIFICATION).isdisjoint(D.JOINT_LINEAGE_IDENTIFICATION)
    assert set(D.ARM_IDENTIFICATION).isdisjoint(D.JOINT_RENDERING_IDENTIFICATION)
    assert set(D.JOINT_LINEAGE_IDENTIFICATION).isdisjoint(D.JOINT_RENDERING_IDENTIFICATION)
    assert set(D.JOINT_LINEAGE_IDENTIFICATION).isdisjoint(PANEL)
    assert set(D.JOINT_LINEAGE_IDENTIFICATION).isdisjoint(STAGED_SCALE_ARMS)


def test_the_base_rung_can_exclude_nothing_and_the_adapted_rungs_are_bounded():
    assert (
        D.JOINT_LINEAGE_IDENTIFICATION["llama-2-7b"]["identification"]
        == "undeclared_corpus_no_exclusion_possible"
    )
    for name in ("prollama-stage-1", "prollama"):
        assert (
            D.JOINT_LINEAGE_IDENTIFICATION[name]["identification"]
            == "unbounded_in_the_model_favouring_direction"
        ), name


def test_the_stage_29_default_run_is_unchanged_by_this_campaign():
    assert set(STAGE29.DEFAULT_ARMS).isdisjoint(L.LINEAGE_RUNGS)


# ------------------------------------------------------- the all-or-stop refusals


@pytest.mark.parametrize("stage", [STAGE20, STAGE29])
def test_a_rung_named_without_its_qualification_directory_is_refused(stage):
    import argparse

    args = argparse.Namespace(arms=["prollama"], joint_qualification_dir=None)
    with pytest.raises(ValueError, match="no --joint-qualification-dir"):
        stage._require_joint_qualification_dir(args)


@pytest.mark.parametrize("stage", [STAGE20, STAGE29])
def test_a_qualification_directory_without_a_rung_is_refused(stage):
    import argparse

    args = argparse.Namespace(arms=["protgpt2"], joint_qualification_dir=Path("/x"))
    with pytest.raises(ValueError, match="no arm in --arms is one of them"):
        stage._require_joint_qualification_dir(args)


def _qualification(tmp_path: Path, rung: str, **overrides) -> Path:
    payload = {
        "artifact": "adaptation_stage_qualification",
        "rung": rung,
        "verdict": "PASS",
        "checkpoint_facts": {"dtype_requested": "bfloat16"},
        "strict_load": {"verdict": "PASS"},
        "nll_self_check": {"verdict": "PASS"},
        "directional_reversal": {"verdict": "PASS", "cost_nats_per_scored_token": 0.14},
    }
    payload.update(overrides)
    path = tmp_path / QUAL.artefact_name(rung)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_an_absent_qualification_is_a_refusal_and_never_a_pass(tmp_path):
    with pytest.raises(FileNotFoundError, match="carries no EXP-R2-226 qualification"):
        QUAL.read_verdict(tmp_path, "prollama", dtype="bfloat16")


def test_a_failed_qualification_is_refused_with_its_verdict(tmp_path):
    _qualification(tmp_path, "prollama", verdict="FAIL")
    with pytest.raises(ValueError, match="did not qualify"):
        QUAL.read_verdict(tmp_path, "prollama", dtype="bfloat16")


def test_a_qualification_taken_at_another_precision_is_refused(tmp_path):
    _qualification(tmp_path, "prollama")
    with pytest.raises(ValueError, match="qualified at bfloat16 and this run scores"):
        QUAL.read_verdict(tmp_path, "prollama", dtype="float16")


def test_a_qualification_for_another_rung_is_refused(tmp_path):
    path = _qualification(tmp_path, "prollama")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rung"] = "prollama-stage-1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="qualifies 'prollama-stage-1', not 'prollama'"):
        QUAL.read_verdict(tmp_path, "prollama", dtype="bfloat16")


def test_a_json_that_is_not_a_qualification_artefact_is_refused(tmp_path):
    (tmp_path / QUAL.artefact_name("prollama")).write_text(
        json.dumps({"artifact": "something_else"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not an EXP-R2-226 qualification artefact"):
        QUAL.read_verdict(tmp_path, "prollama", dtype="bfloat16")


def test_a_passing_qualification_is_admitted(tmp_path):
    _qualification(tmp_path, "prollama")
    assert QUAL.read_verdict(tmp_path, "prollama", dtype="bfloat16")["verdict"] == "PASS"


# ------------------------------------------------------------ the frozen clauses


def test_the_probe_literal_is_pinned_and_a_drifted_one_is_refused(monkeypatch):
    assert QUAL.require_probe().label == "avgfp_n80"
    monkeypatch.setattr(
        QUAL, "PROBE", QUAL.Probe(
            label="avgfp_n80",
            sequence="MSKGEELFTG",
            sha256=QUAL.PROBE.sha256,
            note="truncated",
        )
    )
    with pytest.raises(ValueError, match="not the frozen"):
        QUAL.require_probe()


def test_the_repeat_tolerance_refuses_non_determinism_beyond_it():
    assert QUAL.require_repeat([1.0, 2.0], [1.0, 2.0 + 1e-9]) <= QUAL.REPEAT_MAX_ABS
    with pytest.raises(QUAL.ClauseFailure, match="maximum absolute difference") as raised:
        QUAL.require_repeat([1.0, 2.0], [1.0, 2.1])
    assert raised.value.clause == "nll_self_check"
    with pytest.raises(QUAL.ClauseFailure, match="non-finite"):
        QUAL.require_repeat([float("nan")], [float("nan")])


def test_the_reversal_floor_refuses_a_cost_at_or_below_it():
    assert QUAL.require_reversal_cost(0.1442) == 0.1442
    for cost in (0.05, 0.0, -0.0013):
        with pytest.raises(QUAL.ClauseFailure, match="not strictly >") as raised:
            QUAL.require_reversal_cost(cost)
        assert raised.value.clause == "directional_reversal"


def test_the_frozen_clause_constants_are_the_pre_registered_ones():
    assert QUAL.REVERSAL_COST_MIN == 0.05
    assert QUAL.REPEAT_MAX_ABS == 1e-6
    assert QUAL.CAMPAIGN_DTYPE == "bfloat16"


# ------------------------------------- stage 44: the ladder, the census, the pooling

STAGE44 = _load_stage("adaptation_stage_capability.py")
STAGE42 = _load_stage("scale_capability.py")


def _qualified(tmp_path: Path, *rungs: str) -> Path:
    for rung in rungs:
        _qualification(tmp_path, rung)
    return tmp_path


def test_the_full_ladder_is_three_rungs_when_every_rung_qualifies(tmp_path):
    ladder = STAGE44.resolve_ladder(_qualified(tmp_path, *L.LINEAGE_RUNGS))
    assert ladder["rungs"] == L.LINEAGE_RUNGS
    assert ladder["pairs"] == L.ADJACENT_PAIRS
    assert ladder["record"]["fallback"] is None


def test_a_stage_2_that_did_not_qualify_shortens_the_ladder_to_base_and_stage_1(tmp_path):
    """EXP-R2-226's declared branch, taken before any score exists."""

    ladder = STAGE44.resolve_ladder(_qualified(tmp_path, "llama-2-7b", "prollama-stage-1"))
    assert ladder["rungs"] == ("llama-2-7b", "prollama-stage-1")
    assert ladder["pairs"] == (("llama-2-7b", "prollama-stage-1"),)
    assert ladder["record"]["fallback"] == "base_to_stage_1_only"
    assert "carries no EXP-R2-226 qualification" in ladder["record"]["stage_2_refusal"]
    assert "instruction rendering is NOT fallen back to" in ladder["record"]["note"]


def test_the_fallback_names_the_clause_that_fired_when_the_rung_wrote_one(tmp_path):
    _qualified(tmp_path, "llama-2-7b", "prollama-stage-1")
    (tmp_path / QUAL.artefact_name("prollama")).write_text(
        json.dumps(
            {
                "artifact": "adaptation_stage_qualification",
                "rung": "prollama",
                "verdict": "FAIL",
                "failed_clause": "directional_reversal",
            }
        ),
        encoding="utf-8",
    )
    ladder = STAGE44.resolve_ladder(tmp_path)
    assert ladder["record"]["stage_2_clause"] == "directional_reversal"
    assert ladder["rungs"] == ("llama-2-7b", "prollama-stage-1")


def test_a_stage_1_that_did_not_qualify_stops_the_campaign(tmp_path):
    """The base rung alone is not a ladder, so there is no shorter fallback."""

    _qualified(tmp_path, "llama-2-7b", "prollama")
    with pytest.raises(FileNotFoundError, match="prollama-stage-1"):
        STAGE44.resolve_ladder(tmp_path)


def test_this_ladders_census_is_217_over_174_and_is_not_the_progen2_ladders(tmp_path):
    assert STAGE44.DMS_CENSUS.declared_assays == 217
    assert STAGE44.DMS_CENSUS.declared_clusters == 174
    assert STAGE44.DMS_CENSUS.analysis_assays == 217
    assert STAGE44.DMS_CENSUS.analysis_clusters == 174
    assert STAGE44.DMS_CENSUS.context_excluded_assays == ()
    assert STAGE44.LINEAGE_CONTEXT == 4096
    # And EXP-R2-224's is a different set, on a different context.
    assert STAGE42.DMS_CENSUS.analysis_assays == 201
    assert STAGE42.DMS_CENSUS.analysis_clusters == 163
    assert len(STAGE42.DMS_CENSUS.context_excluded_assays) == 16
    assert STAGE42.PROGEN2_CONTEXT == 1024


def _payload(unit: str | None) -> dict:
    settings: dict = {"dtype": "bfloat16"}
    if unit is not None:
        settings["symbol_unit_accounting"] = {
            "symbol_unit": unit,
            "residues_per_scored_token": 1.54,
        }
    return {"settings": settings}


def test_a_payload_from_a_residue_unit_family_cannot_join_this_ladder():
    """The scoring-functional axis, refused rather than asserted in prose."""

    models = {name: _payload("token") for name in L.LINEAGE_RUNGS}
    record = STAGE44.require_uniform_symbol_unit(
        models, rungs=L.LINEAGE_RUNGS, label="DMS"
    )
    assert record["symbol_unit"] == "token"
    models["prollama"] = _payload("residue")
    with pytest.raises(ValueError, match="mixed or foreign symbol units"):
        STAGE44.require_uniform_symbol_unit(models, rungs=L.LINEAGE_RUNGS, label="DMS")


def test_a_payload_that_records_no_symbol_unit_is_refused_and_never_assumed():
    models = {name: _payload("token") for name in L.LINEAGE_RUNGS}
    models["llama-2-7b"] = _payload(None)
    with pytest.raises(ValueError, match="records no symbol_unit_accounting"):
        STAGE44.require_uniform_symbol_unit(models, rungs=L.LINEAGE_RUNGS, label="DMS")


def test_a_rung_list_that_is_not_this_lineage_is_refused():
    from src.capability.models.scale_comparison import require_rungs

    with pytest.raises(ValueError, match="fixed as"):
        require_rungs(["progen2-medium", "progen2-large", "progen2-xlarge"], L.LINEAGE_RUNGS)


def test_the_bootstrap_freeze_cannot_be_moved_from_the_command_line():
    STAGE44.require_frozen_bootstrap(2000, 20260826)
    for resamples, seed in ((1000, 20260826), (2000, 20260825)):
        with pytest.raises(ValueError, match="freezes this stage at"):
            STAGE44.require_frozen_bootstrap(resamples, seed)


def _interval(low: float, high: float) -> dict:
    return {"interval": [low, high], "degenerate": False}


def test_the_dms_compound_needs_both_conditions_and_never_asks_the_earlier_rung_to_fail():
    pairs = (("llama-2-7b", "prollama-stage-1"),)
    pair = "llama-2-7b__prollama-stage-1"
    dms = {
        "model_minus_lookup": {
            "per_rung": {
                "llama-2-7b": _interval(0.01, 0.2),
                "prollama-stage-1": _interval(0.03, 0.3),
            }
        },
        "model_minus_blosum62": {
            "per_rung": {
                "llama-2-7b": _interval(-0.1, 0.1),
                "prollama-stage-1": _interval(0.02, 0.3),
            }
        },
        "raw_spearman": {"adjacent_delta_rho": {pair: _interval(0.01, 0.2)}},
    }
    gates = STAGE44.adaptation_stage_transitions(dms, None, pairs=pairs)
    assert gates["label"] == "adaptation_stage_transition"
    assert gates["dms"][pair]["verdict"] is True
    # The earlier rung clearing its own contrast does not block the transition.
    assert gates["dms"][pair]["reported_not_gated"]["earlier_model_minus_lookup"] is True
    # And either condition failing sinks it.
    dms["raw_spearman"]["adjacent_delta_rho"][pair] = _interval(-0.05, 0.1)
    assert STAGE44.adaptation_stage_transitions(dms, None, pairs=pairs)["dms"][pair][
        "verdict"
    ] is False


def test_an_unreadable_interval_makes_the_compound_unresolved_rather_than_false():
    pairs = (("llama-2-7b", "prollama-stage-1"),)
    pair = "llama-2-7b__prollama-stage-1"
    dms = {
        "model_minus_lookup": {
            "per_rung": {
                "llama-2-7b": _interval(0.01, 0.2),
                "prollama-stage-1": {"degenerate": True, "interval": None},
            }
        },
        "model_minus_blosum62": {
            "per_rung": {
                "llama-2-7b": _interval(0.0, 0.2),
                "prollama-stage-1": _interval(0.0, 0.2),
            }
        },
        "raw_spearman": {"adjacent_delta_rho": {pair: _interval(0.01, 0.2)}},
    }
    gates = STAGE44.adaptation_stage_transitions(dms, None, pairs=pairs)
    assert gates["dms"][pair]["verdict"] == "unresolved"
