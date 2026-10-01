"""EXP-R2-225's second stage: context resolution, qualification, waves, the label.

What these pin is the boundary of what the campaign can honestly produce, and
they are written negative-path first because every one of the four things a
reviewer would most want refused is a thing that fails silently otherwise:

* a ladder assembled from the wrong rungs still returns numbers;
* a ladder whose rungs were scored at different precisions, or under different
  scoring conventions, still returns a paired difference -- of arithmetic or of
  estimands rather than of checkpoints;
* a cross-family row handed the reserved ``descriptive_gate_transition`` label
  reads exactly like a same-family one;
* and a config that spells its position budget in an unknown key raises
  ``AttributeError`` deep inside a stage, after the weights are on a card,
  instead of before anything loads.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts/capability") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts/capability"))

from src.capability.models import scale_comparison as C  # noqa: E402
from src.capability.core.arms import (  # noqa: E402
    EOS_BOUNDED_BOUNDARY,
    STAGED_ARMS,
    config_context_length,
    eos_bounded_rendering,
)


from scripts.capability.entrypoints import stage_path

def _load_stage(filename: str):
    path = stage_path(REPO_ROOT, filename)
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE43 = importlib.import_module('scripts.capability.stages.second_stage_capability')
QUAL = importlib.import_module('scripts.capability.mutation.second_stage_interface_qualification')
PANEL_CONTRACT = _load_stage("panel_contract.py")


# ------------------------------------------------------- context-length resolution


def test_config_context_length_reads_every_declared_spelling():
    assert config_context_length(SimpleNamespace(n_positions=1024)) == 1024
    assert config_context_length(SimpleNamespace(max_position_embeddings=2048)) == 2048
    assert config_context_length(SimpleNamespace(max_seq_len=1024)) == 1024
    assert config_context_length(SimpleNamespace(seq_length=1024)) == 1024


def test_config_context_length_prefers_the_earlier_spelling():
    """The order is a declaration, so a config carrying two resolves the same way."""

    both = SimpleNamespace(n_positions=1024, max_position_embeddings=131072)
    assert config_context_length(both) == 1024


def test_config_context_length_refuses_an_unknown_spelling():
    """The whole point of resolving it here: it raises before any weight loads."""

    with pytest.raises(AttributeError, match="declares no context length"):
        config_context_length(SimpleNamespace(context_window=4096, hidden_size=768))
    # A None is not a declaration either; it is what an unset key reads as.
    with pytest.raises(AttributeError, match="declares no context length"):
        config_context_length(SimpleNamespace(n_positions=None, max_seq_len=None))


@pytest.mark.parametrize(
    ("name", "expected"),
    [("rita-xl", 1024), ("proteinglm-7b-clm", 1024), ("qwen2.5-7b", 131072)],
)
def test_the_staged_configs_resolve_through_the_declared_order(name, expected):
    """The two spellings that used to raise, read off the checkpoints themselves."""

    config_path = STAGED_ARMS[name].path / "config.json"
    if not config_path.is_file():
        pytest.skip(f"{name} is not staged on this host")
    declared = json.loads(config_path.read_text(encoding="utf-8"))
    assert config_context_length(SimpleNamespace(**declared)) == expected


def test_the_fitness_stages_resolve_context_through_the_one_declaration():
    """Neither stage may keep a second spelling of the same fallback order."""

    for filename in ("retrieval_bound.py", "designed_referent.py"):
        source = (stage_path(REPO_ROOT, filename)).read_text(encoding="utf-8")
        assert "config_context_length(" in source, filename
        assert 'getattr(config, "n_positions"' not in source, filename


# ------------------------------------------------- the generalised truncation rule


def test_the_truncation_rule_reproduces_the_hand_applied_dispatch():
    """progen2-large takes --skip-truncation and progen2-xlarge does not.

    That split was applied by hand to the EXP-R2-224 stage-01 manifest after
    ``budget.truncation_curve`` refused three large cells with exit 1. It is now
    a rule over a declared width, and the rule has to reproduce it.
    """

    assert PANEL_CONTRACT.skips_truncation_curve("progen2-large") is True
    assert PANEL_CONTRACT.skips_truncation_curve("progen2-xlarge") is False
    assert PANEL_CONTRACT.staged_cohort_power_args("progen2-large") == (
        "--skip-truncation",
    )
    assert PANEL_CONTRACT.staged_cohort_power_args("progen2-xlarge") == ()


def test_the_rule_answers_the_second_stage_checkpoints_too():
    for name in ("qwen2.5-7b", "qwen2.5-32b"):
        assert PANEL_CONTRACT.skips_truncation_curve(name) is True
    for name in ("rita-xl", "proteinglm-7b-clm"):
        assert PANEL_CONTRACT.skips_truncation_curve(name) is False


def test_the_panel_buckets_are_unchanged_by_the_generalisation():
    items = {item.item: item for item in PANEL_CONTRACT.cohort_power_items()}
    assert items["protein_large_vocab"].arms == ("protgpt2",)
    assert items["protein_large_vocab"].extra_args == ("--skip-truncation",)
    assert items["protein_default_dtype"].extra_args == ()


@pytest.mark.parametrize("name", sorted(PANEL_CONTRACT.STAGED_OUTPUT_LOGIT_WIDTH))
def test_every_declared_output_width_is_the_checkpoints_own(name):
    """Read off config.json, so the scheduling table cannot drift from the head."""

    config_path = STAGED_ARMS[name].path / "config.json"
    if not config_path.is_file():
        pytest.skip(f"{name} is not staged on this host")
    declared = json.loads(config_path.read_text(encoding="utf-8"))
    width = declared.get("vocab_size")
    if width is None:
        # progen2-xlarge carries no vocab_size key at all.
        width = declared["vocab_size_lm_head"]
    assert PANEL_CONTRACT.STAGED_OUTPUT_LOGIT_WIDTH[name] == int(width)


def test_a_staged_arm_with_no_declared_width_is_refused_at_import():
    original = dict(PANEL_CONTRACT.STAGED_OUTPUT_LOGIT_WIDTH)
    try:
        PANEL_CONTRACT.STAGED_OUTPUT_LOGIT_WIDTH.pop("rita-xl")
        with pytest.raises(AssertionError, match="declare no output-head width"):
            PANEL_CONTRACT._check_staged_output_widths()
        PANEL_CONTRACT.STAGED_OUTPUT_LOGIT_WIDTH["rita-xl"] = 26
        PANEL_CONTRACT.STAGED_OUTPUT_LOGIT_WIDTH["not-a-checkpoint"] = 4
        with pytest.raises(AssertionError, match="not staged arms"):
            PANEL_CONTRACT._check_staged_output_widths()
    finally:
        PANEL_CONTRACT.STAGED_OUTPUT_LOGIT_WIDTH.clear()
        PANEL_CONTRACT.STAGED_OUTPUT_LOGIT_WIDTH.update(original)
    PANEL_CONTRACT._check_staged_output_widths()


# ------------------------------------------------------ the interface qualification


def test_the_probes_are_frozen_and_their_controls_are_anagrams():
    for probe in (QUAL.PROTEIN_PROBE, QUAL.TEXT_PROBE):
        assert QUAL.require_probe(probe) is probe
        assert sorted(probe.control) == sorted(probe.native)
        assert probe.control != probe.native


def test_a_drifted_probe_or_a_control_that_is_not_an_anagram_is_refused():
    from dataclasses import replace

    with pytest.raises(ValueError, match="native probe digest"):
        QUAL.require_probe(replace(QUAL.PROTEIN_PROBE, native="MKT"))
    # A control with the right digest for the wrong string: a real anagram
    # failure, since a cost measured on it would be a cost of content as well.
    other = "AAAA" + QUAL.PROTEIN_PROBE.control[4:]
    with pytest.raises(ValueError, match="control digest"):
        QUAL.require_probe(replace(QUAL.PROTEIN_PROBE, control=other))
    with pytest.raises(ValueError, match="not an anagram"):
        QUAL.require_probe(
            replace(
                QUAL.PROTEIN_PROBE,
                control=other,
                control_sha256=QUAL.sequence_digest(other),
            )
        )


def test_the_frozen_constants_are_not_the_progen2_lineages():
    """A sibling, not a parameterisation: none of ProGen2's facts is reused."""

    progen2 = _load_stage("scale_interface_qualification.py")
    assert QUAL.PROTEIN_PROBE.native != progen2.FIXED_SEQUENCE
    assert QUAL.SHUFFLE_COST_MIN != progen2.WRONG_MARKER_COST_MIN
    assert set(QUAL.REQUIRED_LIVE_WIDTH).isdisjoint(progen2.REQUIRED_LIVE_WIDTH)
    assert set(QUAL.QUALIFIABLE_ARMS).isdisjoint(progen2.SCALE_INTERFACE_ARMS)


def test_an_arm_this_campaign_does_not_stage_is_refused_by_name():
    with pytest.raises(ValueError, match="not an EXP-R2-225 checkpoint"):
        QUAL.require_qualifiable_arm("progen2-large")
    with pytest.raises(ValueError, match="progen3"):
        QUAL.require_qualifiable_arm("progen3-3b")


def test_a_live_width_other_than_the_frozen_declaration_stops_the_arm():
    assert QUAL.require_live_width("rita-xl", 26).measured is True
    assert QUAL.require_live_width("qwen2.5-32b", 152064).measured is False
    with pytest.raises(ValueError, match="live output width must be 26"):
        QUAL.require_live_width("rita-xl", 32)
    with pytest.raises(ValueError, match="no frozen live output width"):
        QUAL.require_live_width("progen2-large", 51200)


def test_a_control_that_is_not_substantially_worse_is_a_failure():
    assert QUAL.require_shuffle_cost(0.9) == 0.9
    with pytest.raises(ValueError, match="is not strictly >"):
        QUAL.require_shuffle_cost(QUAL.SHUFFLE_COST_MIN)
    with pytest.raises(ValueError, match="is not strictly >"):
        QUAL.require_shuffle_cost(-1.0)


def test_a_non_deterministic_repeat_is_a_failure():
    assert QUAL.require_repeat([1.0, 2.0], [1.0, 2.0]) == 0.0
    with pytest.raises(ValueError, match="repeat max abs diff"):
        QUAL.require_repeat([1.0, 2.0], [1.0, 2.01])
    with pytest.raises(ValueError, match="non-finite"):
        QUAL.require_repeat([float("nan")], [float("nan")])


def test_a_declared_unavailable_checkpoint_reports_unavailable_without_loading():
    def _explode(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("an unavailable checkpoint must not be loaded")

    payload = QUAL.qualify_arm(
        "proteinglm-7b-clm", device="cpu", dtype="float32", load_fn=_explode
    )
    assert payload["verdict"] == "UNAVAILABLE"
    assert payload["loaded"] is False
    assert payload["substituted"] is False
    assert "deepspeed" in payload["reason"]


def test_an_arm_loaded_without_strict_counts_is_refused():
    arm = SimpleNamespace(
        name="rita-xl",
        strict_load=None,
        spec=STAGED_ARMS["rita-xl"],
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=26)),
    )
    with pytest.raises(ValueError, match="not loaded strictly"):
        QUAL.qualify_loaded_arm(arm)


def test_a_non_pass_report_cannot_become_an_artefact():
    with pytest.raises(ValueError, match="non-PASS report"):
        QUAL.build_payload(
            "rita-xl", device="cpu", dtype="float32", report={"name": "rita-xl"}
        )
    with pytest.raises(ValueError, match="verdict"):
        QUAL.write_artefact(Path("/tmp"), {"arm": "rita-xl", "verdict": "FAIL"})


def test_one_artefact_per_arm_so_one_failure_stops_only_that_arm():
    assert QUAL.artefact_name("rita-xl") != QUAL.artefact_name("qwen2.5-7b")


# --------------------------------------------------------------- stage 43: waves


def _dms_model(
    arm: str,
    rhos: dict[str, float],
    *,
    digest: str = "d0",
    dtype: str = "bfloat16",
    stratum: str = C.STRATUM_BIDIRECTIONAL,
    self_check: dict | None = "default",
) -> dict:
    loader: dict = {}
    if self_check == "default":
        loader = {
            "self_check": {
                "checkpoint": arm,
                "nll": 1.5,
                "band": [1.4, 1.6],
                "verdict": "PASS",
            }
        }
    elif self_check is not None:
        loader = {"self_check": self_check}
    return {
        "arm": arm,
        "assays": [
            {
                "assay": name,
                "wildtype_id": f"q{index:02d}",
                "mutant_digest": digest,
                "spearman": value,
            }
            for index, (name, value) in enumerate(rhos.items())
        ],
        "skipped": [],
        "loader": loader,
        "corpus": {"declared": "profluent_protein_atlas_v1"},
        "settings": {"dtype": dtype, "scoring_stratum": stratum},
    }


def _lookup(rhos: dict[str, float], *, digest: str = "d0") -> dict:
    return {
        "assays": [
            {
                "assay": name,
                "cluster": f"c{index % 12}",
                "mutant_digest": digest,
                "spearman": {"lookup": 0.10, "blosum62": 0.05},
            }
            for index, name in enumerate(rhos)
        ]
    }


def _progen3_inputs(*, small=0.30, large=0.55, **kwargs):
    # The per-assay gain varies, because a constant offset makes every paired
    # difference identical and the group bootstrap degenerate -- which is a
    # correct "unresolved", not a gate verdict.
    names = [f"a{index:02d}" for index in range(24)]
    small_rhos = {name: small + 0.003 * index for index, name in enumerate(names)}
    large_rhos = {
        name: large + 0.003 * index + 0.02 * ((index % 5) - 2)
        for index, name in enumerate(names)
    }
    models = {
        "progen3-112m": _dms_model("progen3-112m", small_rhos, **kwargs),
        "progen3-3b": _dms_model("progen3-3b", large_rhos),
    }
    return models, _lookup(small_rhos)


def _report(models, lookup, **kwargs):
    return STAGE43.compare_second_stage(
        wave_name="protein_progen3",
        dms_models=models,
        lookup=lookup,
        require_fixed_census=False,
        **kwargs,
    )


def test_the_progen3_wave_reports_a_dms_gate_and_carries_its_stratum():
    models, lookup = _progen3_inputs()
    payload = _report(models, lookup)
    assert payload["rungs"] == ["progen3-112m", "progen3-3b"]
    assert payload["dms"]["scoring_stratum"] == C.STRATUM_BIDIRECTIONAL
    assert payload["qualification"]["source"] == "stage20_loader_self_check"
    gate = payload["descriptive_gate_transitions"]["dms"]["progen3-112m__progen3-3b"]
    assert gate["label"] == "descriptive_gate_transition"
    assert gate["verdict"] is True
    assert gate["blosum62_is_not_a_dms_gate"] is True
    # The bootstrap is not restated by this stage; it is stage 42's frozen pair.
    stage42 = _load_stage("scale_capability.py")
    assert payload["bootstrap"]["resamples"] == stage42.BOOTSTRAP_RESAMPLES
    assert payload["bootstrap"]["seed"] == stage42.DEFAULT_BOOTSTRAP_SEED


def test_a_wrong_rung_set_is_refused():
    models, lookup = _progen3_inputs()
    wrong = {"progen3-112m": models["progen3-112m"], "progen2-medium": models["progen3-3b"]}
    with pytest.raises(ValueError, match="the descriptive comparison is fixed as"):
        _report(wrong, lookup)
    reordered = {
        "progen3-3b": models["progen3-3b"],
        "progen3-112m": models["progen3-112m"],
    }
    with pytest.raises(ValueError, match="the descriptive comparison is fixed as"):
        _report(reordered, lookup)


def test_a_mixed_dtype_ladder_is_refused():
    models, lookup = _progen3_inputs(dtype="float32")
    with pytest.raises(ValueError, match="mixed precision"):
        _report(models, lookup)


def test_a_rung_that_records_no_dtype_is_refused():
    models, lookup = _progen3_inputs()
    models["progen3-112m"]["settings"].pop("dtype")
    with pytest.raises(ValueError, match="records no scoring dtype"):
        _report(models, lookup)


def test_a_mixed_or_missing_scoring_stratum_is_refused():
    models, lookup = _progen3_inputs(stratum=C.STRATUM_N_TO_C)
    with pytest.raises(ValueError, match="mixed strata"):
        _report(models, lookup)
    models, lookup = _progen3_inputs()
    models["progen3-112m"]["settings"].pop("scoring_stratum")
    with pytest.raises(ValueError, match="records no scoring_stratum"):
        _report(models, lookup)
    models, lookup = _progen3_inputs(stratum="left_to_right_ish")
    with pytest.raises(ValueError, match="not one of the declared"):
        _report(models, lookup)


def test_a_wave_scored_under_the_wrong_declared_stratum_is_refused():
    models, lookup = _progen3_inputs()
    for payload in models.values():
        payload["settings"]["scoring_stratum"] = C.STRATUM_N_TO_C
    with pytest.raises(ValueError, match="is declared on the"):
        _report(models, lookup)


def test_a_rung_without_its_progen3_self_check_is_refused():
    models, lookup = _progen3_inputs(self_check=None)
    with pytest.raises(ValueError, match="records no ProGen3 self_check"):
        _report(models, lookup)
    models, lookup = _progen3_inputs(
        self_check={"checkpoint": "progen3-112m", "verdict": "FAIL"}
    )
    with pytest.raises(ValueError, match="self-check verdict"):
        _report(models, lookup)


def test_two_rungs_self_checked_against_one_band_are_refused():
    """A band borrowed from another rung is the failure this wave's loader exists for."""

    models, lookup = _progen3_inputs()
    models["progen3-3b"]["loader"]["self_check"]["checkpoint"] = "progen3-112m"
    with pytest.raises(ValueError, match="same declared checkpoint band"):
        _report(models, lookup)


def test_the_frozen_bootstrap_cannot_be_moved_from_the_command_line():
    models, lookup = _progen3_inputs()
    with pytest.raises(ValueError, match="inherits EXP-R2-224"):
        _report(models, lookup, resamples=500)
    with pytest.raises(ValueError, match="inherits EXP-R2-224"):
        _report(models, lookup, seed=1)


# ------------------------------------------------------------- stage 43: the label


def test_a_cross_family_row_cannot_receive_the_reserved_label():
    wave = STAGE43.require_wave("text_qwen")
    with pytest.raises(ValueError, match="not a rung of wave"):
        STAGE43.require_label_eligible(wave, ("qwen2.5-7b", "galactica-30b"))
    joint = STAGE43.require_wave("joint_galactica")
    with pytest.raises(ValueError, match="not a rung of wave"):
        STAGE43.require_label_eligible(joint, ("galactica-6.7b", "qwen2.5-32b"))


def test_a_non_adjacent_pair_cannot_receive_the_reserved_label():
    wave = STAGE43.require_wave("text_qwen")
    with pytest.raises(ValueError, match="not an adjacent pair"):
        STAGE43.require_label_eligible(wave, ("qwen2.5-0.5b", "qwen2.5-32b"))
    # And the direction is part of the declaration.
    with pytest.raises(ValueError, match="not an adjacent pair"):
        STAGE43.require_label_eligible(wave, ("qwen2.5-7b", "qwen2.5-0.5b"))


def test_the_wave_b_single_points_can_never_receive_the_label():
    wave = STAGE43.require_wave("protein_progen3")
    for name in ("proteinglm-7b-clm", "rita-xl"):
        with pytest.raises(ValueError, match="report existence only"):
            STAGE43.require_label_eligible(wave, ("progen3-112m", name))
    assert set(STAGE43.SINGLE_POINTS) == {"proteinglm-7b-clm", "rita-xl"}


def test_an_adjacent_same_family_pair_is_the_only_thing_that_is_eligible():
    for wave in STAGE43.WAVES:
        for pair in wave.pairs:
            STAGE43.require_label_eligible(wave, pair)


def test_a_wave_declared_with_non_adjacent_pairs_is_refused_at_construction():
    with pytest.raises(ValueError, match="adjacent rungs in order"):
        STAGE43.Wave(
            name="bad",
            family="f",
            rungs=("a", "b", "c"),
            pairs=(("a", "c"),),
            endpoints=("dms",),
            qualification="stage41_shared_report",
            note="",
        )


# ------------------------------------------------- stage 43: undeliverable endpoints


def test_every_endpoint_a_wave_does_not_carry_says_why():
    STAGE43._check_absent_endpoints_are_declared()
    for wave in STAGE43.WAVES:
        for endpoint in STAGE43.ENDPOINTS:
            if endpoint in wave.endpoints:
                continue
            assert f"{endpoint}/{wave.family}" in STAGE43.NOT_DELIVERABLE


def test_the_two_over_assumptions_are_refused_with_their_reasons():
    """The prereg licenses both; the code refuses both, and says so by name."""

    progen3 = STAGE43.require_wave("protein_progen3")
    with pytest.raises(ValueError, match="freeze delivers no MegaScale row"):
        STAGE43.require_endpoint(progen3, "megascale")
    galactica = STAGE43.require_wave("joint_galactica")
    with pytest.raises(ValueError, match="freeze delivers no ProteinGym or MegaScale row"):
        STAGE43.require_endpoint(galactica, "dms")
    with pytest.raises(ValueError, match="freeze delivers no ProteinGym or MegaScale row"):
        STAGE43.require_endpoint(galactica, "megascale")
    qwen = STAGE43.require_wave("text_qwen")
    with pytest.raises(ValueError, match="Qwen does not enter them"):
        STAGE43.require_endpoint(qwen, "dms")


def test_the_megascale_refusal_is_the_one_the_stage_29_table_states():
    """The row is absent because of the freeze, not because of the code.

    Stage 29 admits both ProGen3 rungs and EXP-R2-244 scores them on the single
    N-to-C direction by decision. What stops a MegaScale row on this campaign is
    that no wave of it carries the endpoint.
    """

    designed = importlib.import_module("src.capability.mutation.designed_referent")
    assert "progen3-112m" not in designed.EXCLUDED_ARMS
    assert "progen3-3b" not in designed.EXCLUDED_ARMS
    assert (
        designed.ARM_IDENTIFICATION["progen3-3b"]["identification"]
        == "unbounded_in_the_model_favouring_direction"
    )
    assert (
        designed.ARM_IDENTIFICATION["progen3-112m"]["identification"]
        == "undeclared_corpus_no_exclusion_possible"
    )
    for name in ("progen3-112m", "progen3-3b"):
        assert "N-to-C stratum" in designed.ARM_IDENTIFICATION[name]["note"]
    assert "bidirectional" in STAGE43.NOT_DELIVERABLE["megascale/progen3"]


# ------------------------------------------- stage 43: context-information waves


def _per_rung_reports(rungs, *, row_arm="protein_declared", status="PASS"):
    reports = {}
    for rung in rungs:
        reports[rung] = {
            "arm_results": [
                {
                    "block_id": "b0",
                    "arm": row_arm,
                    "is_unigram_null_control": False,
                    "cohort_digest": "shared-b0",
                    "cohort_name": "protein_scored",
                    "per_arm_identification_status": status,
                    "displacement_corrected_ci_95": [0.03, 0.30],
                },
                {
                    "block_id": "b0",
                    "arm": "protein_reversed",
                    "is_unigram_null_control": False,
                    "cohort_digest": "shared-b0",
                    "cohort_name": "protein_scored",
                    "per_arm_identification_status": "FAIL",
                    "displacement_corrected_ci_95": [-0.2, 0.1],
                },
            ]
        }
    return reports


def test_a_joint_wave_qualifies_from_one_report_per_rung():
    wave = STAGE43.require_wave("joint_galactica")
    payload = STAGE43.compare_second_stage(
        wave_name=wave.name,
        per_rung_qualification=_per_rung_reports(wave.rungs),
        qualification_row_arm="protein_declared",
    )
    record = payload["qualification"]
    assert record["source"] == "stage41_arm_results_per_rung"
    assert record["row_arm"] == "protein_declared"
    assert record["passed"] is True
    assert record["is_a_gate"] is False
    assert record["label_eligible"] is False
    assert payload["descriptive_gate_transitions"]["dms"] == {}


def test_the_reversed_condition_cannot_be_paired_against_the_declared_one():
    wave = STAGE43.require_wave("joint_galactica")
    reports = _per_rung_reports(wave.rungs)
    with pytest.raises(ValueError, match="identification is 'FAIL'"):
        C.qualify_per_rung_stage41(
            reports, rungs=wave.rungs, row_arm="protein_reversed"
        )
    with pytest.raises(ValueError, match="carries no 'text_declared' rows"):
        C.qualify_per_rung_stage41(reports, rungs=wave.rungs, row_arm="text_declared")


def test_a_joint_wave_refuses_a_rung_whose_report_is_missing_or_disagrees():
    wave = STAGE43.require_wave("joint_galactica")
    reports = _per_rung_reports(wave.rungs)
    reports.pop("galactica-30b")
    with pytest.raises(ValueError, match="no stage-41 report for rungs"):
        C.qualify_per_rung_stage41(
            reports, rungs=wave.rungs, row_arm="protein_declared"
        )
    reports = _per_rung_reports(wave.rungs)
    reports["galactica-30b"]["arm_results"][0]["cohort_digest"] = "another"
    with pytest.raises(ValueError, match="cohort_digest disagrees"):
        C.qualify_per_rung_stage41(
            reports, rungs=wave.rungs, row_arm="protein_declared"
        )


def test_the_qwen_wave_qualifies_from_one_shared_report():
    wave = STAGE43.require_wave("text_qwen")
    rows = [
        {
            "block_id": "b0",
            "arm": rung,
            "is_unigram_null_control": False,
            "cohort_digest": "shared-b0",
            "cohort_name": "openwebtext",
            "per_arm_identification_status": "PASS",
            "displacement_corrected_ci_95": [0.5, 1.2],
        }
        for rung in wave.rungs
    ]
    payload = STAGE43.compare_second_stage(
        wave_name=wave.name, shared_qualification={"arm_results": rows}
    )
    assert payload["qualification"]["source"] == "stage41_arm_results"
    assert payload["qualification"]["label_eligible"] is False
    assert "dms" not in payload


def test_an_unknown_wave_is_refused():
    with pytest.raises(ValueError, match="is not an EXP-R2-225 wave"):
        STAGE43.require_wave("wave_b")


# ------------------------------------------------------ the shared library moved


def test_the_uniform_dtype_check_is_one_implementation_for_both_campaigns():
    stage42 = _load_stage("scale_capability.py")
    models = {
        name: {"settings": {"dtype": "bfloat16"}} for name in stage42.SCALE_RUNGS
    }
    assert stage42.require_uniform_dtype(models, label="DMS") == "bfloat16"
    models[stage42.SCALE_RUNGS[-1]]["settings"]["dtype"] = "float32"
    with pytest.raises(ValueError, match="mixed precision"):
        stage42.require_uniform_dtype(models, label="DMS")
    # The same function, pointed at the other campaign's rungs.
    assert C.require_uniform_dtype is not None
    assert "require_uniform_dtype" in C.__all__


# ------------------------------------------------ stage 20: the ProGen3 second rung


def test_stage_20_declares_the_progen3_rungs_without_widening_its_default_run():
    stage = _load_stage("retrieval_bound.py")
    assert set(stage.ARM_CORPUS) == {"protgpt2", "progen2-medium", "progen3-112m"}
    assert set(stage.PROGEN3_CHECKPOINTS) == {"progen3-112m", "progen3-3b"}
    assert "progen3-3b" in stage.SCOREABLE_ARMS
    record = stage.corpus_record("progen3-3b")
    assert record["declared"] == "profluent_protein_atlas_v1"
    assert "not a retrieval bound" in record["identification"]
    assert "NOT identified" in record["note"]


def test_stage_20_labels_the_two_scoring_strata_it_can_produce():
    stage = _load_stage("retrieval_bound.py")
    assert stage._ArmScorer.scoring_stratum == C.STRATUM_N_TO_C
    assert stage._ProGen3Scorer.scoring_stratum == C.STRATUM_BIDIRECTIONAL
    assert (
        stage._ArmScorer.score_description
        == "summed log-likelihood of the rendered variant's scored targets; the ids "
        "this arm's rendering declares as markers are context, not targets"
    )
    assert "bidirectional" in stage._ProGen3Scorer.score_description


def test_stage_20_still_refuses_an_arm_it_declares_nothing_for():
    stage = _load_stage("retrieval_bound.py")
    # Galactica is a new stage-20 scoring door; the frozen EXP-R2-225
    # joint_galactica wave still has no DMS path (see
    # test_the_two_over_assumptions_are_refused_with_their_reasons).
    with pytest.raises(KeyError):
        stage.corpus_record("qwen2.5-7b")
    assert "qwen2.5-7b" not in stage.SCOREABLE_ARMS
    assert "proteinglm-7b-clm" in stage.SCOREABLE_ARMS
    assert "proteinglm-7b-clm" not in stage.ARM_CORPUS


def test_stage_20_admits_rita_xl_without_widening_the_default_run():
    stage = _load_stage("retrieval_bound.py")
    assert set(stage.ARM_CORPUS) == {"protgpt2", "progen2-medium", "progen3-112m"}
    assert "rita-xl" in stage.SCOREABLE_ARMS
    assert "rita-xl" not in stage.ARM_CORPUS
    record = stage.corpus_record("rita-xl")
    assert record["declared"] == "uniref100"
    assert record["identification"] == (
        "external UniRef50 profile baseline, not a retrieval bound"
    )
    assert "NOT identified" in record["note"]
    assert "does not mint a fitness PASS" in record["note"]


def test_stage_20_routes_rita_xl_with_dtype_passed_through(monkeypatch):
    stage = _load_stage("retrieval_bound.py")
    captured: dict[str, object] = {}

    class _StopBeforeWeights:
        def __init__(self, *, checkpoint, device, dtype, batch_size):
            captured["checkpoint"] = checkpoint
            captured["device"] = device
            captured["dtype"] = dtype
            captured["batch_size"] = batch_size
            raise RuntimeError("stop-before-weights")

    monkeypatch.setattr(stage.RITA, "RitaFitnessScorer", _StopBeforeWeights)
    args = SimpleNamespace(device="cpu", dtype="bfloat16", batch_size=8)
    with pytest.raises(RuntimeError, match="stop-before-weights"):
        stage._load_scorer("rita-xl", args)
    assert captured["dtype"] == "bfloat16"
    assert captured["checkpoint"] == STAGED_ARMS["rita-xl"].path
    assert captured["device"] == "cpu"
    assert captured["batch_size"] == 8

    args_f32 = SimpleNamespace(device="cpu", dtype="float32", batch_size=1)
    with pytest.raises(RuntimeError, match="stop-before-weights"):
        stage._load_scorer("rita-xl", args_f32)
    assert captured["dtype"] == "float32"
    assert captured["checkpoint"] == STAGED_ARMS["rita-xl"].path


def test_stage_20_native_extension_arms_bind_csv_and_wildtypes_hashes(tmp_path):
    stage = _load_stage("retrieval_bound.py")
    gym = tmp_path / "gym"
    gym.mkdir()
    csv_path = gym / "assay_a.csv"
    csv_path.write_text(
        "mutant,mutated_sequence,DMS_score,DMS_score_bin\nM1A,AKT,0.1,1\n",
        encoding="utf-8",
    )
    wildtypes = tmp_path / "wildtypes.json"
    wildtypes.write_text("{}" + "\n", encoding="utf-8")
    args = SimpleNamespace(proteingym_dir=gym, out=tmp_path)
    assert stage._native_extension_csv_sha256("protgpt2", "assay_a", args) is None
    assert stage._native_extension_input_fingerprints("protgpt2", args) is None
    csv_hash = stage._native_extension_csv_sha256("galactica-1.3b", "assay_a", args)
    assert csv_hash == stage.sha256_file(csv_path)
    assert stage._native_extension_csv_sha256("rita-xl", "assay_a", args) == csv_hash
    fingerprints = stage._native_extension_input_fingerprints("rita-xl", args)
    assert fingerprints == {"wildtypes_sha256": stage.sha256_file(wildtypes)}


# ------------------------------------- the qualification end to end, on a stub arm


class _ResidueTokenizer:
    """RITA's shape: one id per residue, an appended end-of-sequence id, no pad.

    The document boundary and the terminator are one id, as they are on the
    staged checkpoint, and the tokenizer reports no eos, pad or unk id of its own
    -- so the boundary is resolved by token string rather than from an attribute.
    """

    ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
    EOS = 25

    def __init__(self) -> None:
        self.eos_token_id = None
        self.pad_token_id = None
        self.unk_token_id = None

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.EOS if str(token) == EOS_BOUNDED_BOUNDARY else 0

    def __call__(self, text, return_tensors=None):
        rendered = str(text)
        boundary = []
        if rendered.startswith(EOS_BOUNDED_BOUNDARY):
            boundary = [self.EOS]
            rendered = rendered[len(EOS_BOUNDED_BOUNDARY) :]
        ids = (
            boundary
            + [self.ALPHABET.index(symbol) + 1 for symbol in rendered]
            + [self.EOS]
        )
        return {"input_ids": ids}


class _BigramModel:
    """A deterministic scorer that prefers the native probe's own bigrams.

    Not a model of anything; it is the smallest thing that makes the three
    properties the qualification asserts actually testable end to end -- the
    forward pass is a pure function of its input, so the repeat check has
    something real to verify, and an order-destroying control genuinely reads
    worse, so the negative control is not vacuous.
    """

    def __init__(self, native_ids, width=26):
        import torch

        self.torch = torch
        self.width = width
        self.table = torch.zeros(width, width)
        for left, right in zip(native_ids, native_ids[1:]):
            self.table[left, right] += 4.0
        self.lm_head = SimpleNamespace(out_features=width)
        self.config = SimpleNamespace(vocab_size=width)

    def __call__(self, input_ids=None):
        rows = self.table[input_ids[0]]
        return SimpleNamespace(logits=rows.unsqueeze(0))


def _stub_rita(width=26):
    from src.capability.core.arms import Arm

    tokenizer = _ResidueTokenizer()
    native_ids = tokenizer(eos_bounded_rendering(QUAL.PROTEIN_PROBE.native))["input_ids"]
    model = _BigramModel(native_ids, width=width)
    model.lm_head = SimpleNamespace(out_features=width)
    return Arm(
        spec=STAGED_ARMS["rita-xl"],
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        dtype="float32",
        strict_load={"missing_keys": 0, "unexpected_keys": 0},
    )


def test_the_qualification_runs_end_to_end_and_the_control_costs_something():
    arm = _stub_rita()
    report = QUAL.qualify_loaded_arm(arm)
    assert report["verdict"] == "PASS"
    assert report["live_output_width"]["size"] == 26
    assert report["live_output_width"]["declaration_measured_live"] is True
    assert report["native_repeat_max_abs_diff"] == 0.0
    assert report["shuffle_cost_nats_per_target"] > QUAL.SHUFFLE_COST_MIN
    # A residue tokenizer renders an anagram to the same multiset of ids.
    assert report["control_is_a_token_multiset_permutation"] is True
    assert report["native_target_count"] == len(QUAL.PROTEIN_PROBE.native)

    payload = QUAL.build_payload("rita-xl", device="cpu", dtype="float32", report=report)
    assert payload["verdict"] == "PASS"
    assert payload["campaign"] == "EXP-R2-225"
    assert payload["is_panel_member"] is False
    assert payload["is_staged_second_stage_arm"] is True


def test_a_head_that_loads_at_another_width_stops_the_arm():
    arm = _stub_rita(width=32)
    with pytest.raises(ValueError, match="live output width must be 26"):
        QUAL.qualify_loaded_arm(arm)


def test_a_model_that_cannot_beat_its_own_anagram_is_refused():
    """The negative control is not decorative: a flat scorer fails the clause."""

    arm = _stub_rita()
    arm.model.table.zero_()
    with pytest.raises(ValueError, match="is not strictly >"):
        QUAL.qualify_loaded_arm(arm)


# --------------------------------------------- stage 29: the door that stays shut


def test_stage_29_admits_no_arm_whose_identification_it_does_not_declare():
    """The wall behind the door, asserted so it can never be opened onto.

    ``designed_referent.py`` indexes ``ARM_IDENTIFICATION[name]`` directly
    after admitting an arm, so an arm it admits without an entry there is a
    ``KeyError`` a long way into a scored run. The admitted set is that stage's
    own ``ADMITTED_PROTEIN_DOORS``, so this asserts the invariant over what the
    stage actually admits rather than over a list restated here, and it asserts
    the two identification tables stay disjoint so neither door can be indexed
    with a name the other admitted.
    """

    from src.capability.mutation import designed_referent as D
    from src.capability.core.arms import arm_spec

    stage = _load_stage("designed_referent.py")
    admitted = [
        name
        for _, members in stage.ADMITTED_PROTEIN_DOORS
        for name in members
        if arm_spec(name).modality == "protein" and name not in D.EXCLUDED_ARMS
    ]
    assert admitted, "the admitted set must not be empty or this is vacuous"
    missing = [name for name in admitted if name not in D.ARM_IDENTIFICATION]
    assert missing == [], f"admitted by stage 29 with no ARM_IDENTIFICATION: {missing}"
    assert set(D.ARM_IDENTIFICATION).isdisjoint(D.JOINT_LINEAGE_IDENTIFICATION)
    assert set(D.ARM_IDENTIFICATION).isdisjoint(D.JOINT_RENDERING_IDENTIFICATION)
    assert set(D.JOINT_LINEAGE_IDENTIFICATION).isdisjoint(D.JOINT_RENDERING_IDENTIFICATION)


def test_stage_29_door_admits_exactly_the_declared_protein_doors():
    """What the door admits, what it refuses, and on which ground.

    EXP-R2-225's and the candidate campaign's protein checkpoints are now
    reachable, and their text checkpoints in the same tuples are not; a joint
    checkpoint reached by path is in no door at all; and the ProGen3 rungs stay
    behind the exclusion that records their published convention.
    """

    from src.capability.mutation import designed_referent as D
    from src.capability.core.arms import arm_spec

    stage = _load_stage("designed_referent.py")
    for name in ("protgpt3-1.3b", "rita-xl", "proteinglm-7b-clm"):
        assert name not in stage.DEFAULT_ARMS, name
        assert stage._admitted_door(name) is not None, name
        assert arm_spec(name).modality == "protein", name
        assert name in D.ARM_IDENTIFICATION, name
    for name in ("qwen2.5-7b", "qwen2.5-32b", "qwen3-8b-base"):
        assert stage._admitted_door(name) is not None, name
        assert arm_spec(name).modality == "text", name
    for name in ("progen3-112m", "progen3-3b"):
        assert stage._admitted_door(name) == "a ProGen3 rung", name
        assert name not in D.EXCLUDED_ARMS, name
        assert name in D.ARM_IDENTIFICATION, name
    for name in ("galactica-6.7b", "instructprotein"):
        assert stage._admitted_door(name) is None, name
        assert name in D.JOINT_RENDERING_IDENTIFICATION, name
    assert "rita-xl" not in D.EXCLUDED_ARMS


def test_stage_29_scorer_routes_are_the_declared_ones():
    """Each new arm reaches the scorer its own declaration selects.

    ``_open_scorer`` keys on ``ArmSpec`` rather than on an arm name, so what has
    to hold is that each arm's declaration selects one route and that the generic
    route refuses the other two before a checkpoint is loaded -- a refusal that
    answers in milliseconds rather than hours into a cell. ``float32`` must be an
    admissible precision or the two arms that require it cannot be dispatched at
    all, and ``rita_fitness`` requires exactly that one.
    """

    from src.capability.core.arms import (
        INPUT_FORMAT_BOS_DIRECTION_SEQ,
        INPUT_FORMAT_GMASK_SOP_EOS,
        arm_spec,
    )
    from src.capability.mutation import designed_referent as D
    from src.capability.models import galactica_fitness as G
    from src.capability.models import instructprotein_fitness as IP
    from src.capability.models import rita_fitness as RITA

    stage = _load_stage("designed_referent.py")
    assert arm_spec("proteinglm-7b-clm").input_format == INPUT_FORMAT_GMASK_SOP_EOS
    assert arm_spec("rita-xl").architecture == "rita"
    assert arm_spec("protgpt3-1.3b").input_format == INPUT_FORMAT_BOS_DIRECTION_SEQ
    assert stage._ArmLikelihood.score_description.endswith(
        "markers are context, not targets"
    )
    assert stage._ProteinGLMLikelihood.scoring_stratum == C.STRATUM_N_TO_C
    assert RITA.INFERENCE_DTYPE in stage.DTYPES
    for name in ("zymctrl", "proteinglm-7b-clm", "rita-xl"):
        with pytest.raises(ValueError):
            stage._ArmLikelihood(name, device="cpu", dtype="float32", batch_size=1)
    for name in G.GALACTICA_RUNGS:
        assert name in D.JOINT_RENDERING_IDENTIFICATION, name
        assert stage._admitted_door(name) is None, name
    assert IP.INSTRUCTPROTEIN_ARM in D.JOINT_RENDERING_IDENTIFICATION
    with pytest.raises(ValueError, match="float32-only"):
        stage._GalacticaLikelihood(
            "galactica-125m", device="cpu", dtype="bfloat16", batch_size=1
        )


def test_stage_29_refuses_an_fp32_score_taken_under_a_tf32_matmul(monkeypatch):
    """The two float32 arms record the arithmetic they were taken under.

    At float32 a TF32 matmul is a different arithmetic rather than a faster one,
    so a reading taken under it is not commensurable with the one stage 20
    recorded for the same checkpoint.
    """

    from src.capability.core import precision_policy as PP

    stage = _load_stage("designed_referent.py")
    declared = PP.requested_fp32_matmul_policy()
    record = stage._require_declared_fp32_policy("probe")
    assert record["requested"] == declared
    for field in ("cuda_matmul_allow_tf32", "float32_matmul_precision"):
        assert record["observed"][field] == declared[field]
    monkeypatch.setattr(
        PP,
        "snapshot_matmul_policy",
        lambda torch: {**declared, "cuda_matmul_allow_tf32": True},
    )
    with pytest.raises(RuntimeError, match="cuda_matmul_allow_tf32"):
        stage._require_declared_fp32_policy("probe")
