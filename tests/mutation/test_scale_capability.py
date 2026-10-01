"""Descriptive ProGen2 scale expansion: alphabet, opt-in, alignment, gates."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts/capability") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts/capability"))

from src.capability.core import arms as A  # noqa: E402
from src.capability.core.arms import (  # noqa: E402
    PANEL,
    PROTEIN_SCALE_LADDER,
    STAGED_ARMS,
    STAGED_SCALE_ARMS,
    Arm,
    output_logit_width,
    require_scoring_target_ids,
    scoring_target_alphabet,
)
from src.capability.core.budget import record_statistics  # noqa: E402
from src.capability.core.budget import ScoredTokens  # noqa: E402


from scripts.capability.entrypoints import stage_path

def _load_stage(filename: str):
    path = stage_path(REPO_ROOT, filename)
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Config:
    def __init__(self, vocab_size=None, *, missing=False, vocab_size_lm_head=None):
        self._missing = missing
        if not missing:
            self.vocab_size = vocab_size
        if vocab_size_lm_head is not None:
            self.vocab_size_lm_head = vocab_size_lm_head

    def __getattribute__(self, name):
        if name == "vocab_size" and object.__getattribute__(self, "_missing"):
            raise AttributeError(
                "config.vocab_size was read on the ProGen2 path: progen2-xlarge "
                "declares no such key"
            )
        return object.__getattribute__(self, name)


class _Tokenizer:
    def decode(self, ids):
        return "A" * len(list(ids))


def _arm(spec, *, vocab_size=32, missing=False):
    return Arm(
        spec=spec,
        model=SimpleNamespace(config=_Config(vocab_size, missing=missing)),
        tokenizer=_Tokenizer(),
        device="cpu",
        dtype="float32",
    )


# ------------------------------------------------------------- panel invariant


def test_default_panel_does_not_admit_the_staged_scale_rungs():
    assert "progen2-large" not in PANEL
    assert "progen2-xlarge" not in PANEL
    assert STAGED_SCALE_ARMS == ("progen2-large", "progen2-xlarge")
    assert PROTEIN_SCALE_LADDER[:2] == ("progen2-small", "progen2-medium")
    for name in STAGED_SCALE_ARMS:
        assert name in STAGED_ARMS
        assert name not in PANEL


def test_panel_default_reads_config_vocab_size():
    spec = replace(PANEL["progen2-medium"], scoring_target_alphabet_size=None)
    record = scoring_target_alphabet(spec, _Config(32))
    assert record["size"] == 32
    assert record["source"] == A.SCORING_TARGET_ALPHABET_CONFIG


def test_staged_rungs_declare_thirty_two_and_ignore_a_51200_config():
    spec = STAGED_ARMS["progen2-large"]
    record = scoring_target_alphabet(spec, _Config(51200))
    assert record["size"] == 32
    assert record["source"] == A.SCORING_TARGET_ALPHABET_DECLARED
    xlarge = scoring_target_alphabet(STAGED_ARMS["progen2-xlarge"])
    assert xlarge["size"] == 32


def test_a_missing_declaration_does_not_guess_tokenizer_length():
    spec = replace(PANEL["progen2-medium"], scoring_target_alphabet_size=None)
    with pytest.raises(ValueError, match="tokenizer length is not a substitute"):
        scoring_target_alphabet(spec, _Config(missing=True))
    with pytest.raises(ValueError, match="tokenizer length is not a substitute"):
        scoring_target_alphabet(spec, None)


def test_a_non_positive_declared_alphabet_is_refused():
    with pytest.raises(ValueError, match="positive"):
        replace(STAGED_ARMS["progen2-large"], scoring_target_alphabet_size=0)


def test_target_ids_outside_the_declared_alphabet_are_refused():
    alphabet = scoring_target_alphabet(STAGED_ARMS["progen2-large"])
    require_scoring_target_ids(np.array([0, 31], dtype=np.int64), alphabet, arm="progen2-large")
    with pytest.raises(ValueError, match="outside the scoring-target alphabet"):
        require_scoring_target_ids(
            np.array([0, 32], dtype=np.int64), alphabet, arm="progen2-large"
        )


def test_record_statistics_uses_the_declared_alphabet_not_config_width():
    spec = STAGED_ARMS["progen2-large"]
    arm = _arm(spec, vocab_size=51200)
    scored = ScoredTokens(
        target_ids=np.array([1, 2, 3], dtype=np.int64),
        nll_nats=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        sequence_index=np.array([0, 0, 0], dtype=np.int64),
    )
    records = record_statistics(arm, scored)
    assert records.vocab_size == 32


def test_record_statistics_refuses_an_out_of_range_target():
    arm = _arm(STAGED_ARMS["progen2-xlarge"], missing=True)
    scored = ScoredTokens(
        target_ids=np.array([31, 32], dtype=np.int64),
        nll_nats=np.array([0.1, 0.2], dtype=np.float64),
        sequence_index=np.array([0, 0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="outside the scoring-target alphabet"):
        record_statistics(arm, scored)


# ------------------------------------------------------------- stage 01 opt-in


def test_cohort_power_refuses_staged_rungs_without_the_opt_in():
    stage = _load_stage("cohort_power.py")
    args = argparse.Namespace(kind="protein", with_ec=False, allow_staged_scale_arms=False)
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms(["progen2-large"], args)
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms(["progen2-xlarge"], args)


def test_cohort_power_opt_in_accepts_only_ladder_staged_rungs():
    stage = _load_stage("cohort_power.py")
    args = argparse.Namespace(kind="protein", with_ec=False, allow_staged_scale_arms=True)
    stage.validate_arms(["progen2-large", "progen2-xlarge"], args)
    stage.validate_arms(["progen2-medium", "progen2-large"], args)
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms(["not-a-model"], args)


def test_cohort_power_default_arms_stay_inside_the_panel():
    stage = _load_stage("cohort_power.py")
    names = stage.default_arms("protein", with_ec=False)
    assert "progen2-large" not in names
    assert "progen2-xlarge" not in names
    assert all(name in PANEL for name in names)


def test_cohort_power_stage_contract_does_not_claim_staged_admission():
    stage = _load_stage("cohort_power.py")
    record = stage._staged_scale_record(["progen2-large"], True)
    assert record["not_panel_admission"] is True
    assert record["measured_staged_arms"] == ["progen2-large"]
    assert record["scoring_target_alphabet"]["progen2-large"]["size"] == 32
    spec_record = stage._arm_spec_record("progen2-medium")
    assert "not_panel_admission" not in spec_record
    staged_spec = stage._arm_spec_record("progen2-large")
    assert staged_spec["not_panel_admission"] is True
    assert staged_spec["scoring_target_alphabet_size"] == 32


def test_cohort_power_staged_only_contract_is_not_an_empty_panel_run():
    stage = _load_stage("cohort_power.py")
    contract = stage._cohort_power_stage_contract(["progen2-large", "progen2-xlarge"])
    assert contract["not_panel_admission"] is True
    assert contract["measured_staged_arms"] == ["progen2-large", "progen2-xlarge"]
    assert contract["note"].startswith("staged opt-in qualification")
    assert "eligible_for_this_stage" not in contract.get("arm_selection", {})
    mixed = stage._cohort_power_stage_contract(["progen2-medium", "progen2-large"])
    assert mixed["arm_selection"]["measured"] == ["progen2-medium"]
    default = stage._cohort_power_stage_contract(["progen2-medium"])
    assert default["arm_selection"]["measured"] == ["progen2-medium"]




def test_live_logit_width_is_not_the_scoring_alphabet():
    spec = STAGED_ARMS["progen2-xlarge"]
    head = SimpleNamespace(out_features=51200)
    arm = Arm(
        spec=spec,
        model=SimpleNamespace(config=_Config(missing=True), lm_head=head),
        tokenizer=_Tokenizer(),
        device="cpu",
        dtype="float32",
    )
    width = output_logit_width(arm)
    assert width["size"] == 51200
    assert width["source"] == "lm_head.out_features"
    assert scoring_target_alphabet(spec)["size"] == 32
    config_only = Arm(
        spec=spec,
        model=SimpleNamespace(
            config=_Config(missing=True, vocab_size_lm_head=51200), lm_head=None
        ),
        tokenizer=_Tokenizer(),
        device="cpu",
        dtype="float32",
    )
    fallback = output_logit_width(config_only)
    assert fallback["size"] == 51200
    assert fallback["source"] == "config.vocab_size_lm_head"


# ------------------------------------------------------------- stages 20 / 29


def test_retrieval_bound_default_arms_are_unchanged():
    stage = _load_stage("retrieval_bound.py")
    assert list(stage.ARM_CORPUS) == ["protgpt2", "progen2-medium", "progen3-112m"] or set(
        stage.ARM_CORPUS
    ) == {"protgpt2", "progen2-medium", "progen3-112m"}
    assert "progen2-large" not in stage.ARM_CORPUS
    corpus = stage.corpus_record("progen2-large")
    assert corpus["declared"] == A.arm_spec("progen2-large").pretraining_corpus
    assert "lower bound" in corpus["identification"]
    assert "progen3-112m" in stage.ARM_CORPUS


def test_designed_referent_default_arms_and_progen3_decision_are_recorded():
    """The default arm list is unchanged; ProGen3 is admitted on the N-to-C stratum.

    EXP-R2-244 scores both ProGen3 rungs on the single N-to-C direction by
    decision, so the exclusion that refused them on a stratum ground is gone and
    the decision, the stratum and each rung's corpus identification are what the
    tables have to carry instead.
    """

    stage = _load_stage("designed_referent.py")
    assert stage.DEFAULT_ARMS == (
        "protgpt2",
        "progen2-small",
        "progen2-base",
        "progen2-medium",
    )
    assert "progen3-112m" not in stage.D.EXCLUDED_ARMS
    assert "progen3-3b" not in stage.D.EXCLUDED_ARMS
    assert "progen2-large" in stage.D.ARM_IDENTIFICATION
    assert "progen2-xlarge" in stage.D.ARM_IDENTIFICATION
    assert (
        stage.D.ARM_IDENTIFICATION["progen2-large"]["identification"]
        == "unbounded_in_the_model_favouring_direction"
    )
    assert (
        stage.D.ARM_IDENTIFICATION["progen3-112m"]["identification"]
        == "undeclared_corpus_no_exclusion_possible"
    )
    assert stage._ProGen3Likelihood.scoring_stratum == "n_to_c_summed_log_likelihood"


# ------------------------------------------------------------- stage 42 fixtures


def _context_skip(assay: str, *, context: int = 1024, max_tokens: int = 3424) -> dict:
    return {
        "assay": assay,
        "context": context,
        "max_tokens": max_tokens,
        "reason": (
            "the rendered variant exceeds this arm's context; truncating would "
            "score a sequence that may not contain the mutated position"
        ),
    }


def _dms_model(
    arm: str,
    rhos: dict[str, float],
    digest: str = "d0",
    *,
    skipped: list[dict] | None = None,
    dtype: str = "bfloat16",
) -> dict:
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
        "skipped": list(skipped or []),
        "settings": {"dtype": dtype},
    }


def _lookup(rhos: dict[str, float], *, digest: str = "d0") -> dict:
    return {
        "assays": [
            {
                "assay": name,
                "cluster": f"c{index % 8}",
                "mutant_digest": digest,
                "spearman": {"lookup": 0.10, "blosum62": 0.05},
            }
            for index, name in enumerate(rhos)
        ]
    }


def _mega_model(
    arm: str,
    values: dict[str, float],
    digest: str,
    *,
    kind: str,
    dtype: str = "bfloat16",
) -> dict:
    return {
        "arm": arm,
        "cohort_sha256": digest,
        "settings": {"dtype": dtype},
        "wildtypes": {
            name: {
                "kind": kind,
                "unit": f"u{index % 8}",
                "n_variants": 40,
                "spearman": value,
            }
            for index, (name, value) in enumerate(values.items())
        },
    }


def _baselines(entries: dict[str, str], digest: str) -> dict:
    """entries maps wild-type name to kind."""

    return {
        "cohort_sha256": digest,
        "wildtypes": {
            name: {
                "kind": kind,
                "unit": f"u{index % 8}",
                "n_variants": 40,
                "spearman": {"blosum62": 0.05, "hydropathy_change": 0.02},
            }
            for index, (name, kind) in enumerate(entries.items())
        },
    }


def _design_cohort(names, digest: str) -> dict:
    return {
        "source": "cohort.json",
        "cohort_sha256": digest,
        "zero_hit_designs": sorted(names),
        "note": "test census",
    }


def _stage41_report(blocks: tuple[str, ...] = ("b0",), *, status: str = "PASS") -> dict:
    rows = []
    for block_id in blocks:
        for arm in ("progen2-medium", "progen2-large", "progen2-xlarge"):
            rows.append(
                {
                    "block_id": block_id,
                    "arm": arm,
                    "is_unigram_null_control": False,
                    "cohort_digest": f"digest-{block_id}",
                    "cohort_name": "swissprot_progen2_medium_f32",
                    "per_arm_identification_status": status,
                    "displacement_corrected_ci_95": [0.2, 0.8],
                }
            )
    return {"arm_results": rows, "summary": {"arms": []}}


def test_scale_capability_refuses_a_mutant_digest_mismatch():
    stage = _load_stage("scale_capability.py")
    assays = {f"a{i}": 0.2 for i in range(8)}
    models = {
        "progen2-medium": _dms_model("progen2-medium", assays, "aaa"),
        "progen2-large": _dms_model("progen2-large", assays, "bbb"),
        "progen2-xlarge": _dms_model("progen2-xlarge", assays, "aaa"),
    }
    with pytest.raises(ValueError, match="mutant_digest"):
        stage.align_dms(models, _lookup(assays, digest="aaa"), require_fixed_census=False)


def test_scale_capability_refuses_a_shared_missing_assay():
    stage = _load_stage("scale_capability.py")
    assays = {f"a{i}": 0.2 for i in range(8)}
    shrunk = {name: value for name, value in assays.items() if name != "a0"}
    models = {
        name: _dms_model(name, shrunk)
        for name in ("progen2-medium", "progen2-large", "progen2-xlarge")
    }
    with pytest.raises(ValueError, match="assay set disagrees"):
        stage.align_dms(models, _lookup(assays), require_fixed_census=False)


def test_scale_capability_refuses_a_shrunk_dms_census():
    stage = _load_stage("scale_capability.py")
    assays = {f"a{i}": 0.2 for i in range(8)}
    models = {
        name: _dms_model(name, assays)
        for name in ("progen2-medium", "progen2-large", "progen2-xlarge")
    }
    with pytest.raises(ValueError, match="not the fixed 217"):
        stage.align_dms(models, _lookup(assays), require_fixed_census=True)


def test_scale_capability_refuses_a_cohort_digest_mismatch():
    stage = _load_stage("scale_capability.py")
    names = [f"w{i}" for i in range(8)]
    values = {name: 0.2 for name in names}
    models = {
        "progen2-medium": _mega_model("progen2-medium", values, "one", kind="design"),
        "progen2-large": _mega_model("progen2-large", values, "two", kind="design"),
        "progen2-xlarge": _mega_model("progen2-xlarge", values, "one", kind="design"),
    }
    with pytest.raises(ValueError, match="cohort_sha256"):
        stage.align_megascale(
            models,
            _baselines({name: "design" for name in names}, "one"),
            side="design",
            baseline_name="blosum62",
            require_fixed_census=False,
        )


def test_scale_capability_refuses_a_shrunk_megascale_census():
    stage = _load_stage("scale_capability.py")
    names = [f"d{i}" for i in range(8)]
    values = {name: 0.2 for name in names}
    models = {
        name: _mega_model(name, values, "one", kind="design")
        for name in ("progen2-medium", "progen2-large", "progen2-xlarge")
    }
    with pytest.raises(ValueError, match="not the fixed 130"):
        stage.align_megascale(
            models,
            _baselines({name: "design" for name in names}, "one"),
            side="design",
            baseline_name="blosum62",
            admissible=frozenset(names),
            require_fixed_census=True,
        )


def test_scale_capability_refuses_the_wrong_rung_order():
    stage = _load_stage("scale_capability.py")
    with pytest.raises(ValueError, match="fixed as"):
        stage.require_rung_order(["progen2-large", "progen2-medium", "progen2-xlarge"])
    with pytest.raises(ValueError, match="fixed as"):
        stage.require_rung_order(["progen2-small", "progen2-medium", "progen2-large"])


def test_bootstrap_seed_is_the_declared_constant():
    stage = _load_stage("scale_capability.py")
    assert stage.DEFAULT_BOOTSTRAP_SEED == 20260825
    stage.require_frozen_bootstrap(2000, 20260825)
    with pytest.raises(ValueError, match="freezes stage 42"):
        stage.require_frozen_bootstrap(1999, 20260825)
    with pytest.raises(ValueError, match="freezes stage 42"):
        stage.require_frozen_bootstrap(2000, 1)


def test_compound_megascale_gate_needs_all_five_conditions():
    stage = _load_stage("scale_capability.py")
    positive = {"interval": [0.02, 0.10], "degenerate": False}
    negative = {"interval": [-0.02, 0.08], "degenerate": False}
    missing = {"interval": None, "degenerate": True}
    pair = "progen2-medium__progen2-large"

    def _bundle(design_delta=positive, natural_delta=negative, **overrides):
        larger = {
            "progen2-large": positive,
            "progen2-medium": positive,
            "progen2-xlarge": positive,
        }
        larger.update(overrides)
        larger = dict(larger)
        pairs = {
            "progen2-medium__progen2-large": design_delta,
            "progen2-large__progen2-xlarge": design_delta,
        }
        natural_pairs = {
            "progen2-medium__progen2-large": natural_delta,
            "progen2-large__progen2-xlarge": natural_delta,
        }
        return {
            "designs": {
                "model_minus_hydropathy": {"per_rung": dict(larger)},
                "model_minus_blosum62": {"per_rung": dict(larger)},
                "raw_spearman": {"adjacent_delta_rho": pairs},
            },
            "control": {
                "model_minus_hydropathy": {"per_rung": dict(larger)},
                "model_minus_blosum62": {"per_rung": dict(larger)},
                "raw_spearman": {"adjacent_delta_rho": natural_pairs},
            },
        }

    both_pairs = {
        "progen2-medium__progen2-large": positive,
        "progen2-large__progen2-xlarge": positive,
    }
    dms = {
        "model_minus_lookup": {
            "per_rung": {
                "progen2-medium": positive,
                "progen2-large": positive,
                "progen2-xlarge": positive,
            }
        },
        "raw_spearman": {"adjacent_delta_rho": both_pairs},
    }
    gates = stage.descriptive_gate_transitions(dms, _bundle())
    assert gates["megascale"][pair]["verdict"] is True
    assert "natural_raw_spearman_delta" not in gates["megascale"][pair]["conditions"]
    assert gates["megascale"][pair]["reported_not_gated"]["natural_raw_spearman_delta"] is False
    failed = _bundle()
    failed["designs"]["model_minus_blosum62"]["per_rung"]["progen2-large"] = negative
    assert stage.descriptive_gate_transitions(dms, failed)["megascale"][pair]["verdict"] is False
    unresolved = _bundle()
    unresolved["control"]["model_minus_hydropathy"]["per_rung"]["progen2-large"] = missing
    assert (
        stage.descriptive_gate_transitions(dms, unresolved)["megascale"][pair]["verdict"]
        == "unresolved"
    )
    assert gates["dms"][pair]["blosum62_is_not_a_dms_gate"] is True


def _small_compare_payloads():
    assays = {f"a{i}": 0.20 + 0.01 * (i % 3) for i in range(16)}
    designs = {f"d{i}": 0.15 + 0.01 * (i % 3) for i in range(16)}
    naturals = {f"n{i}": 0.12 + 0.01 * (i % 3) for i in range(16)}
    digest = "abc123"
    dms_models = {
        name: _dms_model(name, {key: value + offset for key, value in assays.items()})
        for name, offset in (
            ("progen2-medium", 0.00),
            ("progen2-large", 0.05),
            ("progen2-xlarge", 0.10),
        )
    }
    mega_models = {}
    for name, offset in (
        ("progen2-medium", 0.00),
        ("progen2-large", 0.04),
        ("progen2-xlarge", 0.08),
    ):
        design_block = _mega_model(
            name, {key: value + offset for key, value in designs.items()}, digest, kind="design"
        )
        natural_block = _mega_model(
            name, {key: value + offset for key, value in naturals.items()}, digest, kind="natural"
        )
        design_block["wildtypes"].update(natural_block["wildtypes"])
        mega_models[name] = design_block
    kinds = {name: "design" for name in designs}
    kinds.update({name: "natural" for name in naturals})
    return (
        dms_models,
        _lookup(assays),
        mega_models,
        _baselines(kinds, digest),
        _design_cohort(designs, digest),
    )


def test_compare_scale_writes_the_required_claims_and_no_total():
    stage = _load_stage("scale_capability.py")
    dms_models, lookup, mega_models, baselines, design_cohort = _small_compare_payloads()
    payload = stage.compare_scale(
        dms_models=dms_models,
        lookup=lookup,
        megascale_models=mega_models,
        baselines=baselines,
        design_cohort=design_cohort,
        fragment_order=None,
        qualification_report=_stage41_report(),
        resamples=200,
        seed=1,
        require_fixed_census=False,
    )
    assert payload["descriptive_not_causal"] is True
    assert payload["no_biological_knowledge_claim"] is True
    assert payload["not_panel_admission"] is True
    assert payload["no_cross_task_total"] is True
    assert "total_score" not in payload
    assert "descriptive_gate_transition" not in payload["dms"]["model_minus_lookup"][
        "adjacent_delta_rho"
    ]["progen2-medium__progen2-large"]
    assert "descriptive_gate_transitions" in payload
    assert payload["qualification"]["passed"] is True
    assert payload["fragment_order"] is None


def test_stage41_summary_only_is_refused():
    stage = _load_stage("scale_capability.py")
    with pytest.raises(ValueError, match="summary-only"):
        stage.qualify_stage41({"summary": {"arms": [{"arm": "progen2-medium", "status": "PASS"}]}})


def test_stage41_missing_block_is_refused():
    stage = _load_stage("scale_capability.py")
    report = _stage41_report(("b0", "b1"))
    report["arm_results"] = [
        row for row in report["arm_results"] if not (row["arm"] == "progen2-large" and row["block_id"] == "b1")
    ]
    with pytest.raises(ValueError, match="covers blocks"):
        stage.qualify_stage41(report)


def test_stage41_fail_is_refused():
    stage = _load_stage("scale_capability.py")
    with pytest.raises(ValueError, match="not PASS"):
        stage.qualify_stage41(_stage41_report(status="FAIL"))
    record = stage.qualify_stage41(_stage41_report())
    assert record["rungs"]["progen2-large"]["blocks"]["b0"]["per_arm_identification_status"] == "PASS"


def test_stage41_pass_label_cannot_override_a_nonpositive_interval():
    stage = _load_stage("scale_capability.py")
    report = _stage41_report()
    report["arm_results"][0]["displacement_corrected_ci_95"] = [-0.01, 0.8]
    with pytest.raises(ValueError, match="strictly above zero"):
        stage.qualify_stage41(report)


def test_stage41_duplicate_rung_block_is_refused():
    stage = _load_stage("scale_capability.py")
    report = _stage41_report()
    report["arm_results"].append(dict(report["arm_results"][0]))
    with pytest.raises(ValueError, match="repeats a block"):
        stage.qualify_stage41(report)


def test_fragment_order_digest_must_match():
    stage = _load_stage("scale_capability.py")
    dms_models, lookup, mega_models, baselines, design_cohort = _small_compare_payloads()
    with pytest.raises(ValueError, match="fragment_order"):
        stage.compare_scale(
            dms_models=dms_models,
            lookup=lookup,
            megascale_models=mega_models,
            baselines=baselines,
            design_cohort=design_cohort,
            fragment_order={"cohort_sha256": "other", "arms": {}},
            qualification_report=_stage41_report(),
            resamples=50,
            seed=1,
            require_fixed_census=False,
        )


# ------------------------------------------- the real stage-20 / stage-29 shapes

RETRIEVAL_BOUND_DIR = REPO_ROOT / "results/R5/retrieval_bound_20260807"
DESIGNED_REFERENT_DIR = REPO_ROOT / "results/shared/designed_referent_20260813"
REAL_DMS = RETRIEVAL_BOUND_DIR / "model_progen2-medium.json"
REAL_LOOKUP = RETRIEVAL_BOUND_DIR / "lookup.json"
REAL_MEGA = DESIGNED_REFERENT_DIR / "model_progen2-medium.json"
REAL_BASELINES = DESIGNED_REFERENT_DIR / "baselines.json"
REAL_COHORT = DESIGNED_REFERENT_DIR / "cohort.json"
REAL_FRAGMENT_ORDER = DESIGNED_REFERENT_DIR / "fragment_order.json"


def _read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _as_three_rungs(payload: dict) -> dict:
    """The same real payload in each rung slot, renamed and independent."""

    import copy

    models = {}
    for name in ("progen2-medium", "progen2-large", "progen2-xlarge"):
        rung = copy.deepcopy(payload)
        rung["arm"] = name
        models[name] = rung
    return models


@pytest.mark.skipif(
    not REAL_DMS.is_file() or not REAL_LOOKUP.is_file(),
    reason="the staged stage-20 artefacts are absent",
)
def test_the_real_stage20_payload_aligns_and_names_its_own_exclusions():
    """The positive control: what stage 20 actually wrote must align.

    ``model_progen2-medium.json`` carries 201 scored assays and 16 skips, every
    one a rendered variant longer than the 1024-position context. Every ProGen2
    rung shares that context, so this is the shape stage 42 will be handed.
    """

    stage = _load_stage("scale_capability.py")
    alignment = stage.align_dms(
        _as_three_rungs(_read_json(REAL_DMS)), _read_json(REAL_LOOKUP)
    )
    assert len(alignment["declared_assays"]) == stage.DMS_DECLARED_ASSAYS
    assert alignment["declared_clusters"] == stage.DMS_DECLARED_CLUSTERS
    assert len(alignment["analysis_assays"]) == stage.DMS_ANALYSIS_ASSAYS
    assert len(set(alignment["units"].values())) == stage.DMS_ANALYSIS_CLUSTERS
    assert alignment["context_excluded_assays"] == sorted(
        stage.DMS_CONTEXT_EXCLUDED_ASSAYS
    )
    assert set(alignment["analysis_assays"]).isdisjoint(
        alignment["context_excluded_assays"]
    )
    assert stage.require_uniform_dtype(
        _as_three_rungs(_read_json(REAL_DMS)), label="DMS"
    )


@pytest.mark.skipif(
    not REAL_MEGA.is_file() or not REAL_BASELINES.is_file() or not REAL_COHORT.is_file(),
    reason="the staged stage-29 artefacts are absent",
)
def test_the_real_stage29_design_side_is_the_130_zero_hit_designs():
    """The design census is reachable only through the cohort that carries the flag."""

    stage = _load_stage("scale_capability.py")
    models = _as_three_rungs(_read_json(REAL_MEGA))
    baselines = _read_json(REAL_BASELINES)
    census = stage.zero_hit_design_cohort(REAL_COHORT)
    assert len(census["zero_hit_designs"]) == stage.MEGASCALE_DESIGN_WILDTYPES
    assert census["cohort_sha256"] == baselines["cohort_sha256"]
    admissible = frozenset(census["zero_hit_designs"])
    units, raw, _ = stage.align_megascale(
        models,
        baselines,
        side="design",
        baseline_name="hydropathy_change",
        admissible=admissible,
    )
    assert len(raw["progen2-medium"]) == stage.MEGASCALE_DESIGN_WILDTYPES
    assert len(set(units.values())) == stage.MEGASCALE_DESIGN_SERIES
    natural_units, natural_raw, _ = stage.align_megascale(
        models, baselines, side="natural", baseline_name="hydropathy_change"
    )
    assert len(natural_raw["progen2-medium"]) == stage.MEGASCALE_NATURAL_WILDTYPES
    assert len(set(natural_units.values())) == stage.MEGASCALE_NATURAL_CLUSTERS
    # The unrestricted design side is the 146 the payload scores, which is why
    # the census cannot be read off the payload.
    unrestricted = stage._side_keys(
        _read_json(REAL_MEGA), side="design", spearman_of=lambda entry: entry.get("spearman")
    )
    assert len(unrestricted) == 146


@pytest.mark.skipif(
    not REAL_FRAGMENT_ORDER.is_file(),
    reason="the staged stage-29 fragment_order artefact is absent",
)
def test_a_fragment_order_without_the_upper_rungs_is_reported_not_refused():
    """3-7-mer margins are reported and never gated, so a missing rung is named."""

    stage = _load_stage("scale_capability.py")
    record = stage._fragment_margins(_read_json(REAL_FRAGMENT_ORDER))
    assert record["reported_not_gated"] is True
    assert record["rungs_present"] == ["progen2-medium"]
    assert record["rungs_missing"] == ["progen2-large", "progen2-xlarge"]
    assert record["incomplete_note"]
    for key, block in record["margins"]["designs"].items():
        assert set(block["per_rung"]) == {"progen2-medium"}, key


def test_a_rung_specific_skip_breaks_the_pairing_and_is_refused():
    stage = _load_stage("scale_capability.py")
    assays = {f"a{i}": 0.2 for i in range(8)}
    scored = {name: value for name, value in assays.items() if name != "a0"}
    models = {
        "progen2-medium": _dms_model(
            "progen2-medium", scored, skipped=[_context_skip("a0")]
        ),
        "progen2-large": _dms_model(
            "progen2-large", scored, skipped=[_context_skip("a0")]
        ),
        "progen2-xlarge": _dms_model("progen2-xlarge", assays),
    }
    with pytest.raises(ValueError, match="common support"):
        stage.align_dms(models, _lookup(assays), require_fixed_census=False)


def test_a_jointly_skipped_assay_is_a_cohort_definition_not_a_refusal():
    stage = _load_stage("scale_capability.py")
    assays = {f"a{i}": 0.2 for i in range(8)}
    scored = {name: value for name, value in assays.items() if name != "a0"}
    models = {
        name: _dms_model(name, scored, skipped=[_context_skip("a0")])
        for name in ("progen2-medium", "progen2-large", "progen2-xlarge")
    }
    alignment = stage.align_dms(models, _lookup(assays), require_fixed_census=False)
    assert alignment["context_excluded_assays"] == ["a0"]
    assert alignment["analysis_assays"] == sorted(scored)
    assert "a0" not in alignment["units"]


def test_a_skip_that_is_not_a_context_overflow_is_refused():
    stage = _load_stage("scale_capability.py")
    assays = {f"a{i}": 0.2 for i in range(8)}
    scored = {name: value for name, value in assays.items() if name != "a0"}
    broken = _context_skip("a0")
    broken["reason"] = "the forward pass raised"
    models = {
        name: _dms_model(name, scored, skipped=[broken])
        for name in ("progen2-medium", "progen2-large", "progen2-xlarge")
    }
    with pytest.raises(ValueError, match="scoring failure"):
        stage.align_dms(models, _lookup(assays), require_fixed_census=False)


def test_a_skip_against_another_context_is_refused():
    stage = _load_stage("scale_capability.py")
    assays = {f"a{i}": 0.2 for i in range(8)}
    scored = {name: value for name, value in assays.items() if name != "a0"}
    models = {
        name: _dms_model(
            name, scored, skipped=[_context_skip("a0", context=2048, max_tokens=3424)]
        )
        for name in ("progen2-medium", "progen2-large", "progen2-xlarge")
    }
    with pytest.raises(ValueError, match="shared 1024"):
        stage.align_dms(models, _lookup(assays), require_fixed_census=False)


def _full_census_lookup(stage) -> dict:
    """A synthetic LOOKUP with the frozen 217/174 declared census.

    The 16 context-excluded assays carry 11 families of their own and share five
    with the fillers, so removing them leaves the frozen 201/163 analysis census.
    """

    excluded = list(stage.DMS_CONTEXT_EXCLUDED_ASSAYS)
    fillers = [f"f{index:03d}" for index in range(stage.DMS_ANALYSIS_ASSAYS)]
    clusters = {
        name: f"c{index % stage.DMS_ANALYSIS_CLUSTERS}"
        for index, name in enumerate(fillers)
    }
    for index, name in enumerate(excluded):
        clusters[name] = f"x{index}" if index < 11 else f"c{index - 11}"
    rows = [
        {
            "assay": name,
            "cluster": clusters[name],
            "mutant_digest": "d0",
            "spearman": {"lookup": 0.10, "blosum62": 0.05},
        }
        for name in fillers + excluded
    ]
    return {"assays": rows}


def test_the_frozen_context_exclusions_cannot_be_swapped_for_another_set():
    stage = _load_stage("scale_capability.py")
    lookup = _full_census_lookup(stage)
    assert len(lookup["assays"]) == stage.DMS_DECLARED_ASSAYS
    assert len({row["cluster"] for row in lookup["assays"]}) == stage.DMS_DECLARED_CLUSTERS
    swapped = list(stage.DMS_CONTEXT_EXCLUDED_ASSAYS[:-1]) + ["f000"]
    scored = {
        row["assay"]: 0.2 for row in lookup["assays"] if row["assay"] not in swapped
    }
    models = {
        name: _dms_model(
            name, scored, skipped=[_context_skip(assay) for assay in swapped]
        )
        for name in ("progen2-medium", "progen2-large", "progen2-xlarge")
    }
    with pytest.raises(ValueError, match="frozen EXP-R2-224 set"):
        stage.align_dms(models, lookup, require_fixed_census=True)


def test_the_frozen_census_passes_on_the_declared_exclusions():
    stage = _load_stage("scale_capability.py")
    lookup = _full_census_lookup(stage)
    excluded = list(stage.DMS_CONTEXT_EXCLUDED_ASSAYS)
    scored = {
        row["assay"]: 0.2 for row in lookup["assays"] if row["assay"] not in excluded
    }
    models = {
        name: _dms_model(
            name, scored, skipped=[_context_skip(assay) for assay in excluded]
        )
        for name in ("progen2-medium", "progen2-large", "progen2-xlarge")
    }
    alignment = stage.align_dms(models, lookup, require_fixed_census=True)
    assert len(alignment["analysis_assays"]) == stage.DMS_ANALYSIS_ASSAYS
    assert len(set(alignment["units"].values())) == stage.DMS_ANALYSIS_CLUSTERS


def test_a_declared_zero_hit_design_missing_from_a_rung_is_refused():
    stage = _load_stage("scale_capability.py")
    names = [f"d{i}" for i in range(8)]
    values = {name: 0.2 for name in names[:-1]}
    models = {
        name: _mega_model(name, values, "one", kind="design")
        for name in ("progen2-medium", "progen2-large", "progen2-xlarge")
    }
    with pytest.raises(ValueError, match="not scored on every rung"):
        stage.align_megascale(
            models,
            _baselines({name: "design" for name in names}, "one"),
            side="design",
            baseline_name="blosum62",
            admissible=frozenset(names),
            require_fixed_census=False,
        )


def test_a_design_census_from_another_cohort_is_refused():
    stage = _load_stage("scale_capability.py")
    dms_models, lookup, mega_models, baselines, design_cohort = _small_compare_payloads()
    design_cohort["cohort_sha256"] = "not-the-scored-cohort"
    with pytest.raises(ValueError, match="different cohort"):
        stage.compare_scale(
            dms_models=dms_models,
            lookup=lookup,
            megascale_models=mega_models,
            baselines=baselines,
            design_cohort=design_cohort,
            fragment_order=None,
            qualification_report=_stage41_report(),
            resamples=50,
            seed=1,
            require_fixed_census=False,
        )


def test_a_mixed_precision_ladder_is_refused():
    stage = _load_stage("scale_capability.py")
    assays = {f"a{i}": 0.2 for i in range(8)}
    models = {
        "progen2-medium": _dms_model("progen2-medium", assays, dtype="float32"),
        "progen2-large": _dms_model("progen2-large", assays, dtype="bfloat16"),
        "progen2-xlarge": _dms_model("progen2-xlarge", assays, dtype="bfloat16"),
    }
    with pytest.raises(ValueError, match="mixed precision"):
        stage.require_uniform_dtype(models, label="DMS")
    unrecorded = {
        name: _dms_model(name, assays) for name in ("progen2-medium", "progen2-large", "progen2-xlarge")
    }
    unrecorded["progen2-large"].pop("settings")
    with pytest.raises(ValueError, match="records no scoring dtype"):
        stage.require_uniform_dtype(unrecorded, label="DMS")


def test_a_transition_reports_the_smaller_rung_without_gating_on_it():
    """A gate that required the smaller rung to fail would depend on a null.

    Both rungs clearing their own baseline contrast is the ordinary case for a
    competent pair, and the transition must still be readable there. The smaller
    rung's contrasts are reported beside the gate instead.
    """

    stage = _load_stage("scale_capability.py")
    dms_models, lookup, mega_models, baselines, design_cohort = _small_compare_payloads()
    payload = stage.compare_scale(
        dms_models=dms_models,
        lookup=lookup,
        megascale_models=mega_models,
        baselines=baselines,
        design_cohort=design_cohort,
        fragment_order=None,
        qualification_report=_stage41_report(),
        resamples=200,
        seed=1,
        require_fixed_census=False,
    )
    gates = payload["descriptive_gate_transitions"]
    pair = "progen2-medium__progen2-large"
    dms_gate = gates["dms"][pair]
    mega_gate = gates["megascale"][pair]
    assert dms_gate["verdict"] is True
    assert mega_gate["verdict"] is True
    assert not any("smaller" in name for name in dms_gate["conditions"])
    assert not any("smaller" in name for name in mega_gate["conditions"])
    assert "smaller_model_minus_lookup" in dms_gate["reported_not_gated"]
    assert "smaller_design_model_minus_hydropathy" in mega_gate["reported_not_gated"]
    assert "smaller_design_model_minus_blosum62" in mega_gate["reported_not_gated"]
    assert "not a claim that the smaller rung failed" in dms_gate["transition_means"]
    assert "not a claim that the smaller rung failed" in mega_gate["transition_means"]
    assert payload["precision"] == {
        "dms": "bfloat16",
        "megascale": "bfloat16",
        "note": stage.PRECISION_NOTE,
    }
    assert payload["dms"]["declared_cohort"]["n_assays"] == len(lookup["assays"])
    assert payload["dms"]["n_assays"] == len(lookup["assays"])
    assert payload["dms"]["context_excluded_assays"] == []
    assert payload["megascale"]["design_census"]["n_zero_hit_designs"] == len(
        design_cohort["zero_hit_designs"]
    )
