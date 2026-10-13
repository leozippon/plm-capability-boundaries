"""Conditions the guided-decoding stage script must hold before it loads a checkpoint.

The protocol binding, the staged-asset resolution and the reference loading are
pure enough to test on CPU in milliseconds, and they are where a cell can go
silently wrong: a domain whose column count does not equal the protocol's ``L``, a
reference directory that was not staged, an unqualified fit, or a dose on a
component the arm does not carry. Each of those refuses here rather than
producing a plausible pool.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.context.family_potts import save_family_model  # noqa: E402
from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.decoding.guided_decoding import GUIDANCE_ARMS  # noqa: E402

STAGE = REPO_ROOT / "scripts/capability/decoding/collect_guided_decoding_pool.py"
LENGTHS = {"AAA": 7, "BBB": 9}


def _stage():
    spec = importlib.util.spec_from_file_location("guided_pool_stage", STAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage = _stage()


def _protocol(**overrides):
    protocol = {
        "schema": "coupling_guided_generation_v1",
        "model_arms": ["protgpt2", "progen2-medium"],
        "guidance_arms": list(GUIDANCE_ARMS),
        "domains": {},
        "strength_grid": [
            {"lambda_fields": 0.0, "lambda_coupling": 0.0},
            {"lambda_fields": 0.0, "lambda_coupling": 1.0},
            {"lambda_fields": 1.0, "lambda_coupling": 0.0},
            {"lambda_fields": 0.5, "lambda_coupling": 0.25},
        ],
        "sequences_per_cell": 4,
        "seed_streams": [1000, 3000],
        "permutation_seed": 20261013,
        "sampling": {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "batch_size": 2},
        "native_contract": {
            "batch_size": 1, "budget": 1024, "dtype": "float32",
            "score": "native scored-span full-sum log-likelihood (-nll_sum)",
        },
    }
    protocol.update(overrides)
    return protocol


def _write_reference(directory: Path, domain: str, length: int, *, seed: int,
                     qualified: bool = True) -> str:
    rng = np.random.default_rng(seed)
    wildtype = "".join(rng.choice(list(AA20), size=length))
    save_family_model(
        directory, domain, wildtype=wildtype, h=rng.normal(size=(length, 20)),
        pairs=rng.normal(size=(length * (length - 1) // 2, 20, 20)),
        profile_freq=rng.dirichlet(np.ones(20), size=length),
        profile_column_weight=rng.uniform(5, 50, size=length),
        qualified=qualified, qualification_failures=[] if qualified else ["not_converged"],
        extra={})
    return stage._sequence_sha(wildtype)


@pytest.fixture()
def references(tmp_path):
    directory = tmp_path / "refs"
    directory.mkdir()
    domains = {
        name: {"L": length, "wt_sha256": _write_reference(directory, name, length, seed=index)}
        for index, (name, length) in enumerate(sorted(LENGTHS.items()))
    }
    return directory, domains


def _bind(protocol, **overrides):
    call = {"model_arm": "protgpt2", "domain": "AAA", "guidance_arm": "coupling",
            "strength_index": 1, "stream_index": 0}
    call.update(overrides)
    return stage.validate_protocol(protocol, **call)


# ------------------------------------------------------------- the happy binding


def test_a_valid_protocol_binds_one_cell(references):
    _directory, domains = references
    resolved = _bind(_protocol(domains=domains))
    assert resolved["model_arm"] == "protgpt2" and resolved["domain"] == "AAA"
    assert resolved["strength"].fields == 0.0 and resolved["strength"].coupling == 1.0
    assert resolved["stream_seed"] == 1000
    assert resolved["sequences_per_cell"] == 4
    assert resolved["sampling"] == {"temperature": 1.0, "top_k": 0.0, "top_p": 1.0, "batch_size": 2}
    assert resolved["native_contract"]["dtype"] == "float32"


def test_the_parser_is_flat_and_accepts_the_queue_argument_order():
    """The campaign queue injects --device and --out before the manifest's own arguments."""

    assert "add_subparsers" not in STAGE.read_text()
    parsed = stage.build_parser().parse_args([
        "--device", "cuda:3", "--out", "/gpfs/run/cell",
        "--protocol", "/gpfs/inputs/protocol.json",
        "--family-reference-dir", "/gpfs/inputs/refs",
        "--model-arm", "protgpt2", "--domain", "AAA", "--guidance-arm", "coupling",
        "--strength-index", "1", "--stream-index", "0",
    ])
    assert parsed.device == "cuda:3" and str(parsed.out) == "/gpfs/run/cell"
    assert str(parsed.family_reference_dir) == "/gpfs/inputs/refs"
    assert parsed.strength_index == 1 and parsed.stream_index == 0
    assert parsed.guidance_arm == "coupling" and parsed.model_arm == "protgpt2"


# ------------------------------------------------------------- protocol refusals


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"schema": "something_else"}, "schema"),
        ({"model_arms": ["gpt2-large"]}, "model_arms"),
        ({"guidance_arms": ["coupling_only"]}, "guidance_arms"),
        ({"sampling": {"temperature": 0.85, "top_k": 0, "top_p": 1.0, "batch_size": 2}}, "temperature"),
        ({"sampling": {"temperature": 1.0, "top_k": 50, "top_p": 1.0, "batch_size": 2}}, "top_k"),
        ({"sampling": {"temperature": 1.0, "top_k": 0, "top_p": 0.95, "batch_size": 2}}, "top_p"),
        ({"sampling": {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "batch_size": 0}}, "positive"),
        ({"native_contract": {"batch_size": 8, "budget": 1024, "dtype": "float32", "score": "x"}},
         "native_contract.batch_size"),
        ({"native_contract": {"batch_size": 1, "budget": 1024, "dtype": "bfloat16", "score": "x"}},
         "native_contract.dtype"),
        ({"sequences_per_cell": 0}, "positive"),
        ({"strength_grid": []}, "strength_grid"),
        ({"strength_grid": [{"lambda_fields": 0.0}]}, "lambda_fields and"),
        ({"strength_grid": [{"lambda_fields": 0.0, "lambda_coupling": -1.0}]}, "non-negative"),
        ({"strength_grid": [{"lambda_fields": 0.0, "lambda_coupling": 3.0}]}, "strength axes"),
        ({"strength_grid": [{"lambda_fields": 0.0, "lambda_coupling": 1.0},
                            {"lambda_fields": 0.0, "lambda_coupling": 1.0}]}, "repeats"),
        ({"seed_streams": [1000, 1003]}, "disjoint"),
        ({"seed_streams": [1000, 1000]}, "distinct"),
    ],
)
def test_a_deviating_protocol_is_refused(references, overrides, message):
    _directory, domains = references
    with pytest.raises(ValueError, match=message):
        _bind(_protocol(domains=domains, **overrides))


def test_an_undeclared_cell_selector_is_refused(references):
    _directory, domains = references
    protocol = _protocol(domains=domains)
    with pytest.raises(ValueError, match="not declared by the protocol"):
        _bind(protocol, domain="ZZZ")
    with pytest.raises(ValueError, match="strength index"):
        _bind(protocol, strength_index=9)
    with pytest.raises(ValueError, match="stream index"):
        _bind(protocol, stream_index=9)
    with pytest.raises(ValueError, match="not declared by the protocol"):
        _bind(_protocol(domains=domains, model_arms=["progen2-medium"]))
    with pytest.raises(ValueError, match="not declared by the protocol"):
        _bind(_protocol(domains=domains, guidance_arms=["native"]))


# ------------------------------------------------------- the staged assets


def test_the_reference_directory_is_an_explicit_option(references, tmp_path):
    directory, domains = references
    assert stage.resolve_reference_dir(directory, domains) == directory.resolve()
    with pytest.raises(ValueError, match="--family-reference-dir"):
        stage.resolve_reference_dir(tmp_path / "absent", domains)
    (directory / "AAA.npz").unlink()
    with pytest.raises(ValueError, match=r"lacks \['AAA.npz'\]"):
        stage.resolve_reference_dir(directory, domains)


def test_every_declared_domain_must_bind_its_length_and_wild_type(references):
    directory, domains = references
    loaded = stage.load_declared_models(directory, domains)
    assert sorted(loaded) == sorted(LENGTHS)
    wrong_length = {**domains, "AAA": {**domains["AAA"], "L": LENGTHS["AAA"] + 1}}
    with pytest.raises(ValueError, match="column-aligned comparison"):
        stage.load_declared_models(directory, wrong_length)
    wrong_sha = {**domains, "AAA": {**domains["AAA"], "wt_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="does not bind"):
        stage.load_declared_models(directory, wrong_sha)


def test_an_unqualified_reference_is_refused_with_no_opt_out(tmp_path):
    directory = tmp_path / "refs"
    directory.mkdir()
    sha = _write_reference(directory, "AAA", 7, seed=0, qualified=False)
    with pytest.raises(ValueError, match="not qualified"):
        stage.load_declared_models(directory, {"AAA": {"L": 7, "wt_sha256": sha}})
    assert "allow_unqualified" not in STAGE.read_text()


def test_a_fresh_output_directory_is_required(tmp_path):
    out = tmp_path / "cell"
    assert stage.prepare_output_dir(out) == out
    assert stage.prepare_output_dir(out) == out  # still empty, still fine
    (out / "sequences.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="non-empty"):
        stage.prepare_output_dir(out)


def test_the_code_digest_set_names_only_present_sources():
    for relative in stage.CODE_PATHS:
        assert (REPO_ROOT / relative).is_file(), relative


def test_the_completion_schema_names_are_stable():
    assert stage.SCHEMA == "coupling_guided_generation_v1"
    assert stage.COMPLETION_FILENAME == "guided_decoding_pool.json"
    assert stage.SEQUENCES_FILENAME == "sequences.jsonl"
    assert json.loads(json.dumps(stage.SETTLED_NATIVE_CONTRACT))["dtype"] == "float32"


# ------------------------------------------- the join onto the measurement contract


def test_the_contract_projection_is_the_siblings_twelve_keys():
    from src.capability.generation import fold_recapitulation as fr

    diagnostics = {
        "id": "1UFM_protgpt2_C_g1_seed100000", "domain": "1UFM", "model": "protgpt2",
        "arm_code": "C", "guidance_arm": "coupling", "lambda_nominal": 1.0,
        "lambda_fields": 0.0, "lambda_coupling": 1.0, "strength_index": 1, "seed": 100000,
        "sequence": "ACDEF", "native_loglik": -12.5, "guidance_score": 3.25,
        "fields_score": None, "coupling_score": 3.25, "guidance_bias_applied": 3.25,
        "mean_step_entropy": 2.5, "attempt_budget": 1, "token_ids": [1, 2],
    }
    row = stage.contract_record(diagnostics)
    assert tuple(row) == fr.FIELDS
    assert row["schema_version"] == fr.SCHEMA_VERSION
    assert row["arm"] == "C" and row["lambda"] == 1.0
    assert set(row) == set(fr.FIELDS)


def test_the_dose_rule_follows_the_arms_own_components():
    from src.capability.decoding.guided_decoding import GuidanceStrength

    assert stage.require_contract_cell("coupling", GuidanceStrength(0.0, 1.0)) == "C"
    assert stage.require_contract_cell("permuted_coupling", GuidanceStrength(0.0, 1.0)) == "Cx"
    assert stage.require_contract_cell("native", GuidanceStrength(0.0, 0.0)) == "N"
    # a guided arm at zero strength is the native arm; the contract requires lambda > 0
    with pytest.raises(ValueError, match="positive lambda"):
        stage.require_contract_cell("coupling", GuidanceStrength(0.0, 0.0))
    with pytest.raises(ValueError, match="zero strength"):
        stage.require_contract_cell("native", GuidanceStrength(0.0, 1.0))


def test_a_code_the_contract_does_not_list_is_refused_naming_the_fix(monkeypatch):
    """Independent of which codes the measurement side has adopted so far."""

    from src.capability.decoding.guided_decoding import GuidanceStrength

    monkeypatch.setattr(stage.fr, "ARMS", ("N", "C"))
    monkeypatch.setattr(stage.fr, "GUIDED_ARMS", ("C",))
    assert stage.require_contract_cell("coupling", GuidanceStrength(0.0, 1.0)) == "C"
    with pytest.raises(ValueError, match="fold_recapitulation.ARMS"):
        stage.require_contract_cell("cross_domain_coupling", GuidanceStrength(0.0, 1.0))
    with pytest.raises(ValueError, match="fold_recapitulation.ARMS"):
        stage.require_contract_cell("profile", GuidanceStrength(1.0, 0.0))


def test_a_contract_that_misclassifies_an_arm_is_refused(monkeypatch):
    from src.capability.decoding.guided_decoding import GuidanceStrength

    monkeypatch.setattr(stage.fr, "ARMS", ("N", "M", "C", "F", "Pr", "Cx", "Cd"))
    monkeypatch.setattr(stage.fr, "GUIDED_ARMS", ("M", "C", "F", "Cx"))
    with pytest.raises(ValueError, match="GUIDED_ARMS"):
        stage.require_contract_cell("cross_domain_coupling", GuidanceStrength(0.0, 1.0))
    with pytest.raises(ValueError, match="GUIDED_ARMS"):
        stage.require_contract_cell("profile", GuidanceStrength(1.0, 0.0))


def test_the_declared_strength_axis_and_development_grid():
    assert stage.FIELDS_AXIS == (0.0, 0.25, 0.5, 1.0, 2.0)
    assert stage.COUPLING_AXIS == (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    assert len(stage.DEVELOPMENT_STRENGTH_GRID) == 40
    assert {"lambda_fields": 0.0, "lambda_coupling": 0.0} in stage.DEVELOPMENT_STRENGTH_GRID
    assert {"lambda_fields": 1.0, "lambda_coupling": 1.0} in stage.DEVELOPMENT_STRENGTH_GRID
    assert {"lambda_fields": 2.0, "lambda_coupling": 16.0} in stage.DEVELOPMENT_STRENGTH_GRID
    assert {"lambda_fields": 4.0, "lambda_coupling": 1.0} not in stage.DEVELOPMENT_STRENGTH_GRID


# ------------------------------------------------------------- the pool summary


def _pool(**overrides):
    record = {
        "native_loglik": -10.0, "guidance_score": 2.0, "fields_score": None,
        "coupling_score": 2.0, "guidance_bias_applied": 2.0, "mean_step_entropy": 3.0,
        "residues_per_token": 2.5,
    }
    record.update(overrides)
    return record


def test_a_pool_summary_is_null_where_the_arm_has_no_such_component():
    """The native arm has no guidance quantity at all, and a mean of nulls is not zero."""

    guided = stage.pool_summaries([_pool(), _pool(guidance_score=4.0, coupling_score=4.0)])
    assert guided["mean_guidance_score"] == 3.0
    assert guided["mean_coupling_score"] == 3.0
    assert guided["mean_fields_score"] is None
    assert guided["mean_native_loglik"] == -10.0

    native = stage.pool_summaries([
        _pool(guidance_score=None, coupling_score=None, guidance_bias_applied=0.0)])
    assert native["mean_guidance_score"] is None
    assert native["mean_coupling_score"] is None
    assert native["mean_guidance_bias_applied"] == 0.0
    assert native["mean_step_entropy_nats"] == 3.0


def test_a_partially_null_pool_is_refused():
    with pytest.raises(ValueError, match="null for part of the pool"):
        stage.pool_summaries([_pool(), _pool(coupling_score=None)])
