"""What has to hold about the ProGen3 robustness campaign before any cell is dispatched.

The campaign is a robustness check on E20's confirmatory conclusion, so the thing
that would quietly invalidate it is not a crash: it is a cell that runs, writes a
plausible pool, and is not the cell it is labelled as. Four such failures are
possible here and each is pinned below.

**A dose that nothing calibrated.** The doses are not E20's nominal lambdas --
nominal lambda is not comparable across models -- so they are declared constants
and the generation stage refuses anything off the axis admitted for the cell's own
model arm. The protocol is therefore bound through the real stage validator, once
per cell, rather than inspected.

**A seed reused from E20.** A stream that overlapped a prior campaign's would
replicate its draws under a new label. The disjointness is proved here against
both E20 protocol families as they are staged, not asserted from the base.

**Two precisions merged into one cell.** The float16 rung is the precision control
and shares its checkpoint and its doses with the bfloat16 112M, so only the model
label separates them.

**A batch wider than one.** ProGen3's expert mixture reduces over the flattened
batch, so a wider extent is a different draw and the protocol must declare 1.

The executor's tests are the contract the stage parsers are written against:
``--device`` and ``--out`` are injected and a manifest that spells either is
refused, and a row can only reach the card its manifest assigned it.
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

from src.capability.generation import fold_recapitulation as fr  # noqa: E402

REFERENCES = REPO_ROOT / "data/pairwise_assets/family_references_jointh_20261013"
E20_CONFIRMATORY = REPO_ROOT / "data/e20_confirmatory_20261013"
E20_PROTOCOLS = sorted(REPO_ROOT.glob("data/e20_*/guided_generation_protocol*.json"))

pytestmark = pytest.mark.skipif(
    not REFERENCES.is_dir() or not (E20_CONFIRMATORY / "guided_generation_protocol.json").is_file(),
    reason="data/ is git-ignored; the staged family references and the E20 protocol are absent",
)


def _module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _module("scripts/capability/decoding/build_progen3_robustness_campaign.py", "pg3rb_build")
stage = _module("scripts/capability/decoding/collect_guided_decoding_pool.py", "pg3rb_stage")
runner = _module("scripts/capability/decoding/run_local_gpu_manifest.py", "pg3rb_runner")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The campaign, built into a scratch tree by the committed builder itself."""

    root = tmp_path_factory.mktemp("campaign")
    data, run = root / "data", root / "run"
    code = builder.main([
        "--data", str(data), "--run", str(run),
        "--family-reference-dir", str(REFERENCES),
        "--roles-protocol", str(E20_CONFIRMATORY / "guided_generation_protocol.json"),
        "--esmfold2", "/nonexistent/ESMFold2-hf",
    ])
    assert code == 0
    protocols = {rung: json.loads((data / builder.protocol_name(rung)).read_text())
                 for rung in builder.RUNGS}
    return {"data": data, "run": run, "protocols": protocols,
            "cells": json.loads((run / "queue" / "cell_plan.json").read_text())}


# ------------------------------------------------------------------- the design


def test_the_cell_plan_is_three_rungs_eight_domains_and_five_arms(built):
    cells = built["cells"]
    assert len(cells) == 120 == len(builder.RUNGS) * 8 * 5
    assert {cell["domain"] for cell in cells} == set(builder.DOMAINS)
    assert {cell["arm_code"] for cell in cells} == set(builder.ARMS)
    assert {(cell["model_arm"], cell["dtype"]) for cell in cells} == set(builder.RUNGS)
    labels = [builder.label(cell) for cell in cells]
    assert len(set(labels)) == len(labels)
    assert all(cell["pool_size"] == 200 for cell in cells)
    # the sealed evaluation set is E20's, not this campaign's to choose
    roles = json.loads((E20_CONFIRMATORY / "guided_generation_protocol.json").read_text())["roles"]
    assert {fr.domain_id(name) for name in roles["evaluation"]} == set(builder.DOMAINS)


def test_every_protocol_declares_the_single_row_batch_the_architecture_requires(built):
    """ProGen3's logits move with the batch's extent, so E20's batch of 50 must not carry over."""

    for protocol in built["protocols"].values():
        assert protocol["sampling"]["batch_size"] == 1
        assert protocol["native_contract"]["batch_size"] == 1
        assert protocol["sampling"]["temperature"] == 1.0
        assert protocol["sampling"]["top_k"] == 0
        assert protocol["sampling"]["top_p"] == 1.0


def test_the_three_protocols_differ_only_in_the_rung_specific_keys(built):
    """Any one of them is a complete declaration for the merge and the measurement."""

    varying = {"model_arms", "native_contract", "strength_grid"}
    bodies = list(built["protocols"].values())
    for key in bodies[0]:
        values = {json.dumps(body[key], sort_keys=True) for body in bodies}
        assert (len(values) == 1) is (key not in varying), key
    for protocol in bodies:
        assert set(protocol["native_contract"]) == {"batch_size", "budget", "dtype", "score"}
        assert protocol["native_contract"]["dtype"] in stage.PROGEN3_AMENDMENT["dtypes"]
        assert protocol["amendment"] == stage.PROGEN3_AMENDMENT["id"]


def test_the_precision_control_shares_its_twins_doses_and_differs_only_in_the_kernel(built):
    """Precision is the only thing that moves between the two 112M rungs."""

    twin, control = ("progen3-112m", "bfloat16"), ("progen3-112m", "float16")
    assert builder.DOSE_SOURCE[control] == builder.DOSE_SOURCE[twin] == "progen3-112m"
    assert builder.doses(stage, control) == builder.doses(stage, twin)
    by_rung = {rung: {} for rung in (twin, control)}
    for cell in built["cells"]:
        rung = (cell["model_arm"], cell["dtype"])
        if rung in by_rung:
            by_rung[rung][(cell["domain"], cell["arm_code"])] = (
                cell["lambda_fields"], cell["lambda_coupling"])
    assert by_rung[twin] == by_rung[control] and by_rung[twin]
    labels = {cell["model_label"] for cell in built["cells"]}
    assert labels == {"progen3-3b-bfloat16", "progen3-112m-bfloat16", "progen3-112m-float16"}


# ---------------------------------------------------------------- the binding


def test_every_cell_binds_through_the_real_stage_validator(built):
    """The frozen protocol is bound once per cell, so no cell is launchable-only-in-principle.

    This is what makes the dose declaration load-bearing rather than decorative:
    the stage refuses a dose off the axis admitted for the cell's own model arm, so
    a cell that binds is a cell scheduled at a calibrated dose.
    """

    for cell in built["cells"]:
        rung = (cell["model_arm"], cell["dtype"])
        protocol = built["protocols"][rung]
        point = {"lambda_fields": cell["lambda_fields"], "lambda_coupling": cell["lambda_coupling"]}
        resolved = stage.validate_protocol(
            protocol, model_arm=cell["model_arm"], domain=cell["domain"],
            guidance_arm=cell["guidance_arm"],
            strength_index=protocol["strength_grid"].index(point),
            stream_index=cell["stream_index"])
        assert resolved["model_label"] == cell["model_label"]
        assert resolved["stream_seed"] == cell["stream_seed"]
        assert resolved["strength"].fields == cell["lambda_fields"]
        assert resolved["strength"].coupling == cell["lambda_coupling"]
        assert resolved["native_contract"]["dtype"] == cell["dtype"]
        assert resolved["native_contract"]["amendment"] == stage.PROGEN3_AMENDMENT["id"]
        assert stage.require_contract_cell(cell["guidance_arm"], resolved["strength"]) == \
            cell["arm_code"]


def test_the_frozen_strength_declaration_covers_every_cell_and_nothing_else(built):
    """``require_frozen_strength`` is the evaluation guard; the plan must satisfy it exactly."""

    protocol = next(iter(built["protocols"].values()))
    rows = [{"model": cell["model_label"], "arm": cell["arm_code"],
             "lambda_fields": cell["lambda_fields"], "lambda_coupling": cell["lambda_coupling"]}
            for cell in built["cells"]]
    fr.require_frozen_strength(rows, protocol["strengths"])
    declaration = fr.parse_strength_declaration(protocol["strengths"])
    assert len(declaration) == len(builder.RUNGS) * len(builder.ARMS)
    for (_, doses, sizes) in declaration.values():
        assert doses == () and set(sizes.values()) == {200}
    with pytest.raises(fr.ContractError):
        fr.require_frozen_strength(
            [{**rows[0], "lambda_coupling": rows[0]["lambda_coupling"] + 1.0}],
            protocol["strengths"])


# -------------------------------------------------------- the seed disjointness


def test_the_seed_streams_are_disjoint_from_every_staged_E20_campaign(built):
    """A reused stream would replicate a prior campaign's draws under a new label.

    Proved against the staged protocols rather than against the declared base, so
    the proof moves if either campaign's seeds ever do.
    """

    protocol = next(iter(built["protocols"].values()))
    depth = protocol["sequences_per_cell"]
    mine = set()
    for seed in protocol["seed_streams"]:
        mine.update(range(seed, seed + depth))
    assert len(protocol["seed_streams"]) == 120
    assert len(mine) == 120 * depth
    prior: set[int] = set()
    seen = 0
    for path in E20_PROTOCOLS:
        if path.parent.name == built["data"].name:
            continue
        body = json.loads(path.read_text())
        for seed in body["seed_streams"]:
            prior.update(range(seed, seed + body["sequences_per_cell"]))
        seen += 1
    assert seen >= 2, "both E20 campaign families must be staged for this proof to mean anything"
    assert not mine & prior
    assert min(mine) - max(prior) > depth


def test_the_streams_of_this_campaign_are_pairwise_disjoint_and_non_adjacent(built):
    """The stage re-proves this from the file; it is the reason a cell owns its draws."""

    protocol = next(iter(built["protocols"].values()))
    streams = sorted(protocol["seed_streams"])
    depth = protocol["sequences_per_cell"]
    assert all(right - left > depth for left, right in zip(streams, streams[1:]))


# ------------------------------------------------------------- the executor


def _manifest(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "campaign_test.tsv"
    path.write_text("# comment\n" + "".join(rows))
    return path


def _row(gpu="0", label="cell", args="--domain 1E0L", slot=1, interpreter=sys.executable):
    return runner_row(slot, gpu, "collect_guided_decoding_pool.py", label, interpreter, args)


def runner_row(slot, gpu, stage_name, label, interpreter, args):
    env = f"TRANSFER_PYTHON={interpreter} PYTHONDONTWRITEBYTECODE=1"
    return "\t".join((str(slot), "pg3rb", str(gpu), stage_name, label, env,
                      "guided_decoding_pool.json", args)) + "\n"


def test_the_executor_injects_the_device_and_out_the_stage_parsers_expect(tmp_path):
    """Injected before the row's own arguments, and the card is the one the manifest named."""

    manifest = _manifest(tmp_path, [_row(gpu="3", label="a"), _row(gpu="cpu", label="b", slot=2)])
    rows = runner.read_manifest(manifest)
    gpu = runner.resolve(rows[0], run=tmp_path / "run", out_prefix="cells")
    assert gpu["command"][2:6] == [
        "--device", "cuda:0", "--out", str(tmp_path / "run/cells/a")]
    assert gpu["command"][6:] == ["--domain", "1E0L"]
    assert gpu["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    cpu = runner.resolve(rows[1], run=tmp_path / "run", out_prefix=".")
    assert cpu["command"][2:6] == ["--device", "cpu", "--out", str(tmp_path / "run/b")]
    assert cpu["env"]["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("args", ["--device cuda:0 --domain 1E0L", "--out /tmp/x --domain 1E0L"])
def test_a_manifest_that_spells_an_injected_argument_is_refused(tmp_path, args):
    with pytest.raises(ValueError, match="this runner injects"):
        runner.read_manifest(_manifest(tmp_path, [_row(args=args)]))


def test_a_malformed_manifest_row_is_refused_rather_than_guessed(tmp_path):
    with pytest.raises(ValueError, match="tab-separated fields"):
        runner.read_manifest(_manifest(tmp_path, ["1\tpg3rb\t0\n"]))
    with pytest.raises(ValueError, match="neither a card index"):
        runner.read_manifest(_manifest(tmp_path, [_row(gpu="gpu0")]))
    with pytest.raises(ValueError, match="no TRANSFER_PYTHON"):
        runner.read_manifest(_manifest(tmp_path, [
            "1\tpg3rb\t0\tcollect_guided_decoding_pool.py\tc\tFOO=1\tx\t--domain 1E0L\n"]))
    with pytest.raises(ValueError, match="same label"):
        runner.read_manifest(_manifest(tmp_path, [_row(label="same"), _row(label="same", gpu="1")]))
    with pytest.raises(ValueError, match="no manifest row"):
        runner.read_manifest(_manifest(tmp_path, []))


def test_two_rows_of_one_slot_may_not_own_one_card(tmp_path):
    manifest = _manifest(tmp_path, [_row(gpu="2", label="a"), _row(gpu="2", label="b")])
    with pytest.raises(SystemExit, match="one card to two rows"):
        runner.main(["--manifest", str(manifest), "--run", str(tmp_path / "run"), "--dry-run"])


def test_a_dry_run_validates_and_writes_nothing(tmp_path):
    run = tmp_path / "run"
    manifest = _manifest(tmp_path, [_row(gpu="0", label="a")])
    assert runner.main(["--manifest", str(manifest), "--run", str(run), "--dry-run"]) == 0
    assert not run.exists()
    assert not manifest.with_suffix(".status.tsv").exists()


def test_a_non_empty_output_directory_is_refused_unless_it_is_already_complete(tmp_path):
    run = tmp_path / "run"
    (run / "a").mkdir(parents=True)
    (run / "a" / "stray.txt").write_text("x")
    manifest = _manifest(tmp_path, [_row(gpu="0", label="a")])
    with pytest.raises(SystemExit, match="cannot be launched"):
        runner.main(["--manifest", str(manifest), "--run", str(run), "--dry-run"])
    (run / "a" / "guided_decoding_pool.json").write_text("{}")
    assert runner.main(["--manifest", str(manifest), "--run", str(run), "--dry-run",
                        "--resume"]) == 0


# ----------------------------------------------- the teacher-forced replay's refusal


def test_the_teacher_forced_strength_script_refuses_an_amended_arms_records():
    """A wrong strength is worse than a missing one, because the dose matching reads it.

    ``measure_effective_guidance_strength.py`` rests on ``replay_strength``, whose
    single teacher-forced pass is not the computation ProGen3's sampler ran. Those
    records are refused by name rather than measured and quietly believed.
    """

    strength = _module("scripts/capability/decoding/measure_effective_guidance_strength.py",
                       "pg3rb_strength")
    amended = stage.amended_model_arms()
    assert amended, "the amendment declares no arm, so this refusal would be vacuous"
    rows = [{"model": f"{amended[0]}-bfloat16"}, {"model": "progen2-medium"}]
    with pytest.raises(SystemExit, match="not prefix-exact"):
        strength.require_prefix_exact_models(rows, amended)
    strength.require_prefix_exact_models(
        [{"model": "progen2-medium"}, {"model": "protgpt2"}], amended)
