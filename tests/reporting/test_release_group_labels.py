"""Keep the frozen release partition while correcting its human-readable meaning."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.capability.reporting import audit_followup_support as audit
from scripts.capability.reporting import followup_quantities as followup
from src.capability.core.evidence_ledger import Artifacts, Ledger
from src.capability.core.io import sha256_file

# Independent frozen roster: changing a label must never regroup a checkpoint.
FROZEN_FAMILIES = {
    "ByGPT5": ("no", ("bygpt5-base-en", "bygpt5-medium-en", "bygpt5-small-en")),
    "DialoGPT": ("no", ("dialogpt-small",)),
    "Galactica": ("no", ("galactica-125m", "galactica-1.3b", "galactica-6.7b", "galactica-30b")),
    "GPT2": ("no", ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")),
    "Llama2": ("no", ("llama-2-7b",)),
    "Llama3": ("no", ("llama-3.2-3b",)),
    "Qwen2.5": ("no", ("qwen2.5-0.5b", "qwen2.5-0.5b-instruct", "qwen2.5-7b", "qwen2.5-32b")),
    "Qwen3": ("no", ("qwen3-8b-base",)),
    "InstructProtein": ("yes", ("instructprotein",)),
    "ProGen2": ("yes", ("progen2-small", "progen2-base", "progen2-medium", "progen2-large", "progen2-xlarge")),
    "ProGen3": ("yes", ("progen3-112m", "progen3-3b")),
    "ProLLaMA": ("yes", ("prollama-stage-1", "prollama")),
    "ProteinGLM": ("yes", ("proteinglm-7b-clm",)),
    "ProtGPT2": ("yes", ("protgpt2",)),
    "ProtGPT3": ("yes", ("protgpt3-1.3b",)),
    "RITA": ("yes", ("rita-xl",)),
}
YES_LABEL = "protein-specialized or protein-adapted release families"
NO_LABEL = "general-purpose text and scientific language–protein release families"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_exact_frozen_roster_and_group_assignments():
    assert audit.FAMILIES == FROZEN_FAMILIES
    assert list(audit.FAMILIES) == list(FROZEN_FAMILIES)
    roster = [arm for _, arms in audit.FAMILIES.values() for arm in arms]
    assert len(roster) == len(set(roster)) == 33
    assert set(roster) == set(audit.ROSTER)
    for flag, checkpoints in (("yes", 14), ("no", 19)):
        assert sum(group == flag for group, _ in audit.FAMILIES.values()) == 8
        assert sum(len(arms) for group, arms in audit.FAMILIES.values() if group == flag) == checkpoints


@pytest.fixture
def synthetic_lineage(tmp_path, monkeypatch):
    """Run the real producer on 99 tiny cells, replacing only its costly bootstrap."""
    admission = {"report_sha256": {}, "summaries": {}}
    expected_arm_values = {}
    for index, arm in enumerate(sorted(audit.ROSTER)):
        expected_arm_values[arm] = np.array([index / 100 + .001, index / 100 + .0011])
        for seed_index, seed in enumerate(audit.SPLIT_SEEDS):
            assays = []
            for group_index in range(2):
                increment = index / 100 + seed_index / 1000 + group_index / 10000
                for offset in (-.00002, .00002):
                    assays.append({
                        "cluster": f"g{group_index}", "C_P_wall_spearman": .25,
                        "C_P_wall+M_spearman": .25 + increment + offset,
                    })
            source = tmp_path / "sources" / f"{arm}-{seed}.json"
            write_json(source, {"arm": arm, "fold_seed": seed, "assays": assays})
            admission["report_sha256"][str(source)] = sha256_file(source)
            admission["summaries"][f"{arm}/{seed}"] = {
                "increment_M_C_P_wall": {
                    "point": index / 100 + seed_index / 1000 + .00005,
                    "interval": [-.1, .5],
                },
            }
    admission_path = tmp_path / (
        "results/R5/local_context_20260923/20260923233257_e429ce7f31e4/"
        "lcgp_admission/local_context_measurement.json")
    write_json(admission_path, admission)
    matrices = []

    def deterministic_bands(values):
        # No bootstrap or random draws: retain the estimator's output shape only.
        values = np.asarray(values)
        matrices.append(values.copy())
        points = values.mean(axis=0)
        return {
            "method": "deterministic test substitute; no resampling",
            "draws": 10000, "groups": len(values), "family_size": values.shape[1],
            "critical_value": 1.0, "point": points.tolist(),
            "interval": np.column_stack([points - .01, points + .01]).tolist(),
        }

    monkeypatch.setattr(audit, "simultaneous_bands", deterministic_bands)
    out = tmp_path / followup.FOLLOWUP
    out.mkdir(parents=True)
    audit.lineage(tmp_path, out)
    return tmp_path, out, admission_path, admission, expected_arm_values, matrices


def test_producer_adds_grouping_metadata_without_changing_legacy_selection(synthetic_lineage):
    _, out, admission_path, admission, arm_values, matrices = synthetic_lineage
    declaration = json.loads((out / "lineage_declaration.json").read_text())
    result = json.loads((out / "lineage_robustness.json").read_text())
    for payload in (declaration, result):
        assert payload["grouping"] == audit.GROUPING
        assert payload["grouping"]["labels"] == {"yes": YES_LABEL, "no": NO_LABEL}
        assert "not protein-exposure status" in payload["grouping"]["interpretation"]
        assert payload["grouping"]["joint_model_assignments"] == {
            "InstructProtein": "yes", "Galactica": "no"}
        assert "with protein adaptation in the first group" in payload["grouping"]["joint_model_note"]
        assert "Galactica is a joint scientific language–protein model in the second group" in (
            payload["grouping"]["joint_model_note"])
        assert payload["grouping"]["legacy_keys"] == [
            "protein_pretraining", "protein_minus_text_release_mean"]
    assert declaration["families"] == {
        family: [flag, list(arms)] for family, (flag, arms) in FROZEN_FAMILIES.items()}
    assert result["schema"] == "local_context_release_family_robustness_v1"
    assert result["family_order"] == list(FROZEN_FAMILIES)
    assert result["admission"] == {"path": str(admission_path), "sha256": sha256_file(admission_path)}
    assert result["source_reports_sha256"] == admission["report_sha256"]
    assert result["declaration_sha256"] == sha256_file(out / "lineage_declaration.json")
    expected_families = {
        family: np.mean([arm_values[arm] for arm in arms], axis=0)
        for family, (_, arms) in FROZEN_FAMILIES.items()}
    assert len(matrices) == 3
    np.testing.assert_allclose(matrices[0], np.column_stack(list(expected_families.values())), atol=1e-12)
    for index, (key, excluded) in enumerate((
            ("protein_minus_text_release_mean", ()),
            ("exclude_shared_llama2_prollama", ("Llama2", "ProLLaMA"))), start=1):
        means = {
            flag: np.mean([values for family, values in expected_families.items()
                           if FROZEN_FAMILIES[family][0] == flag and family not in excluded], axis=0)
            for flag in ("yes", "no")}
        expected = means["yes"] - means["no"]
        np.testing.assert_allclose(matrices[index][:, 0], expected, atol=1e-12)
        assert result[key]["point"][0] == pytest.approx(expected.mean())
    for family, (flag, arms) in FROZEN_FAMILIES.items():
        record = result["family_records"][family]
        assert record["protein_pretraining"] == flag
        assert record["checkpoints"] == list(arms)
        assert record["point"] == pytest.approx(expected_families[family].mean())
    for arm, record in result["checkpoint_records"].items():
        assert record["per_seed_point"] == [
            admission["summaries"][f"{arm}/{seed}"]["increment_M_C_P_wall"]["point"]
            for seed in audit.SPLIT_SEEDS]


def test_lineage_display_keeps_legacy_ids_and_selectors_without_requiring_new_metadata(synthetic_lineage):
    root, out, _, _, _, _ = synthetic_lineage
    # Frozen inputs predate the metadata. The consumer must still read them.
    path = out / "lineage_robustness.json"
    payload = json.loads(path.read_text())
    del payload["grouping"]
    write_json(path, payload)
    block = {"compositions": 3, "clusters": 9, "point": .2, "interval": [.1, .3]}
    write_json(out / "lineage_correlation.json", {
        "checkpoint_count": 18, "release_family_count": 9,
        "cluster_resampling": block, "family_means": block,
        "merge_galactica_instructprotein_sensitivity": block,
    })
    ledger = Ledger(Artifacts(root))
    followup._release_lineage(ledger, followup._emitter(ledger))
    rows = {quantity.id: quantity for quantity in ledger.quantities}
    expected_ids = set()
    for name in ("family_bootstrap", "protein_minus_text_release_mean", "exclude_shared_llama2_prollama"):
        for field in ("draws", "groups", "critical_value", "family_size"):
            identifier = f"lineage_robustness/{name}/{field}"
            expected_ids.add(identifier)
            assert rows[identifier].source_pointer == (name, field)
            assert rows[identifier].value == payload[name][field]
        if name == "family_bootstrap":
            continue
        identifier = f"lineage_robustness/{name}"
        expected_ids.add(identifier)
        row = rows[identifier]
        assert row.source_path == f"{followup.FOLLOWUP}/lineage_robustness.json"
        assert row.source_sha256 == sha256_file(path)
        assert row.source_pointer == (name, "point", "[0]")
        assert row.value == payload[name]["point"][0]
        assert row.interval == tuple(payload[name]["interval"][0])
        assert row.seed_set == followup.SPLIT_SEEDS
        for quantity in (rows[key] for key in expected_ids if key.startswith(identifier)):
            assert YES_LABEL in quantity.claim
            assert NO_LABEL in quantity.claim
            assert "protein-minus-text" not in quantity.claim
            if name == "exclude_shared_llama2_prollama":
                assert "excluding the Llama2 and ProLLaMA release families" in quantity.claim
    for index, family in enumerate(FROZEN_FAMILIES):
        identifier = f"lineage_robustness/family/{family}"
        expected_ids.add(identifier)
        assert rows[identifier].source_pointer == ("family_bootstrap", "point", f"[{index}]")
        assert rows[identifier].value == payload["family_bootstrap"]["point"][index]
    assert {key for key in rows if key.startswith("lineage_robustness/")} == expected_ids
    assert "not protein-exposure status" in ledger.supports["lineage_release_families"].description


def test_other_flag_based_displays_keep_counts_ids_and_selectors(tmp_path):
    write_json(tmp_path / followup.LINEAGE_DECLARATION, {"families": FROZEN_FAMILIES})
    arms = ["instructprotein", "galactica-125m", "llama-2-7b", "prollama"]
    contrast_rows = [
        {"arm": arm, "key": "joint_over_full", "families": 2, "point": .2,
         "interval": [-.1, .3] if arm == "prollama" else [.1, .3]}
        for arm in arms]
    write_json(tmp_path / followup.POSITION / "position_simultaneous.json", {
        "arms": arms, "keys": ["joint_over_full"],
        "controls": {"wall": {"contrasts": contrast_rows, "critical_value": 1,
                              "draws": 10000, "family_size": 4, "family_universe": 2}},
    })
    native_arms = {}
    for arm in arms:
        interval = [.1, .3] if arm == "prollama" else [-.3, -.1]
        native_arms[arm] = {str(seed): {"conditioned": {
            "increment_R_C_P": {
                "rho_control_vs_increment": {"interval": interval},
                "rho_native_vs_increment_given_control": {"interval": [-.1, .1]},
            },
            "increment_R_after_M_C_P": {
                "rho_native_vs_increment_given_control": {"interval": [-.1, .1]},
            },
        }} for seed in followup.SPLIT_SEEDS}
    write_json(tmp_path / followup.NATIVE_EXPRESSION, {"arms": native_arms})
    ledger = Ledger(Artifacts(tmp_path))
    emit = followup._emitter(ledger)
    followup._position_simultaneous(ledger, emit)
    followup._native_expression(ledger, emit)
    rows = {quantity.id: quantity for quantity in ledger.quantities}
    for key, count, label in (
            ("protein_families_in_panel", 2, YES_LABEL),
            ("protein_families_resolving", 1, YES_LABEL),
            ("text_arms_resolving", 2, "general-purpose text and scientific language–protein checkpoints")):
        row = rows[f"position_terms/wall/{key}"]
        assert row.value == count
        assert label in row.claim
        assert row.source_path == followup.LINEAGE_DECLARATION
        assert row.source_pointer == ("families", f"<{key}>")
    for key, count, label in (
            ("all", 3, "all"),
            ("protein", 1, "protein-specialized or protein-adapted"),
            ("text", 2, "general-purpose text and scientific language–protein")):
        row = rows[f"native_expression/control_vs_increment_negative/{key}"]
        assert row.value == count
        assert row.claim.startswith(f"{label} checkpoints")
        assert row.source_path == followup.NATIVE_EXPRESSION
        assert row.source_pointer == ("arms", "<control_vs_increment_negative>")
