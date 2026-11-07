"""Frozen selection, twenty complete candidates, controls and the 66-column family.

Every fixture is synthetic, so these tests need neither the corpus TSV nor a GPU.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.capability.core.amino_acids import AA20
from src.capability.extensions import family_conditionals as fc
from src.capability.extensions import family_distribution as fd

ROOT = Path(__file__).resolve().parents[2]
CONFIG = fc.read_config(ROOT / "configs/family_conditionals.json")
PARENT = fd.read_config(ROOT / "configs/family_distribution.json")
WILDTYPE = "ACDEFGHIKLMNPQRSTVWY" * 2
ARMS = ("arm-a", "arm-b")
PROTOCOL = {
    "declaration": {"id": PARENT["declaration"]["id"]},
    "scoring": {"arms": list(ARMS), "dtype_policy": "synthetic fixture policy",
                "score_function": "synthetic fixture score", "max_len": 1024},
    "queue": {"pilot_models": [ARMS[0]]},
    "output": "parent",
}


def mutant(offset: int) -> str:
    """A distinct AA20 string of the declared wild-type length."""

    return "".join(AA20[(fc.AA_INDEX[residue] + offset) % len(AA20)] for residue in WILDTYPE)


def payload(row_id, native, dropped=(), *, cluster="c000", split="train", representative=True):
    """One row whose deletions at these query positions shorten its native string."""

    kept = [position for position in range(len(native)) if position not in set(dropped)]
    residues = "".join(native[position] for position in kept)
    return {"row_id": row_id, "subject": f"UniRef50_{row_id}", "bitscore": 100.0,
            "qstart": 1, "qend": len(native), "native": residues, "query_index": kept,
            "fragments": [[0, len(residues)]], "cluster": cluster, "split": split,
            "representative": representative}


def unit_payload(unit_id, *, partition="evaluation", heldout=5, dropped=({3, 4}, {3, 4, 5})):
    rows = [payload(f"{unit_id}#t{index}", mutant(index + 1), cluster=f"c{index // 3:03d}")
            for index in range(6)]
    rows.append(payload(f"{unit_id}#n0", mutant(9), cluster="c000", representative=False))
    for index in range(heldout):
        gaps = dropped[index] if index < len(dropped) else ()
        rows.append(payload(f"{unit_id}#h{index}", mutant(index + 2), gaps,
                            cluster=f"c{4 + index // 2:03d}", split="test"))
    wildtype_id = f"q{unit_id[1:]}"
    return {"unit_id": unit_id, "wildtype_ids": [wildtype_id],
            "source_wildtype_ids": [wildtype_id], "family_clusters": [int(unit_id[1:])],
            "partition": partition, "selection_rank": 0,
            "wildtypes": [{"wildtype_id": wildtype_id, "wildtype": WILDTYPE, "rows": rows}]}


def make_config(tmp_path: Path, **overrides) -> dict:
    config = json.loads(json.dumps(CONFIG))
    config["output"] = "run"
    config["parents"]["cohort"] = "parent/queue.json"
    config["statistics"] = dict(config["statistics"], bootstrap_draws=200, min_inference_units=2)
    config["selection"] = dict(config["selection"], max_sites_per_unit=3, max_contexts_per_site=2)
    for key, value in overrides.items():
        config[key] = value
    return config


def freeze_parent(tmp_path: Path, config: dict, units=6, insufficient=False) -> dict:
    entries = [unit_payload(f"u{index:03d}", partition="pilot" if index < 2 else "evaluation")
               for index in range(units)]
    if insufficient:
        entries[0] = unit_payload("u000", partition="pilot", heldout=3)
    queue = {"schema": fd.SCHEMA_QUEUE, "protocol": PROTOCOL, "units": entries}
    fc._atomic_json(tmp_path / config["parents"]["cohort"], queue)
    fd.scientific_inputs(tmp_path, PROTOCOL, queue, create=True)
    return queue


def prepared(tmp_path: Path, **overrides):
    config = make_config(tmp_path, **overrides)
    queue = freeze_parent(tmp_path, config)
    fc.prepare(tmp_path, config)
    return config, queue, fc.load_manifest(tmp_path, config)


def site_row(site, context):
    """A row whose query map places this site's position at its frozen native index."""

    offset = site["query_position"] - context["native_index"]
    native = context["native"]
    return fd.NativeRow(row_id=context["row_id"], subject="UniRef50_TEST", bitscore=100.0,
                        qstart=1, qend=len(native) + offset, native=native,
                        query_index=np.arange(len(native), dtype=np.int64) + offset,
                        fragments=((0, len(native)),))


def conditional(observed_index: int, value: float) -> list[float]:
    """A normalised twenty-way vector whose observed residue scores ``value``."""

    rest = (1.0 - math.exp(value)) / (len(AA20) - 1)
    vector = np.full(len(AA20), math.log(rest))
    vector[observed_index] = value
    return [float(entry) for entry in fc.log_softmax(vector)]


# ------------------------------------------------------------- declarations


def test_the_protocol_inherits_its_roster_instead_of_copying_one():
    raw = json.loads((ROOT / "configs/family_conditionals.json").read_text(encoding="utf-8"))
    assert "scoring" not in raw and "arms" not in raw["parents"]
    assert len(PARENT["scoring"]["arms"]) == 33
    assert len(fc.column_labels(PARENT["scoring"]["arms"])) == 66
    labels = fc.column_labels(PARENT["scoring"]["arms"])
    assert labels[:2] == [f"{PARENT['scoring']['arms'][0]}|minus_profile",
                          f"{PARENT['scoring']['arms'][0]}|minus_markov"]
    assert set(PARENT["queue"]["pilot_models"]) <= set(PARENT["scoring"]["arms"])


def test_the_manifest_freezes_the_roster_read_from_the_cohort(tmp_path):
    config, _, manifest = prepared(tmp_path)
    assert manifest["parent"]["roster"] == list(ARMS)
    assert manifest["parent"]["pilot_models"] == [ARMS[0]]
    assert fc.column_labels(manifest["parent"]["roster"]) == [
        f"{arm}|minus_{baseline}" for arm in ARMS for baseline in fc.BASELINES]
    # The pilot rule comes from the cohort, so the second arm is evaluation only.
    assert fc.validate_arm_partition(ARMS[0], "pilot", manifest) is None
    with pytest.raises(ValueError, match="inherited pilot models"):
        fc.validate_arm_partition(ARMS[1], "pilot", manifest)
    with pytest.raises(ValueError, match="outside the inherited"):
        fc.validate_arm_partition("gpt2-xl", "evaluation", manifest)


# ------------------------------------------------ twenty complete candidates


def test_twenty_candidates_differ_at_one_position_and_restore_the_original():
    native = mutant(3)
    index = 7
    variants = fc.variant_strings(native, index)
    fc.check_variants(native, index, variants)
    assert variants[AA20.index(native[index])] == native
    assert [variant[index] for variant in variants] == list(AA20)
    assert len({variant[:index] + variant[index + 1:] for variant in variants}) == 1
    # A misaligned map, a repeated candidate, a foreign residue and a second
    # change are each refused instead of scored.
    with pytest.raises(ValueError, match="twenty distinct"):
        fc.check_variants(native, index, variants[:-1] + [variants[0]])
    foreign = list(variants)
    foreign[0] = "X" + foreign[0][1:]
    with pytest.raises(ValueError, match="other than the queried one"):
        fc.check_variants(native, index, foreign)
    swapped = list(variants)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="another residue than its AA20 slot"):
        fc.check_variants(native, index, swapped)
    moved = list(variants)
    moved[4] = moved[4][:2] + "K" + moved[4][3:]
    with pytest.raises(ValueError, match="other than the queried one"):
        fc.check_variants(native, index, moved)
    with pytest.raises(ValueError, match="leaves a sequence"):
        fc.variant_strings(native, len(native))


# ------------------------------------------------------- declared baselines


def test_profile_baseline_cancels_the_untouched_positions(tmp_path):
    config, _, manifest = prepared(tmp_path)
    unit = manifest["units"][0]
    frequencies = {int(key): np.asarray(value)
                   for key, value in unit["baselines"]["profile_frequencies"].items()}
    for site in unit["sites"]:
        context = site["contexts"][0]
        row = site_row(site, context)
        raw = [fd.profile_score(replace(row, native=variant), frequencies)
               for variant in fc.variant_strings(context["native"], context["native_index"])]
        np.testing.assert_allclose(fc.log_softmax(raw), site["baseline"]["profile"], atol=1e-12)
        # The twenty values are exactly the position's own distribution: the
        # untouched positions contribute one identical constant each.
        frequency = frequencies[site["query_position"]]
        ordered = [frequency[AA20.index(context["native"][context["native_index"]])]]
        np.testing.assert_allclose(np.exp(site["baseline"]["profile"]).sum(), 1.0, atol=1e-12)
        assert 0.0 < ordered[0] < 1.0


def test_markov_baseline_scores_the_complete_string_on_both_sides(tmp_path):
    config, _, manifest = prepared(tmp_path)
    unit = manifest["units"][0]
    model = {"start": np.asarray(unit["baselines"]["markov"]["start"]),
             "transitions": np.asarray(unit["baselines"]["markov"]["transitions"])}
    for site in unit["sites"]:
        for index, context in enumerate(site["contexts"]):
            native, position = context["native"], context["native_index"]
            row = site_row(site, context)
            raw = [fd.markov_score(replace(row, native=variant), model)
                   for variant in fc.variant_strings(native, position)]
            np.testing.assert_allclose(fc.log_softmax(raw),
                                       site["baseline"]["markov"][index], atol=1e-12)
            previous, following = row.native[position - 1], row.native[position + 1]
            two_sided = [math.log(model["transitions"][fc.AA_INDEX[previous], fc.AA_INDEX[residue]])
                         + math.log(model["transitions"][fc.AA_INDEX[residue], fc.AA_INDEX[following]])
                         for residue in AA20]
            np.testing.assert_allclose(fc.log_softmax(two_sided),
                                       site["baseline"]["markov"][index], atol=1e-12)
            forward_only = [math.log(model["transitions"][fc.AA_INDEX[previous], fc.AA_INDEX[residue]])
                            for residue in AA20]
            assert np.abs(fc.log_softmax(forward_only)
                          - np.asarray(site["baseline"]["markov"][index])).max() > 1e-3
            for vector in (site["baseline"]["profile"],) + tuple(site["baseline"]["markov"]):
                assert abs(sum(math.exp(value) for value in vector) - 1.0) < 1e-12


# ---------------------------------------------------------- frozen selection


def test_selection_is_deterministic_and_records_its_support(tmp_path):
    config = make_config(tmp_path)
    freeze_parent(tmp_path, config)
    first = fc.selection_from_parent(tmp_path, config)
    second = fc.selection_from_parent(tmp_path, config)
    assert first == second
    assert [unit["unit_id"] for unit in first["units"]] == sorted(
        unit["unit_id"] for unit in first["units"])
    for unit in first["units"]:
        assert len(unit["sites"]) <= config["selection"]["max_sites_per_unit"]
        assert unit["eligible_positions"] >= len(unit["sites"])
        positions = [site["query_position"] for site in unit["sites"]]
        assert positions == sorted(positions)
        # Positions 3, 4, 5 are carried by fewer than four held-out rows.
        assert not set(positions) & {3, 4, 5}
        for site in unit["sites"]:
            assert len(site["contexts"]) <= config["selection"]["max_contexts_per_site"]
            assert len(site["contexts"]) <= site["contexts_available"]
            assert site["training_rows_with_site"] >= 4
            assert site["heldout_rows_with_site"] >= 4
            for context in site["contexts"]:
                assert context["observed"] == context["native"][context["native_index"]]
    # A different seed draws a different selection, and the counts are recorded.
    other = dict(config, selection=dict(config["selection"], seed=config["selection"]["seed"] + 1))
    assert fc._counts(fc.selection_from_parent(tmp_path, other)["units"]) == \
        fc._counts(first["units"])
    assert fc.selection_from_parent(tmp_path, other) != first


def test_units_without_support_are_recorded_rather_than_dropped(tmp_path):
    config = make_config(tmp_path)
    freeze_parent(tmp_path, config, insufficient=True)
    manifest = fc.prepare(tmp_path, config)
    counts = manifest["counts"]["partitions"]
    assert counts["pilot"]["units"] == 2 and counts["pilot"]["units_with_sites"] == 1
    assert counts["pilot"]["units_insufficient_support"] == 1
    unit = manifest["units"][0]
    assert unit["sites"] == [] and unit["eligible_positions"] == 0
    assert manifest["counts"]["partitions"]["evaluation"]["units_with_sites"] == 4


# ------------------------------------------------------------- site weights


def test_a_site_weighs_the_same_whatever_its_context_count(tmp_path):
    config, _, manifest = prepared(tmp_path)
    unit = manifest["units"][0]
    observed = [[fc.AA_INDEX[context["observed"]] for context in site["contexts"]]
                for site in unit["sites"]]
    stored = {"sites": {}}
    for site, indices in zip(unit["sites"], observed):
        value = -1.0 if site is unit["sites"][0] else -3.0
        stored["sites"][str(site["query_position"])] = {
            "row_ids": [context["row_id"] for context in site["contexts"]],
            "logprob": [conditional(index, value) for index in indices],
            "scored_tokens": [[len(context["native"])] * len(AA20) for context in site["contexts"]]}
    expected = np.mean([-1.0, -3.0, -3.0][:len(unit["sites"])])
    assert fc._unit_logscores(unit, stored)["model"] == pytest.approx(expected)
    # Halving one site's contexts leaves that site's weight and the family value alone.
    trimmed = unit["sites"][0]
    assert len(trimmed["contexts"]) > 1
    shortened = json.loads(json.dumps(stored))
    record = shortened["sites"][str(trimmed["query_position"])]
    for key in ("row_ids", "logprob", "scored_tokens"):
        record[key] = record[key][:1]
    trimmed["contexts"] = trimmed["contexts"][:1]
    assert fc._unit_logscores(unit, shortened)["model"] == pytest.approx(expected)
    # Pooling every context instead would weight the wider site twice as much.
    assert _pooled(unit, shortened) != pytest.approx(expected)


def _pooled(unit, stored) -> float:
    """The rejected alternative: every context weighted by itself."""

    values = []
    for site in unit["sites"]:
        record = stored["sites"][str(site["query_position"])]
        observed = [fc.AA_INDEX[context["observed"]] for context in site["contexts"]]
        logprob = np.asarray(record["logprob"])
        values.extend(logprob[np.arange(len(observed)), observed])
    return float(np.mean(values))


# ---------------------------------------------------------------- identity


@pytest.mark.parametrize("field", ["native", "split", "seed", "missing_manifest"])
def test_same_identifiers_do_not_authorise_changed_inputs(tmp_path, field):
    config, queue, manifest = prepared(tmp_path)
    if field == "missing_manifest":
        (tmp_path / config["output"] / fc.INPUT_MANIFEST).unlink()
        path = tmp_path / config["output"] / "scores" / ARMS[0] / "evaluation.json"
        fc._atomic_json(path, {"schema": fc.SCHEMA_SCORES})
        with pytest.raises(ValueError, match="not adopted retroactively"):
            fc.load_manifest(tmp_path, config)
        return
    if field == "native":
        queue["units"][0]["wildtypes"][0]["rows"][0]["native"] = mutant(11)
    elif field == "split":
        queue["units"][0]["wildtypes"][0]["rows"][0]["split"] = "test"
    else:
        config["selection"]["seed"] += 1
    fc._atomic_json(tmp_path / config["parents"]["cohort"], queue)
    expected = "another declaration" if field == "seed" else "scientific input snapshot differs"
    with pytest.raises(ValueError, match=expected):
        fc.load_manifest(tmp_path, config)
    with pytest.raises(ValueError):
        fc.prepare(tmp_path, config)


def test_repeated_preparation_reproduces_the_manifest(tmp_path):
    config = make_config(tmp_path)
    freeze_parent(tmp_path, config)
    first = fc.prepare(tmp_path, config)
    path = tmp_path / config["output"] / fc.INPUT_MANIFEST
    before = path.read_bytes()
    assert fc.prepare(tmp_path, config) == first
    assert path.read_bytes() == before
    assert all("baseline" not in site for unit in first["units"] for site in unit["sites"])
    loaded = fc.load_manifest(tmp_path, config)
    assert all("baseline" in site for unit in loaded["units"] for site in unit["sites"])
    assert path.read_bytes() == before  # In-memory derivation never rewrites the manifest.


@pytest.mark.parametrize("field", ["baseline", "altered_baseline", "counts"])
def test_redundant_baselines_and_changed_counts_are_refused(tmp_path, field):
    config, _, loaded = prepared(tmp_path)
    path = tmp_path / config["output"] / fc.INPUT_MANIFEST
    stored = fc.load_json(path)
    if field == "counts":
        stored["counts"]["partitions"]["pilot"]["contexts"] += 1
    else:
        vector = loaded["units"][0]["sites"][0]["baseline"]
        if field == "altered_baseline":
            vector["markov"][0][0] -= 0.5
        stored["units"][0]["sites"][0]["baseline"] = vector
    fc._atomic_json(path, stored)
    # Even a correct redundant vector is rejected, not used as a second authority.
    for reader in (fc.load_manifest, fc.prepare, fc.analyse):
        with pytest.raises(ValueError, match="different selection|frozen manifest differs"):
            reader(tmp_path, config)


def test_missing_parent_snapshot_cannot_be_recreated(tmp_path):
    config, _, _ = prepared(tmp_path)
    path = tmp_path / PROTOCOL["output"] / fd.INPUT_SNAPSHOT
    path.unlink()
    for reader in (fc.load_manifest, fc.prepare):
        with pytest.raises(ValueError, match="scientific input snapshot is missing"):
            reader(tmp_path, config)
    assert not path.exists()


# ----------------------------------------------------------------- scoring


def scored(tmp_path, monkeypatch):
    config, queue, manifest = prepared(tmp_path)
    from types import SimpleNamespace
    from torch.nn import Linear

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    raw = {"model_type": "synthetic_fixture", "n_layer": 1, "vocab_size": 20}
    fc._atomic_json(checkpoint / "config.json", raw)
    (checkpoint / "model.pt").write_bytes(b"synthetic fixture, not model weights")
    model: Any = Linear(1, 1)
    model.config = SimpleNamespace(to_dict=lambda: dict(raw))
    handle = SimpleNamespace(name=ARMS[0], spec=SimpleNamespace(path=checkpoint), model=model)
    loads, calls = [], []

    def load(name, *, device):
        loads.append(device)
        return handle

    def score(handle, sequences, *, batch_size, max_len):
        calls.append(list(sequences))
        return ([float(sum(fc.AA_INDEX[residue] for residue in value) / 10.0)
                 for value in sequences], [len(value) for value in sequences])

    monkeypatch.setattr(fc, "load_causal", load)
    monkeypatch.setattr(fc, "_nll_and_tokens", score)
    return config, queue, manifest, handle, loads, calls


def test_scoring_records_normalised_conditionals_and_resumes(tmp_path, monkeypatch):
    config, queue, manifest, handle, loads, calls = scored(tmp_path, monkeypatch)
    document = fc.score_partition(tmp_path, config, arm=ARMS[0], partition="evaluation",
                                  device="cpu")
    assert loads == [None, "cpu"]
    assert document["complete"] is True
    units = [unit for unit in manifest["units"] if unit["partition"] == "evaluation"]
    assert set(document["units"]) == {unit["unit_id"] for unit in units}
    contexts = [context for unit in units for site in unit["sites"] for context in site["contexts"]]
    assert len(calls) == len(contexts)
    for call, context in zip(calls, contexts):
        assert call == fc.variant_strings(context["native"], context["native_index"])
        fc.check_variants(context["native"], context["native_index"], call)
    site = next(site for unit in units for site in unit["sites"])
    record = document["units"][units[0]["unit_id"]]["sites"][str(site["query_position"])]
    logprob = np.asarray(record["logprob"])
    assert logprob.shape == (len(site["contexts"]), len(AA20))
    np.testing.assert_allclose(np.exp(logprob).sum(axis=1), 1.0, atol=1e-12)
    observed = [fc.AA_INDEX[context["observed"]] for context in site["contexts"]]
    assert logprob[np.arange(len(observed)), observed].max() <= 0.0
    # The token diagnostic is stored beside the conditional and never divides it.
    nll = np.asarray([fc.AA_INDEX[residue] for residue in AA20]) / 10.0
    assert record["scored_tokens"][0] == [len(site["contexts"][0]["native"])] * len(AA20)
    np.testing.assert_allclose(logprob[0], fc.log_softmax(-nll), atol=1e-12)
    # Resume does not load weights again and reproduces the stored document.
    assert fc.score_partition(tmp_path, config, arm=ARMS[0], partition="evaluation",
                              device="cpu") == document
    assert loads == [None, "cpu", None]
    path = tmp_path / config["output"] / "scores" / ARMS[0] / "evaluation.json"
    document["config"]["score_definition"] = "another definition"
    fc._atomic_json(path, document)
    with pytest.raises(ValueError, match="another declaration"):
        fc.score_partition(tmp_path, config, arm=ARMS[0], partition="evaluation", device="cpu")


@pytest.mark.parametrize("field", ["native", "native_index", "observed"])
def test_same_id_target_changes_cannot_resume(tmp_path, monkeypatch, field):
    config, queue, manifest, *_ = scored(tmp_path, monkeypatch)
    fc.score_partition(tmp_path, config, arm=ARMS[0], partition="evaluation", device="cpu")
    path = tmp_path / config["output"] / fc.INPUT_MANIFEST
    stored = json.loads(path.read_text(encoding="utf-8"))
    context = stored["units"][0]["sites"][0]["contexts"][0]
    if field == "native":
        context["native"] = ("A" if context["native"][0] != "A" else "C") + context["native"][1:]
    elif field == "native_index":
        context["native_index"] += 1
    else:
        context["observed"] = "A" if context["observed"] != "A" else "C"
    fc._atomic_json(path, stored)
    with pytest.raises(ValueError, match="frozen manifest differs"):
        fc.score_partition(tmp_path, config, arm=ARMS[0], partition="evaluation", device="cpu")


def test_nonfinite_and_unnormalised_conditionals_are_refused():
    with pytest.raises(RuntimeError, match="sum to"):
        fc._check_conditional(np.full(len(AA20), -0.1), 0)
    favoured = np.full(len(AA20), math.log(0.05 / (len(AA20) - 1)))
    favoured[3] = math.log(0.95)
    fc._check_conditional(favoured, 3)  # A high-probability observed residue is valid.
    with pytest.raises(RuntimeError, match="positive conditional"):
        fc._check_conditional(np.zeros(len(AA20)) + np.eye(len(AA20))[3] * 0.1, 3)
    with pytest.raises(RuntimeError, match="not finite"):
        fc._check_conditional(np.full(len(AA20), np.nan), 0)
    with pytest.raises(ValueError, match="twenty finite"):
        fc.log_softmax([0.0] * 19)


# ---------------------------------------------------------------- analysis


def fake_document(manifest, config, arm, partition, value_of):
    document: dict[str, Any] = {
        "schema": fc.SCHEMA_SCORES, "scientific_inputs": fc.INPUT_MANIFEST,
        "config": fc.score_config(config, manifest, arm=arm, partition=partition),
        "checkpoint": {"arm": arm, "config": {"model_type": "synthetic_fixture"},
                       "tensor_files": [{"name": "fixture", "bytes": 0}]},
        "dtype": fc._declared_dtype(arm), "units": {}, "complete": True}
    for unit in manifest["units"]:
        if unit["partition"] != partition or not unit["sites"]:
            continue
        sites = {}
        for site in unit["sites"]:
            observed = [fc.AA_INDEX[context["observed"]] for context in site["contexts"]]
            sites[str(site["query_position"])] = {
                "row_ids": [context["row_id"] for context in site["contexts"]],
                "logprob": [conditional(index, value_of(arm, unit, site))
                            for index in observed],
                "scored_tokens": [[len(context["native"])] * len(AA20)
                                  for context in site["contexts"]]}
        document["units"][unit["unit_id"]] = {"wildtype_id": unit["wildtype_id"],
                                             "sites": sites}
    return document


def write_scores(tmp_path, config, manifest, arms, value_of):
    for arm in arms:
        for partition in fc.PARTITIONS:
            document = fake_document(manifest, config, arm, partition, value_of)
            if not document["units"]:
                continue
            fc._atomic_json(tmp_path / config["output"] / "scores" / arm / f"{partition}.json",
                            document)


def test_analyse_reports_the_labelled_family_and_the_declared_controls(tmp_path):
    config, queue, manifest = prepared(tmp_path)
    write_scores(tmp_path, config, manifest, ARMS, lambda arm, unit, site: -2.0 if arm == ARMS[0] else -4.0)
    summary = fc.analyse(tmp_path, config)
    evaluation = summary["partitions"]["evaluation"]
    assert evaluation["status"] == "complete"
    assert evaluation["columns"] == [f"{arm}|minus_{baseline}" for arm in ARMS for baseline in fc.BASELINES]
    assert evaluation["hypothesis_family"] == len(ARMS) * 2
    assert evaluation["units"] == 4 and evaluation["samples"]["families"] == 4
    assert evaluation["samples"]["sites"] == 4 * config["selection"]["max_sites_per_unit"]
    assert evaluation["samples"]["contexts"] == 4 * config["selection"]["max_sites_per_unit"] * 2
    assert evaluation["absolute_logscore"]["model"][ARMS[0]] == pytest.approx(-2.0)
    assert evaluation["absolute_logscore"]["uniform_aa"] == pytest.approx(-math.log(20))
    assert summary["uniform_aa_logscore"] == pytest.approx(-math.log(20))
    assert evaluation["inference"]["primary"]["family_size"] == 4
    assert summary["partitions"]["pilot"]["status"] == "descriptive_only"
    points = {entry["arm"]: entry["point"]
              for entry in evaluation["inference"]["primary"]["hypotheses"]}
    profile = evaluation["absolute_logscore"]["profile"]
    assert points[f"{ARMS[0]}|minus_profile"] == pytest.approx(-2.0 - profile)
    joined = " ".join(summary["notes"])
    assert "R1/R2/R3" in joined and "pilot partition is descriptive only" in joined


def test_a_missing_arm_leaves_the_partition_descriptive(tmp_path):
    config, queue, manifest = prepared(tmp_path)
    write_scores(tmp_path, config, manifest, [ARMS[0]], lambda arm, unit, site: -2.0)
    summary = fc.analyse(tmp_path, config)
    evaluation = summary["partitions"]["evaluation"]
    assert evaluation["status"] == "descriptive_only"
    assert evaluation["roster"]["missing"] == [ARMS[1]]
    assert "different, undeclared one" in evaluation["reason"]
    assert "inference" not in evaluation


def test_an_unsupported_unit_blocks_formal_completion(tmp_path):
    config = make_config(tmp_path)
    freeze_parent(tmp_path, config, insufficient=True)
    manifest = fc.prepare(tmp_path, config)
    write_scores(tmp_path, config, manifest, ARMS, lambda arm, unit, site: -2.5)
    summary = fc.analyse(tmp_path, config)
    evaluation = summary["partitions"]["evaluation"]
    assert evaluation["status"] == "complete"
    pilot = summary["partitions"]["pilot"]
    assert pilot["status"] == "descriptive_only"
    assert pilot["unsupported_units"] == ["u000"]
    assert "no eligible query position" in pilot["reason"]


def test_analysis_refuses_unscored_units_and_incomplete_documents(tmp_path):
    config, queue, manifest = prepared(tmp_path)
    write_scores(tmp_path, config, manifest, ARMS, lambda arm, unit, site: -2.0)
    path = tmp_path / config["output"] / "scores" / ARMS[0] / "evaluation.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    dropped = sorted(document["units"])[0]
    del document["units"][dropped]
    fc._atomic_json(path, document)
    with pytest.raises(ValueError, match="is unscored"):
        fc.analyse(tmp_path, config)
    document["complete"] = False
    fc._atomic_json(path, document)
    with pytest.raises(ValueError, match="incomplete"):
        fc.analyse(tmp_path, config)

def test_a_nonfinite_model_score_stops_the_run_without_a_checkpoint(tmp_path, monkeypatch):
    config, queue, manifest, *_ = scored(tmp_path, monkeypatch)
    monkeypatch.setattr(fc, "_nll_and_tokens",
                        lambda handle, sequences, *, batch_size, max_len:
                        ([float("inf")] * len(sequences), [len(value) for value in sequences]))
    with pytest.raises(ValueError, match="twenty finite"):
        fc.score_partition(tmp_path, config, arm=ARMS[0], partition="pilot", device="cpu")
    path = tmp_path / config["output"] / "scores" / ARMS[0] / "pilot.json"
    assert not path.exists()
