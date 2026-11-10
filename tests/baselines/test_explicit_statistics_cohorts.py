"""Conditions the cohort-parameterised explicit-statistics producer must satisfy.

The producer was generalised so that one code path serves the R1 mutation-effect
ladder and any other wild-type cohort. Three classes of claim are held here.

Refusal. A descriptor that does not describe its cohort, a mutation that
disagrees with its reference, a duplicated variant, a reference with no variant
and an output directory that already holds an export each stop the stage rather
than produce a table that looks complete.

Depth. The alignment is rebuilt at the cap the descriptor declares, the cap is
recorded per reference, and a retained profile store is compared against the
rebuild only where it describes an alignment of the same depth -- at the
precision the store was actually written with.

Absence. A reference with too little homologous support is refused with its
reason and counted as refused; its samples keep their rows with no Potts value,
and nothing is imputed, dropped or padded with a zero.

The data-side tests skip when the retained cohorts and the published R1 tables
are absent from a checkout.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.capability.baselines import r1_msa_baselines as R  # noqa: E402
from scripts.capability.baselines import reproduce_explicit_statistics as gate  # noqa: E402
from scripts.capability.baselines import score_r1_gemme as gemme  # noqa: E402
from src.capability.baselines import cohorts as C  # noqa: E402
from src.capability.baselines import msa_baselines as B  # noqa: E402
from src.capability.context.homology import Hit  # noqa: E402

CONFIG = REPO_ROOT / "configs/explicit_statistics_cohorts.json"
LADDER = REPO_ROOT / "results/extensions/statistical_ladder_20261106"
WILDTYPE = "ACDEFGHIKL"


def reference(name: str = "dom-1", sequence: str = WILDTYPE) -> C.Reference:
    return C.Reference(wildtype_id=name, sequence=sequence, query_id=name, aliases=(name,))


def tiny_cohort(tmp_path: Path, *, domains=None, expect=None, cap: int = 5000) -> Path:
    """A one-domain cohort in the Domainome descriptor's own shape."""

    domains = domains if domains is not None else [
        {"name": "dom-1", "wildtype": WILDTYPE,
         "variants": [{"position": 1, "mutant": "V"}, {"position": 3, "mutant": "W"}]}]
    (tmp_path / "cohort.json").write_text(json.dumps({"domains": domains}))
    (tmp_path / "hits.tsv").write_text("")
    # Absolute paths, so that an entry point rooted at the repository resolves
    # the fixture rather than a same-named file beside the real cohorts.
    cohort_path, hits_path = tmp_path / "cohort.json", tmp_path / "hits.tsv"
    config = {
        "schema": C.SCHEMA_CONFIG,
        "protocol": {"max_sequences_rule": "r", "profile_store_role": "r",
                     "search_depth_limitation": "r", "refusal_rule": "r"},
        "cohorts": {"tiny": {
            "title": "a one-domain cohort",
            "reader": "domainome_abundance",
            "out": str(tmp_path / "out"),
            "log_dir": str(tmp_path / "log"),
            "fit_subdir": "potts",
            "max_sequences": cap,
            "search_max_target_seqs": 10000,
            "variant_source_label": "cohort",
            "inputs": {"cohort": str(cohort_path), "hits": str(hits_path)},
            "hits": str(hits_path),
            "expect": expect if expect is not None else {"references": 1, "samples": 2},
        }},
    }
    path = tmp_path / "cohorts.json"
    path.write_text(json.dumps(config))
    return path


def hits_tsv(path: Path, count: int, *, query: str = "dom-1",
             wildtype: str = WILDTYPE) -> None:
    """``count`` full-length hits in the retained DIAMOND field order."""

    rows = []
    for index in range(count):
        subject = list(wildtype)
        subject[index % len(wildtype)] = "G"
        rows.append("\t".join(str(field) for field in (
            query, f"s{index}", 90.0, len(wildtype), len(wildtype) - 1, 1, len(wildtype),
            len(wildtype), len(wildtype), 1e-9, 100.0 - index, wildtype, "".join(subject))))
    path.write_text("\n".join(rows) + "\n")


def hits_of(count: int, *, wildtype: str = WILDTYPE) -> list[Hit]:
    """``count`` full-length hits of descending bit score, each one substitution away."""

    hits = []
    for index in range(count):
        subject = list(wildtype)
        subject[index % len(wildtype)] = "G"
        hits.append(Hit(query="dom-1", subject=f"s{index}", pident=90.0, length=len(wildtype),
                        nident=len(wildtype) - 1, qstart=1, qend=len(wildtype),
                        qlen=len(wildtype), slen=len(wildtype), evalue=1e-9,
                        bitscore=100.0 - index, qseq_gapped=wildtype,
                        sseq_gapped="".join(subject)))
    return hits


# ------------------------------------------------------------------- refusal


def test_rows_refuse_a_mutation_that_disagrees_with_its_reference():
    with pytest.raises(ValueError, match="the wild type carries"):
        C.rows_of([reference()], {"dom-1": [("dom-1", "V1A")]})


def test_rows_refuse_a_reference_with_no_variant():
    with pytest.raises(ValueError, match="no variant"):
        C.rows_of([reference()], {})


def test_validate_rows_refuses_a_duplicate_variant_and_a_broken_order():
    rows = C.rows_of([reference()], {"dom-1": [("dom-1", "A1V"), ("dom-1", "D3W")]})
    C.validate_rows(rows, [reference()])
    duplicated = rows + [dict(rows[0], original_index=2)]
    with pytest.raises(ValueError, match="duplicate"):
        C.validate_rows(duplicated, [reference()])
    shuffled = [dict(rows[1], original_index=0), dict(rows[0], original_index=1)]
    with pytest.raises(ValueError, match="variant_index"):
        C.validate_rows(shuffled, [reference()])


def test_validate_rows_refuses_a_row_of_an_unknown_reference():
    rows = C.rows_of([reference()], {"dom-1": [("dom-1", "A1V")]})
    with pytest.raises(ValueError, match="not a cohort reference"):
        C.validate_rows([dict(rows[0], wildtype_id="dom-2")], [reference()])


def test_a_reference_identifier_must_be_one_path_component():
    for broken in ("a/b", "", ".hidden", "with space"):
        with pytest.raises(ValueError):
            reference(broken)
    reference("v2R31S|R32S_2N5D.pdb")  # a retained cohort name, kept verbatim


def test_load_cohort_refuses_an_unknown_name_and_a_missing_input(tmp_path):
    config = tiny_cohort(tmp_path)
    with pytest.raises(SystemExit, match="unknown cohort"):
        C.load_cohort("absent", config_path=config, root=tmp_path)
    (tmp_path / "hits.tsv").unlink()
    with pytest.raises(FileNotFoundError, match="input hits"):
        C.load_cohort("tiny", config_path=config, root=tmp_path)


def test_load_cohort_refuses_a_declared_shape_it_no_longer_has(tmp_path):
    config = tiny_cohort(tmp_path, expect={"references": 2})
    with pytest.raises(ValueError, match="shape its descriptor declares"):
        C.load_cohort("tiny", config_path=config, root=tmp_path)


def test_load_cohort_refuses_an_unknown_declared_quantity(tmp_path):
    config = tiny_cohort(tmp_path, expect={"nonsense": 1})
    with pytest.raises(ValueError, match="unknown quantities"):
        C.load_cohort("tiny", config_path=config, root=tmp_path)


def test_r1_requires_the_frozen_identifier_function_to_be_injected():
    declared = C.load_config(CONFIG)["cohorts"]["r1"]
    assert declared["sample_id"] == "frozen_cohort_sha"
    with pytest.raises(ValueError, match="frozen cohort-sha"):
        C._read_r1(declared, root=REPO_ROOT, sample_id_of=None)


def test_export_refuses_to_overwrite_an_existing_export(tmp_path):
    config = tiny_cohort(tmp_path)
    (tmp_path / "out" / "inputs").mkdir(parents=True)
    arguments = argparse.Namespace(cohort="tiny", config=config, out=tmp_path / "out",
                                   workers=1, threads=1)
    log = R.Log(tmp_path / "log" / "export.log")
    with pytest.raises(FileExistsError, match="written once"):
        R.stage_export(arguments, log)


@pytest.mark.skipif(R.available_memory() < R.MIN_HEADROOM_BYTES,
                    reason="the export's declared memory headroom is not available here")
@pytest.mark.parametrize("hits,exported,refused", [(1, 0, 1), (4, 1, 0)])
def test_the_export_refuses_a_reference_without_enough_support(tmp_path, hits, exported, refused):
    config = tiny_cohort(tmp_path, expect={"references": 1, "samples": 2, "refusals": refused})
    hits_tsv(tmp_path / "hits.tsv", hits)
    arguments = argparse.Namespace(cohort="tiny", config=config, out=tmp_path / "out",
                                   workers=1, threads=1)
    manifest = R.stage_export(arguments, R.Log(tmp_path / "log" / "export.log"))
    assert manifest["counts"]["backgrounds_exported"] == exported
    assert manifest["counts"]["backgrounds_refused"] == refused
    refusals = json.loads((tmp_path / "out" / "inputs" / "refusals.json").read_text())
    support = json.loads((tmp_path / "out" / "inputs" / "support.json").read_text())["wildtypes"]
    table = json.loads((tmp_path / "out" / "inputs" / "variant_table.json").read_text())
    contract = json.loads((tmp_path / "out" / "contract.json").read_text())
    # Whatever the support, both variants keep their rows and neither is imputed.
    assert len(table["samples"]) == 2
    assert all("profile_delta" not in row for row in table["samples"])
    assert contract["refusals"]["count"] == refused == len(refusals["refused"])
    if refused:
        assert refusals["refused"]["dom-1"]["rows"] == 1
        assert refusals["samples_without_support"] == 2
        assert "2-row floor" in refusals["refused"]["dom-1"]["reason"]
    else:
        # No retained profile describes this alignment, so the cross-check is
        # recorded as skipped rather than silently claimed.
        assert support["dom-1"]["frequencies_verified"] is None
        assert support["dom-1"]["rows"] == 4
        assert contract["depth_cap"]["max_sequences"] == 5000


@pytest.mark.skipif(R.available_memory() < R.MIN_HEADROOM_BYTES,
                    reason="the export's declared memory headroom is not available here")
def test_the_export_refuses_a_refusal_count_the_descriptor_does_not_declare(tmp_path):
    config = tiny_cohort(tmp_path, expect={"references": 1, "samples": 2, "refusals": 0})
    hits_tsv(tmp_path / "hits.tsv", 1)
    arguments = argparse.Namespace(cohort="tiny", config=config, out=tmp_path / "out",
                                   workers=1, threads=1)
    with pytest.raises(RuntimeError, match="references are refused where the descriptor"):
        R.stage_export(arguments, R.Log(tmp_path / "log" / "export.log"))


def test_a_directory_input_is_digested_and_verifies(tmp_path):
    """A cohort can be defined by a directory of measured rows, and both the
    producer and the verifier of a manifest must agree on what that binds to.
    A verifier that only understood files reported a staged directory as missing
    and refused a fit whose inputs were in fact intact."""

    rows = tmp_path / "rows"
    rows.mkdir()
    (rows / "a.jsonl").write_text('{"x": 1}\n')
    (rows / "b.jsonl").write_text('{"x": 2}\n')
    first = B.path_digest(rows)
    assert first == C.path_digest(rows), "one definition, one digest"
    assert first != B.path_digest(rows / "a.jsonl")
    # Every file in the directory is bound: adding, editing or removing one moves it.
    (rows / "c.jsonl").write_text('{"x": 3}\n')
    assert B.path_digest(rows) != first
    (rows / "c.jsonl").unlink()
    assert B.path_digest(rows) == first
    (rows / "b.jsonl").write_text('{"x": 9}\n')
    assert B.path_digest(rows) != first


@pytest.mark.skipif(R.available_memory() < R.MIN_HEADROOM_BYTES,
                    reason="the export's declared memory headroom is not available here")
def test_verify_inputs_accepts_a_directory_the_manifest_names(tmp_path):
    config = tiny_cohort(tmp_path, expect={"references": 1, "samples": 2, "refusals": 0})
    hits_tsv(tmp_path / "hits.tsv", 4)
    extra = tmp_path / "rows"
    extra.mkdir()
    (extra / "one.jsonl").write_text('{"row": 1}\n')
    declared = json.loads(config.read_text())
    declared["cohorts"]["tiny"]["inputs"]["rows"] = str(extra)
    config.write_text(json.dumps(declared))
    arguments = argparse.Namespace(cohort="tiny", config=config, out=tmp_path / "out",
                                   workers=1, threads=1)
    R.stage_export(arguments, R.Log(tmp_path / "log" / "export.log"))
    # The root is the repository's, as in production: this cohort's own inputs are
    # absolute and the plmc binary the manifest names is repository-relative.
    verified = B.verify_inputs(tmp_path / "out", root=REPO_ROOT)
    assert verified["source"]["rows"]["sha256"] == B.path_digest(extra)
    # And a change inside it is drift, not silence.
    (extra / "one.jsonl").write_text('{"row": 2}\n')
    with pytest.raises(RuntimeError, match="no longer match their manifest"):
        B.verify_inputs(tmp_path / "out", root=REPO_ROOT)


# --------------------------------------------------------------------- depth


def test_the_declared_depth_cap_cuts_the_alignment_and_is_recorded():
    rows, weights, _, statistics = B.reconstruct_support(
        WILDTYPE, "dom-1", hits_of(6), max_sequences=2)
    assert statistics["rows"] == len(rows) == len(weights) == 2
    assert statistics["max_sequences"] == 2 and statistics["depth_cap_reached"] is True
    assert statistics["frequencies_verified"] is None
    uncapped = B.reconstruct_support(WILDTYPE, "dom-1", hits_of(6), max_sequences=10)[3]
    assert uncapped["rows"] == 6 and uncapped["depth_cap_reached"] is False


def test_the_cross_check_is_skipped_exactly_where_the_cap_binds(tmp_path):
    config = tiny_cohort(tmp_path, cap=5000)
    cohort = C.load_cohort("tiny", config_path=config, root=tmp_path)
    frequencies = np.zeros((len(WILDTYPE), 20))
    store = C.ProfileStore("profile_features", (), 20000, {
        "shallow": {"frequencies": frequencies, "n_profile_sequences": 4000, "neff": 1.0},
        "at_the_cap": {"frequencies": frequencies, "n_profile_sequences": 5000, "neff": 1.0},
        "deeper": {"frequencies": frequencies, "n_profile_sequences": 9000, "neff": 1.0}})
    capped = C.Cohort(**{**cohort.__dict__, "profile_store": store})
    assert R.comparable_store(capped, "shallow") is not None
    assert R.comparable_store(capped, "at_the_cap") is not None
    assert R.comparable_store(capped, "deeper") is None
    assert R.comparable_store(capped, "absent") is None
    same = C.Cohort(**{**cohort.__dict__,
                       "profile_store": C.ProfileStore("retrieval_bound", (), 5000,
                                                       store._expectations)})
    assert R.comparable_store(same, "deeper") is not None


def test_the_cross_check_holds_the_stores_retained_precision():
    rows, weights, neff, _ = B.reconstruct_support(WILDTYPE, "dom-1", hits_of(4),
                                                   max_sequences=10)
    frequencies = B.reconstructed_frequencies(rows, weights)
    # What a four-decimal store can differ from the rebuild by, and no more.
    rounded = {"frequencies": frequencies, "n_profile_sequences": 4,
               "neff": neff + 2e-5, "neff_tolerance": C.PROFILE_FEATURES_NEFF_PRECISION}
    assert B.compare_frequencies(rows, weights, neff=neff, query_id="dom-1",
                                 expect=rounded)["frequencies_verified"] is True
    with pytest.raises(RuntimeError, match="retained precision"):
        B.compare_frequencies(rows, weights, neff=neff, query_id="dom-1",
                              expect=rounded | {"neff_tolerance": 1e-9})
    with pytest.raises(RuntimeError, match="retained precision"):
        B.compare_frequencies(rows, weights, neff=neff, query_id="dom-1",
                              expect=rounded | {"neff": neff + 1.0})
    with pytest.raises(RuntimeError, match="homolog rows"):
        B.compare_frequencies(rows, weights, neff=neff, query_id="dom-1",
                              expect=rounded | {"n_profile_sequences": 3})


# ------------------------------------------------------------------- absence


def test_declared_refusals_are_counted_and_never_imputed():
    table = [{"sample_id": "a", "assay": "A", "wildtype_id": "q0", "mutation": "A1V",
              "n_substitutions": 1},
             {"sample_id": "b", "assay": "B", "wildtype_id": "q1", "mutation": "A1V",
              "n_substitutions": 1}]
    records = {"q0": {"status": B.STATUS_FITTED,
                      "regularization_policy": B.POLICY_FIXED_LOW_DEPTH,
                      "sample_scores": {"a": -0.5}}}
    document, arrays = B.assemble_scores(
        table, records, wildtype_ids=["q0"], protocol={}, inputs={},
        refused={"q1": {"reason": "1 retained homolog row", "rows": 1, "samples": 1}})
    assert [row["status"] for row in document["samples"]] == [
        B.STATUS_FITTED, B.STATUS_NO_SUPPORT]
    assert document["samples"][1]["potts_delta"] is None
    assert np.isnan(arrays["potts_delta"][1])
    assert document["counts"]["backgrounds_refused"] == 1
    assert document["counts"]["samples_refused"] == 1
    assert document["counts"]["samples_with_potts"] == 1
    # A declared refusal is a stated limit, not an unfinished run.
    assert document["counts"]["complete"] is True
    assert document["refused"]["q1"]["reason"].startswith("1 retained")


def test_a_cohort_without_a_profile_rung_carries_no_invented_one():
    table = [{"sample_id": "a", "assay": "A", "wildtype_id": "q0", "mutation": "A1V",
              "n_substitutions": 1}]
    records = {"q0": {"status": B.STATUS_FITTED,
                      "regularization_policy": B.POLICY_FIXED_LOW_DEPTH,
                      "sample_scores": {"a": 2.0}}}
    document, arrays = B.assemble_scores(table, records, wildtype_ids=["q0"], protocol={},
                                         inputs={})
    assert document["samples"][0]["profile_delta"] is None
    assert np.isnan(arrays["profile_delta"][0])
    assert arrays["potts_delta"][0] == 2.0


def test_a_degenerate_gemme_wildtype_is_refused_rather_than_scored():
    aggregated = {
        "degenerate": True,
        "deficient_note": "every mutation scored 0.0 in every run",
        "variants": [{"sample_id": "a", "gemme_score": 0.0, "gemme_sd": 0.0,
                      "gemme_run1": 0.0, "gemme_run2": 0.0, "gemme_run3": 0.0}],
    }
    refusal = gemme.degenerate_refusal(aggregated, {"rows": 9})
    assert refusal["alignment_rows"] == 9 and refusal["samples"] == 1
    variant = gemme.refused_variant(aggregated["variants"][0])
    assert variant["gemme_score"] is None and variant["gemme_sd"] is None
    assert variant["status"] == gemme.REFUSED_DEGENERATE
    # The raw runs stay as the evidence of the refusal.
    assert variant["gemme_run1"] == 0.0
    with pytest.raises(ValueError, match="not degenerate"):
        gemme.degenerate_refusal(aggregated | {"degenerate": False}, {"rows": 9})


# ----------------------------------------------------- the reproduction gate


@pytest.mark.skipif(not (LADDER / "gemme" / "scores.json").is_file(),
                    reason="the published R1 GEMME table is absent")
def test_the_official_gemme_multi_mutant_score_is_the_sum_of_its_sites():
    outcome = gate.check_gemme_additivity(gate.published(LADDER))
    assert outcome["multi_mutants_with_every_single_scored"] >= 11
    assert outcome["max_absolute_difference_from_the_sum"] < 1e-12
    assert outcome["passed"] is True


@pytest.mark.skipif(not (LADDER / "inputs" / "variant_table.json").is_file(),
                    reason="the frozen R1 export is absent")
def test_the_gate_detects_a_changed_row_rather_than_passing_it():
    frozen = json.loads((LADDER / "inputs" / "variant_table.json").read_text())
    cohort = R.load_cohort(argparse.Namespace(cohort="r1", config=CONFIG))
    assert gate.check_cohort(cohort, frozen)["passed"] is True
    tampered = {**frozen, "samples": [dict(row) for row in frozen["samples"]]}
    tampered["samples"][0]["mutation"] = "A1V"
    outcome = gate.check_cohort(cohort, tampered)
    assert outcome["passed"] is False and outcome["field_drift"]


@pytest.mark.skipif(not (REPO_ROOT / "results/R4/gate_stability_20260924/cohort/cohort.json"
                         ).is_file(), reason="the retained pairwise cohorts are absent")
def test_the_pairwise_cohort_resolves_to_its_declared_shape_and_reuse():
    cohort = C.load_cohort("pairwise", config_path=CONFIG, root=REPO_ROOT)
    counts = cohort.counts()
    assert counts["wildtypes"] == 109 and counts["reused_wildtypes"] == 47
    assert counts["multiple_substitution"] == 8192
    for name, reuse in cohort.reuse.items():
        assert reuse["cohort"] == "r1" and reuse["wildtype_id"].startswith("q")
        assert reuse["support"]["msa"].endswith(f"{reuse['wildtype_id']}.fa")
    # Every double mutant is a cycle corner of the same reference as its singles,
    # which is what a non-additive contrast needs and GEMME cannot supply.
    doubles = [row for row in cohort.rows if row["n_substitutions"] == 2]
    assert doubles and all(len(row["mutation"].split(":")) == 2 for row in doubles)
