"""Conditions the R1 homologous-support baselines must always satisfy.

Three classes of claim are held here. The first is algebra: the Potts delta is
``E(mutant) - E(wild type)`` on the complete sequence, so a multiple mutant
carries every pair coupling and the singles never add up to it. The second is
refusal: a variant table, an export or a manifest that does not describe the
frozen cohort stops the stage instead of producing a table that looks complete.
The third is the support convention: the query row must be weightless, which a
real plmc run holds, and the rebuilt alignment must reproduce the retained
profile store.

The data-side tests (real profile store, real cohort) skip when those artefacts
are absent from a checkout, and the plmc test skips when the pinned binary is.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.capability.interactions import fit_pairwise_sequence_baseline as archived  # noqa: E402
from src.capability.baselines import msa_baselines as B  # noqa: E402
from src.capability.context.homology import Hit  # noqa: E402
from src.capability.context.profiles import AA20, GAP_CODE  # noqa: E402
from src.capability.core.sequence_alignment import alignment_rows  # noqa: E402
from src.capability.stability.pairwise_stability import PlmcParameters, read_plmc  # noqa: E402

PLMC = REPO_ROOT / "data/pairwise_assets/plmc/debbiemarkslab-plmc-18c9e55/bin/plmc"
STORE = REPO_ROOT / "results/R5/retrieval_bound_20260807"
COHORT = REPO_ROOT / "results/R5/context_mutation_rescue_20260923/cohort.json"
WILDTYPE = "ACDEFGHIKLMN"


def codes_of(sequence: str) -> np.ndarray:
    return np.array([AA20.index(residue) for residue in sequence], dtype=np.int64)


def random_model(length: int, seed: int = 0) -> B.PottsScorer:
    rng = np.random.default_rng(seed)
    q = len(AA20)
    pairs = length * (length - 1) // 2
    wildtype = "".join(rng.choice(list(AA20), size=length))
    return B.PottsScorer(wildtype, rng.normal(size=(length, q)),
                         rng.normal(size=(pairs, q, q)).astype(np.float32))


def plmc_parameters(scorer: B.PottsScorer) -> PlmcParameters:
    return PlmcParameters("".join(AA20), scorer.wildtype,
                          np.arange(1, len(scorer.wildtype) + 1),
                          scorer.fields.astype(np.float32),
                          scorer.couplings.astype(np.float32))


# ------------------------------------------------------------------- the delta


@pytest.mark.parametrize("length", [1, 2, 5, 9])
def test_potts_delta_is_the_complete_energy_difference(length):
    scorer = random_model(length, seed=length)
    reference = plmc_parameters(scorer)
    rng = np.random.default_rng(100 + length)
    wild = codes_of(scorer.wildtype)
    wildtype_score = reference.score(scorer.wildtype)
    for draw in range(12):
        codes = wild.copy()
        size = int(rng.integers(1, length + 1))
        for position in rng.choice(length, size=size, replace=False):
            codes[position] = rng.integers(0, len(AA20))
        mutant = "".join(AA20[code] for code in codes)
        assert B.PottsScorer.delta(scorer, codes) == pytest.approx(
            reference.score(mutant) - wildtype_score, abs=1e-4)


def test_potts_delta_carries_couplings_between_the_changed_sites():
    length = 4
    wildtype = "ACDE"
    q = len(AA20)
    pairs = length * (length - 1) // 2
    couplings = np.zeros((pairs, q, q), dtype=np.float64)
    # Only the mutant pair (0, 2) = (W, K) couples; every other block is zero, so
    # the double mutant's delta can come from nowhere else.
    couplings[1, AA20.index("W"), AA20.index("K")] = 1.5
    scorer = B.PottsScorer(wildtype, np.zeros((length, q)), couplings)
    wild = codes_of(wildtype)
    first = wild.copy(); first[0] = AA20.index("W")
    second = wild.copy(); second[2] = AA20.index("K")
    both = wild.copy(); both[0] = AA20.index("W"); both[2] = AA20.index("K")
    single_first, single_second = scorer.delta(first), scorer.delta(second)
    double = scorer.delta(both)
    assert single_first == 0.0 and single_second == 0.0
    assert double == pytest.approx(1.5, abs=1e-12)
    assert double != pytest.approx(single_first + single_second, abs=1e-12)


def test_potts_delta_of_the_wild_type_is_zero_and_refuses_other_states():
    scorer = random_model(5, seed=3)
    assert scorer.delta(codes_of(scorer.wildtype)) == 0.0
    with pytest.raises(ValueError):
        scorer.delta(np.zeros(4, dtype=np.int64))
    with pytest.raises(ValueError):
        scorer.delta(np.full(5, len(AA20), dtype=np.int64))


def test_potts_scorer_refuses_a_model_of_another_protein():
    scorer = random_model(5, seed=4)
    params = plmc_parameters(scorer)
    other = PlmcParameters(params.alphabet, "ACDEF", params.positions, params.fields,
                           params.couplings)
    with pytest.raises(ValueError):
        B.potts_scorer(other, "ACDEG")


def test_sign_check_refuses_an_inverted_or_flat_direction():
    length = 4
    q = len(AA20)
    flat = B.PottsScorer("ACDE", np.zeros((length, q)),
                         np.zeros((length * (length - 1) // 2, q, q)))
    with pytest.raises(ValueError):
        B.shuffle_sign_check(flat, draws=2)


def test_sign_check_holds_the_wild_type_on_a_favourable_model():
    length = 6
    q = len(AA20)
    scorer = random_model(length, seed=11)
    wild = codes_of(scorer.wildtype)
    # Give the wild type a large field advantage at every site.
    fields = np.full((length, q), -10.0)
    fields[np.arange(length), wild] = 10.0
    favoured = B.PottsScorer(scorer.wildtype, fields, scorer.couplings * 0.0)
    assert B.shuffle_sign_check(favoured, draws=3)["fraction_shuffles_below_wildtype"] == 1.0


# ------------------------------------------------------- the selection objective


def test_selection_objective_matches_the_archived_baseline():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, len(AA20) + 1, size=(40, 12)).astype(np.int64)
    rows[rows == len(AA20)] = GAP_CODE
    np.testing.assert_array_equal(B.held_out_mask(rows.astype(np.int8)),
                                  archived.held_out_mask(rows.astype(np.int8), 20260924))
    # The split is a property of the residues, not of the array's storage type:
    # the archived helper hashes whatever dtype it is handed, and alignment_rows
    # hands it int8.
    np.testing.assert_array_equal(B.held_out_mask(rows), B.held_out_mask(rows.astype(np.int8)))
    np.testing.assert_array_equal(B.scored_support(rows), archived.scored_support(rows))
    weights = rng.uniform(0.1, 1.0, size=len(rows))
    np.testing.assert_allclose(B.independent_site(rows, weights),
                               archived.independent_site(rows, weights), rtol=0, atol=0)
    mask = B.scored_support(rows)
    profile = B.independent_site(rows, weights)
    assert B.profile_conditional(profile, rows, mask) == pytest.approx(
        archived.profile_conditional(profile, rows, mask), abs=0)
    q = len(AA20)
    pairs = 12 * 11 // 2
    fields = rng.normal(size=(12, q))
    couplings = rng.normal(size=(pairs, q, q)).astype(np.float32)
    assert B.plmc_conditional(fields, couplings, rows, mask) == pytest.approx(
        archived.plmc_conditional(fields, couplings, rows, mask), abs=1e-12)


def test_regularization_policy_follows_the_declared_floors():
    assert B.regularization_policy(20, 5, scorable=True) == B.POLICY_CROSS_VALIDATED
    assert B.regularization_policy(19, 5, scorable=True) == B.POLICY_FIXED_LOW_DEPTH
    assert B.regularization_policy(20, 4, scorable=True) == B.POLICY_FIXED_LOW_DEPTH
    assert B.regularization_policy(200, 50, scorable=False) == B.POLICY_FIXED_LOW_DEPTH
    assert B.FIXED_LOW_DEPTH_LAMBDA_E == 16.0
    assert B.FIXED_LOW_DEPTH_LAMBDA_E in B.LAMBDA_E_GRID


def test_the_validated_grid_is_the_archived_one():
    assert B.LAMBDA_E_GRID == archived.LAMBDA_E_GRID
    assert B.LAMBDA_H == archived.LAMBDA_H
    assert B.MIN_TRAIN_ROWS == archived.MIN_TRAIN_ROWS
    assert B.MIN_HELD_OUT_ROWS == archived.MIN_HELD_OUT_ROWS
    assert B.INDEPENDENT_SITE_PSEUDOCOUNT == archived.PSEUDOCOUNT
    assert B.SPLIT_SEED == archived.SPLIT_SEED


# ---------------------------------------------------------------- the variants


def synthetic_samples(mutations, assays=None, wildtype="ACDEFGHIKLMN"):
    assays = assays or ["A"] * len(mutations)
    return [{"assay": assay, "cluster": 1, "mutation": mutation, "variant_index": index,
             "original_index": index, "sample_id": f"id{index}"}
            for index, (assay, mutation) in enumerate(zip(assays, mutations))]


def build(samples, wildtype="ACDEFGHIKLMN"):
    return B.build_variant_table(samples, wildtype_of_assay={"A": "q0", "B": "q1"},
                                 sequences={"q0": wildtype, "q1": wildtype})


def test_variant_table_preserves_order_and_counts_substitutions():
    rows = build(synthetic_samples(["A1V", "C2W", "A1V:G6E"]))
    assert [row["sample_id"] for row in rows] == ["id0", "id1", "id2"]
    assert [row["n_substitutions"] for row in rows] == [1, 1, 2]
    assert rows[2]["substitutions"] == [["A", 1, "V"], ["G", 6, "E"]]
    assert B.substitution_counts(rows) == {
        "samples": 3, "single_substitution": 2, "multiple_substitution": 1}


def test_variant_table_refuses_duplicate_and_missing_keys():
    duplicate_id = synthetic_samples(["A1V", "C2W"])
    duplicate_id[1]["sample_id"] = "id0"
    with pytest.raises(ValueError, match="duplicate sample_id"):
        build(duplicate_id)
    duplicate_pair = synthetic_samples(["A1V", "A1V"])
    with pytest.raises(ValueError, match="duplicate variant"):
        build(duplicate_pair)
    missing = synthetic_samples(["A1V"])
    del missing[0]["cluster"]
    with pytest.raises(ValueError, match="missing"):
        build(missing)


def test_variant_table_refuses_a_broken_source_order():
    rows = synthetic_samples(["A1V", "C2W"])
    rows[1]["original_index"] = 5
    with pytest.raises(ValueError, match="original_index"):
        build(rows)
    rows = synthetic_samples(["A1V", "C2W"])
    rows[1]["variant_index"] = 4
    with pytest.raises(ValueError, match="variant_index"):
        build(rows)


def test_variant_table_refuses_a_mutation_that_disagrees_with_the_wild_type():
    for mutation in ("A2V", "A0V", "A1A", "A1V:C2W:A1G", "A1X", "A99V"):
        with pytest.raises(ValueError):
            build(synthetic_samples([mutation]))


def test_variant_table_refuses_a_foreign_sample_id_when_lineage_is_supplied():
    with pytest.raises(ValueError, match="frozen identifier"):
        B.build_variant_table(synthetic_samples(["A1V"]), wildtype_of_assay={"A": "q0"},
                              sequences={"q0": "ACDEFGHIKLMN"},
                              sample_id_of=lambda assay, mutation: "something-else")


def test_variant_table_checks_the_measured_cohort():
    cohort = {"assays": {"A": {"mutants": ["A1V"], "sequences": ["VCDEFGHIKLMN"],
                              "wildtype": "ACDEFGHIKLMN", "wildtype_id": "q0"}}}
    rows = B.build_variant_table(synthetic_samples(["A1V"]), wildtype_of_assay={"A": "q0"},
                                 sequences={"q0": "ACDEFGHIKLMN"}, cohort=cohort)
    assert rows[0]["wildtype_id"] == "q0"
    wrong = json.loads(json.dumps(cohort))
    wrong["assays"]["A"]["sequences"] = ["WCDEFGHIKLMN"]
    with pytest.raises(ValueError, match="does not rebuild"):
        B.build_variant_table(synthetic_samples(["A1V"]), wildtype_of_assay={"A": "q0"},
                              sequences={"q0": "ACDEFGHIKLMN"}, cohort=wrong)


# ----------------------------------------------------------------- the support


def hit(subject: str, query_aligned: str, subject_aligned: str, bitscore: float,
        nident: int, qlen: int = 12) -> Hit:
    return Hit(query="q0", subject=subject, pident=100.0 * nident / len(query_aligned),
               length=len(query_aligned), nident=nident, qstart=1, qend=qlen, qlen=qlen,
               slen=qlen, evalue=1e-9, bitscore=bitscore,
               qseq_gapped=query_aligned, sseq_gapped=subject_aligned)


def support_fixture():
    wildtype = "ACDEFGHIKLMN"
    hits = [hit("s0", wildtype, "VCDEFGHIKLMN", 90.0, 11),
            hit("s1", wildtype, "ACDEYGHIKLMN", 80.0, 11),
            hit("s2", wildtype, "ACDEFGHIKLMK", 70.0, 11),
            hit("s2", wildtype, "ACDEFGHIKLMN", 60.0, 12)]  # lower score, same subject
    rows, weights, neff = alignment_rows(wildtype, "q0", hits, max_sequences=B.MAX_SEQUENCES)
    frequencies = B.reconstructed_frequencies(rows, weights)
    record = {"n_hits": len(hits), "n_profile_sequences": len(rows), "hit_list_saturated": False,
              "neff": neff, "max_identity_over_query": max(h.identity_over_query for h in hits)}
    return wildtype, hits, frequencies, record, rows, weights


def test_reconstructed_support_reproduces_and_verifies_the_retained_profile():
    wildtype, hits, frequencies, record, rows, weights = support_fixture()
    expect = {"frequencies": frequencies, "n_profile_sequences": len(rows),
              "neff": record["neff"]}
    rebuilt, rebuilt_weights, neff, statistics = B.reconstruct_support(
        wildtype, "q0", hits, max_sequences=B.MAX_SEQUENCES, expect=expect)
    assert statistics["frequencies_verified"]
    assert statistics["rows"] == len(rows) == 3
    assert statistics["max_sequences"] == B.MAX_SEQUENCES
    np.testing.assert_array_equal(rebuilt, rows)
    assert neff == pytest.approx(record["neff"])

    with pytest.raises(RuntimeError, match="does not reproduce"):
        B.reconstruct_support(wildtype, "q0", hits, max_sequences=B.MAX_SEQUENCES,
                              expect=expect | {"frequencies": frequencies + 0.01})


def test_exported_alignment_round_trips_with_a_weightless_query_row(tmp_path):
    wildtype, hits, frequencies, record, rows, weights = support_fixture()
    fasta, weight_file = tmp_path / "q0.fa", tmp_path / "q0.weights"
    B.write_query_first_alignment(fasta, weight_file, wildtype, rows, weights)
    lines = weight_file.read_text().split()
    assert float(lines[0]) == B.FOCUS_WEIGHT == 0.0
    rebuilt, rebuilt_weights = B.read_exported_alignment(fasta, weight_file, wildtype)
    np.testing.assert_array_equal(rebuilt, rows)
    np.testing.assert_allclose(rebuilt_weights, weights, rtol=0, atol=5e-7)
    with pytest.raises(ValueError, match="first row"):
        B.read_exported_alignment(fasta, weight_file, "VCDEFGHIKLMN")
    weight_file.write_text("\n".join(["1.000000"] + lines[1:]) + "\n")
    with pytest.raises(ValueError, match="weight"):
        B.read_exported_alignment(fasta, weight_file, wildtype)


# ------------------------------------------------------------- the real plmc run


@pytest.mark.skipif(not PLMC.is_file(), reason="the pinned plmc binary is not staged")
def test_real_plmc_leaves_a_zero_weight_focus_contributionless(tmp_path):
    rng = np.random.default_rng(2)
    length = len(WILDTYPE)
    mutants = []
    for _ in range(30):
        residues = list(WILDTYPE)
        for position in rng.choice(length, size=int(rng.integers(1, 4)), replace=False):
            residues[position] = rng.choice(list(AA20))
        mutants.append("".join(residues))
    weights = np.full(len(mutants), 1.0 / 3.0)
    records = {}
    for name, focus in (("wild", WILDTYPE), ("shifted", mutants[0])):
        fasta, weight_file = tmp_path / f"{name}.fa", tmp_path / f"{name}.weights"
        B.write_query_first_alignment(fasta, weight_file, focus, np.array(
            [B.encode_sequence(mutant) for mutant in mutants]), weights)
        fit = B.run_plmc(PLMC, fasta, weight_file, tmp_path / f"{name}.params",
                         tmp_path / f"{name}.log", lambda_e=16.0, lambda_h=B.LAMBDA_H,
                         max_iter=200, threads=1)
        assert fit["converged"], fit["status"]
        assert fit["effective_number_of_samples"] == pytest.approx(weights.sum(), abs=0.05)
        records[name] = read_plmc(tmp_path / f"{name}.params")
    np.testing.assert_array_equal(records["wild"].fields, records["shifted"].fields)
    np.testing.assert_array_equal(records["wild"].couplings, records["shifted"].couplings)


# ------------------------------------------------------- the convergence gate


def fake_runner(converges_at: int | None):
    """A plmc stand-in that converges only once the cap reaches ``converges_at``."""

    calls = []

    def run(alignment, weights, params, log, *, lambda_e, lambda_h, max_iter, threads):
        calls.append(max_iter)
        converged = converges_at is not None and max_iter >= converges_at
        return {"status": B.CONVERGED_STATUS if converged else "MAXIMUMITERATION",
                "converged": converged, "cap_hit": not converged,
                "iterations": 137, "seconds": 1.0, "lambda_e": lambda_e,
                "lambda_h": lambda_h, "max_iter": max_iter, "threads": threads,
                "effective_number_of_samples": 10.0, "terminal_condition": 0.01,
                "final_objective": 1.0}

    return run, calls


def sweep_entries(converged_everywhere: bool = True) -> list[dict]:
    entries = []
    for index, lambda_e in enumerate(B.LAMBDA_E_GRID):
        entries.append({"lambda_e": lambda_e, "status": B.STATUS_ATTEMPT_CONVERGED,
                        "converged": True, "held_out_conditional": -1.0 - 0.1 * index,
                        "max_iter": 200, "attempts": [{}]})
    if not converged_everywhere:
        entries[-1].update({"status": B.STATUS_ATTEMPT_UNCONVERGED, "converged": False})
    return entries


def test_escalation_retries_the_same_fit_at_each_cap_until_it_converges(tmp_path):
    run, calls = fake_runner(converges_at=1000)
    result = B.escalating_fit(run, alignment=tmp_path / "a.fa", weights=tmp_path / "a.w",
                              params=tmp_path / "a.params", log=tmp_path / "a.log",
                              lambda_e=4.0, threads=1)
    assert calls == [200, 1000]
    assert [entry["max_iter"] for entry in result["attempts"]] == [200, 1000]
    assert result["converged"] and result["status"] == B.STATUS_ATTEMPT_CONVERGED
    assert result["max_iter"] == 1000 and result["seconds"] == pytest.approx(2.0)
    assert result["caps_offered"] == list(B.MAX_ITER_CAPS)


def test_escalation_reports_exhaustion_rather_than_a_converged_fit(tmp_path):
    run, calls = fake_runner(converges_at=None)
    result = B.escalating_fit(run, alignment=tmp_path / "a.fa", weights=tmp_path / "a.w",
                              params=tmp_path / "a.params", log=tmp_path / "a.log",
                              lambda_e=4.0, threads=1)
    assert calls == list(B.MAX_ITER_CAPS)
    assert not result["converged"]
    assert result["status"] == B.STATUS_ATTEMPT_UNCONVERGED
    assert result["max_iter"] == B.MAX_ITER_CAPS[-1]


def test_cross_validation_refuses_an_unconverged_grid_value():
    decision = B.select_cross_validated(sweep_entries())
    assert decision["status"] == B.STATUS_FITTED
    assert decision["selected_lambda_e"] == B.LAMBDA_E_GRID[0]
    refused = B.select_cross_validated(sweep_entries(converged_everywhere=False))
    assert refused["status"] == B.STATUS_CV_NOT_CONVERGED
    assert refused["selected_lambda_e"] is None
    assert refused["unconverged_lambda_e"] == [B.LAMBDA_E_GRID[-1]]
    failed = sweep_entries()
    failed[2] = {"lambda_e": B.LAMBDA_E_GRID[2], "status": B.STATUS_PLMC_FAILED,
                 "converged": False, "attempts": [], "failure": "plmc exited 1"}
    assert B.select_cross_validated(failed)["status"] == B.STATUS_PLMC_FAILED
    assert B.select_cross_validated(sweep_entries()[:3])["status"] == B.STATUS_CV_NOT_CONVERGED


def test_a_refused_cross_validation_is_never_completed_by_a_later_fit():
    table = [{"sample_id": "a", "assay": "A", "wildtype_id": "q0", "mutation": "A1V",
              "n_substitutions": 1, "profile_delta": 0.0}]
    refused = {"q0": {"status": B.STATUS_CV_NOT_CONVERGED,
                       "regularization_policy": B.POLICY_CROSS_VALIDATED,
                       "sample_scores": {"a": 2.5}}}
    document, arrays = B.assemble_scores(table, refused, wildtype_ids=["q0"], protocol={},
                                         inputs={})
    assert np.isnan(arrays["potts_delta"][0])
    assert document["counts"]["backgrounds_cv_not_converged"] == 1
    assert document["counts"]["complete"] is False
    final_ok = {"q0": dict(refused["q0"], status=B.STATUS_FINAL_NOT_CONVERGED)}
    document, arrays = B.assemble_scores(table, final_ok, wildtype_ids=["q0"], protocol={},
                                         inputs={})
    assert arrays["potts_delta"][0] == 2.5
    assert document["counts"]["backgrounds_final_not_converged"] == 1
    assert document["counts"]["complete"] is False


# ------------------------------------------------------- end to end on one thin fit


@pytest.mark.skipif(not PLMC.is_file(), reason="the pinned plmc binary is not staged")
def test_fit_background_records_the_caps_and_scores_a_thin_alignment(tmp_path):
    wildtype = "ACDEFGHIKL"
    rows = np.array([B.encode_sequence(wildtype) for _ in range(6)])
    rows[1, 0] = AA20.index("V")
    rows[2, 1] = AA20.index("W")
    rows[3, 5] = AA20.index("E")
    rows[4, 8] = AA20.index("R")
    weights = np.ones(len(rows))
    fasta, weight_file = tmp_path / "q0.fa", tmp_path / "q0.weights"
    B.write_query_first_alignment(fasta, weight_file, wildtype, rows, weights)
    record = B.fit_background({
        "wildtype_id": "q0", "wildtype": wildtype,
        "support": {"rows": len(rows), "msa": str(fasta), "weights": str(weight_file)},
        "variants": [{"sample_id": "a", "substitutions": [("A", 1, "V")]},
                     {"sample_id": "b", "substitutions": [("G", 6, "E")]}],
        "msa": str(fasta), "weights": str(weight_file),
        "msa_sha256": B.sha256_file(fasta), "weights_sha256": B.sha256_file(weight_file),
        "plmc": str(PLMC), "plmc_record": {"path": str(PLMC), "sha256": "test"},
        "work": str(tmp_path / "work"), "params": str(tmp_path / "params" / "q0.params"),
        "threads": 1})
    assert record["regularization_policy"] == B.POLICY_FIXED_LOW_DEPTH
    assert record["max_iter_caps"] == list(B.MAX_ITER_CAPS)
    assert record["final"]["converged"], record["final"]["status_text"]
    assert record["status"] == B.STATUS_FITTED
    assert record["effective_samples_check"]["within_rounding"]
    assert set(record["sample_scores"]) == {"a", "b"}
    assert record["score_variation"]["variants"] == 2
    assert Path(record["params"]["path"]).is_file()


# ------------------------------------------------------------------- the tables


def test_assembly_keeps_every_row_and_states_each_failure():
    table = [
        {"sample_id": "a", "assay": "A", "wildtype_id": "q0", "mutation": "A1V",
         "n_substitutions": 1, "profile_delta": 0.5},
        {"sample_id": "b", "assay": "A", "wildtype_id": "q0", "mutation": "C2W",
         "n_substitutions": 1, "profile_delta": 0.1},
        {"sample_id": "c", "assay": "B", "wildtype_id": "q1", "mutation": "A1V",
         "n_substitutions": 1, "profile_delta": -0.2},
        {"sample_id": "d", "assay": "C", "wildtype_id": "q2", "mutation": "A1V",
         "n_substitutions": 1, "profile_delta": 0.0},
    ]
    records = {
        "q0": {"status": B.STATUS_FITTED, "regularization_policy": B.POLICY_CROSS_VALIDATED,
               "sample_scores": {"a": 0.0, "b": -1.25}},
        "q1": {"status": B.STATUS_PLMC_FAILED, "regularization_policy": B.POLICY_FIXED_LOW_DEPTH,
               "failure": "plmc exited 1"},
    }
    document, arrays = B.assemble_scores(table, records, wildtype_ids=["q0", "q1", "q2"],
                                         protocol={}, inputs={})
    assert len(document["samples"]) == len(table)
    assert [row["status"] for row in document["samples"]] == [
        B.STATUS_FITTED, B.STATUS_FITTED, B.STATUS_BACKGROUND_FAILED, B.STATUS_NOT_FITTED]
    assert np.isfinite(arrays["potts_delta"][0]) and arrays["potts_delta"][0] == 0.0
    assert np.isnan(arrays["potts_delta"][2]) and np.isnan(arrays["potts_delta"][3])
    assert document["counts"]["complete"] is False
    assert document["counts"]["backgrounds_pending"] == 1
    assert document["counts"]["backgrounds_failed"] == 1


def test_complete_requires_every_background_to_converge():
    table = [{"sample_id": "a", "assay": "A", "wildtype_id": "q0", "mutation": "A1V",
              "n_substitutions": 1, "profile_delta": 0.0}]
    not_converged = {"q0": {"status": B.STATUS_FINAL_NOT_CONVERGED,
                            "regularization_policy": B.POLICY_FIXED_LOW_DEPTH,
                            "sample_scores": {"a": 1.0}}}
    document, arrays = B.assemble_scores(table, not_converged, wildtype_ids=["q0"],
                                         protocol={}, inputs={})
    assert document["counts"]["backgrounds_final_not_converged"] == 1
    assert document["counts"]["complete"] is False
    assert arrays["status"][0] == B.STATUS_FINAL_NOT_CONVERGED
    assert arrays["potts_delta"][0] == 1.0
    converged = {"q0": {"status": B.STATUS_FITTED,
                         "regularization_policy": B.POLICY_FIXED_LOW_DEPTH,
                         "sample_scores": {"a": 1.0}}}
    document, _ = B.assemble_scores(table, converged, wildtype_ids=["q0"], protocol={},
                                    inputs={})
    assert document["counts"]["complete"] is True


# ---------------------------------------------------------------- the manifests


def frozen_export_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A miniature frozen export: one source input, one MSA, one weight file."""

    samples = tmp_path / "samples.json"
    samples.write_text("[]\n")
    out = tmp_path / "ladder"
    (out / "inputs" / "msa").mkdir(parents=True)
    fasta, weights = out / "inputs" / "msa" / "q0.fa", out / "inputs" / "msa" / "q0.weights"
    fasta.write_text(">focus\nACDE\n>h000000\nACDF\n")
    weights.write_text("0.000000\n1.000000\n")
    (out / "inputs" / "manifest.json").write_text(json.dumps({
        "schema": B.SCHEMA_MANIFEST,
        "inputs": {"samples": {"path": str(samples), "sha256": B.sha256_file(samples)}},
        "code": {"entrypoint": "retired-generator-hash"},
        "protocol": {},
    }))
    (out / "inputs" / "variant_table.json").write_text(json.dumps({
        "source_sha256": B.sha256_file(samples), "samples": []}))
    (out / "inputs" / "support.json").write_text(json.dumps({"wildtypes": {"q0": {
        "msa": str(fasta), "msa_sha256": B.sha256_file(fasta),
        "weights": str(weights), "weights_sha256": B.sha256_file(weights)}}}))
    return out, samples


def test_input_verification_refuses_a_tampered_frozen_export(tmp_path):
    out, samples = frozen_export_fixture(tmp_path)
    record = B.verify_inputs(out, root=tmp_path)
    assert record["manifest"]["sha256"] == B.sha256_file(out / "inputs" / "manifest.json")
    assert set(record["derived"]["msa_weights"]) == {"q0"}
    samples.write_text("[1]\n")
    with pytest.raises(RuntimeError, match="source samples"):
        B.verify_inputs(out, root=tmp_path)
    samples.write_text("[]\n")
    fasta = out / "inputs" / "msa" / "q0.fa"
    fasta.write_text(">focus\nACDE\n>h000000\nACDG\n")
    with pytest.raises(RuntimeError, match="msa q0"):
        B.verify_inputs(out, root=tmp_path)


def test_execution_manifest_binds_code_and_protocol(tmp_path):
    out, _ = frozen_export_fixture(tmp_path)
    inputs = B.verify_inputs(out, root=tmp_path)
    fit_out = tmp_path / "potts-v2"
    code_path = tmp_path / "code.py"
    code_path.write_text("# execution code\n")
    code_paths = {"entrypoint": code_path}
    plmc = {"path": "plmc", "sha256": "abc"}
    execution = B.load_or_create_execution(fit_out, inputs=inputs, code_paths=code_paths,
                                           plmc_record=plmc)
    assert execution["schema"] == B.SCHEMA_EXECUTION
    assert execution["code"] == {"entrypoint": B.sha256_file(code_path)}
    assert execution["inputs"] == inputs
    # The retired input-generation code is provenance, not a comparison target.
    assert "input_generation_code_provenance" in execution
    again = B.load_or_create_execution(fit_out, inputs=inputs, code_paths=code_paths,
                                       plmc_record=plmc)
    assert again["created_utc"] == execution["created_utc"]
    code_path.write_text("# different code\n")
    with pytest.raises(RuntimeError, match="code entrypoint"):
        B.load_or_create_execution(fit_out, inputs=inputs, code_paths=code_paths,
                                   plmc_record=plmc)
    code_path.write_text("# execution code\n")
    with pytest.raises(RuntimeError, match="optimization protocol"):
        B.load_or_create_execution(fit_out, inputs=inputs, code_paths=code_paths,
                                   plmc_record={"path": "plmc", "sha256": "def"})


def test_execution_manifest_refuses_inputs_that_moved_under_it(tmp_path):
    out, samples = frozen_export_fixture(tmp_path)
    inputs = B.verify_inputs(out, root=tmp_path)
    fit_out = tmp_path / "potts-v2"
    code_path = tmp_path / "code.py"
    code_path.write_text("# execution code\n")
    execution = B.load_or_create_execution(fit_out, inputs=inputs,
                                           code_paths={"entrypoint": code_path},
                                           plmc_record={"path": "plmc", "sha256": "abc"})
    moved = dict(inputs, derived=dict(inputs["derived"], support={"path": "x", "sha256": "y"}))
    with pytest.raises(RuntimeError, match="frozen inputs"):
        B.verify_execution(execution, inputs=moved, code_paths={"entrypoint": code_path},
                           plmc_record={"path": "plmc", "sha256": "abc"})
    samples.write_text("[1]\n")
    with pytest.raises(RuntimeError, match="source samples"):
        B.verify_inputs(out, root=tmp_path)


# --------------------------------------------------------------- the real data


def test_the_frozen_profile_delta_is_the_retained_cohort_score():
    if not (STORE / "profiles.npz").is_file() or not COHORT.is_file():
        pytest.skip("the retained support or cohort is absent")
    stored = np.load(STORE / "profiles.npz", allow_pickle=False)
    catalogue = json.loads((STORE / "wildtypes.json").read_text())
    background = np.array([catalogue["corpus"]["background"][residue] for residue in AA20])
    cohort = json.loads(COHORT.read_text())
    checked = 0
    for record in cohort["assays"][:5]:
        variants = [B.parse_substitutions(mutation, record["wildtype"])
                    for mutation in record["mutants"]]
        deltas = B.profile_deltas(np.asarray(stored[record["wildtype_id"]], dtype=np.float64),
                                  background, variants)
        np.testing.assert_allclose(deltas, record["profile_scores"], rtol=0, atol=1e-12)
        checked += len(deltas)
    assert checked == 5 * 128


def test_the_retained_support_rebuilds_for_a_real_wild_type():
    if not (STORE / "corpus_hits.tsv").is_file():
        pytest.skip("the retained support is absent")
    from src.capability.context.homology import ALIGNMENT_FIELDS, parse_hits
    catalogue = json.loads((STORE / "wildtypes.json").read_text())
    profiles = json.loads((STORE / "profiles.json").read_text())
    stored = np.load(STORE / "profiles.npz", allow_pickle=False)
    name = "q00001"
    wildtype = {n: s for n, s in B.read_fasta(STORE / "wildtypes.faa")}[name]
    hits = [hit for hit in parse_hits(STORE / "corpus_hits.tsv", fields=ALIGNMENT_FIELDS)
            if hit.query == name]
    assert hits
    record = profiles["profiles"][name]
    _, _, _, statistics = B.reconstruct_support(
        wildtype, name, hits, max_sequences=B.MAX_SEQUENCES,
        expect={"frequencies": stored[name],
                "n_profile_sequences": record["n_profile_sequences"],
                "neff": record["neff"]})
    assert statistics["frequencies_verified"]
    assert statistics["rows"] == profiles["profiles"][name]["n_profile_sequences"]
    assert catalogue["wildtypes"][name]["length"] == len(wildtype)
