"""The MODEL - LOOKUP channel, its controls, and the hazards it is built around.

Each test here corresponds to a way this stage could report a false result rather
than to a line of its implementation:

* an aligner whose masking truncates the alignment of a verbatim corpus member
  under-states retrieval and inflates MODEL - LOOKUP -- the EXP-R2-061 failure,
  which had no executable anchor until this stage;
* DIAMOND's ``qseq``/``sseq`` are the aligned strings *with gaps removed*, so
  walking them together shifts every column after the first indel and builds a
  profile of the wrong residues at the right positions;
* a difficulty control that is fitted on the units it adjusts removes real signal
  and manufactures a null, which is the partial-correlation shape this repository
  has retracted twice;
* an interval taken over assays rather than over wild-type families is an
  interval over a population whose members are copies of each other;
* a bin edge chosen after the fact turns a non-monotone gradient into a finding.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.context import profiles as P  # noqa: E402
from src.capability.core.arms import PANEL  # noqa: E402
from src.capability.core.sequence_metadata import PROTEINGYM_ROOT  # noqa: E402
from src.capability.models.fitness import (  # noqa: E402
    BLOSUM62,
    available_assays,
    load_assay,
    parse_mutant,
    wildtype_of,
)
from src.capability.context.homology import (  # noqa: E402
    ALIGNMENT_FIELDS,
    DIAMOND_FIELDS,
    Hit,
    parse_hits,
    truncated_alignment,
)


def _stage():
    path = REPO_ROOT / "scripts" / "capability" / "stages/retrieval_bound.py"
    spec = importlib.util.spec_from_file_location("_stage_20", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hit(query="q0", subject="s0", *, qseq, sseq, qstart=1, qlen=None, bitscore=100.0):
    ungapped = qseq.replace("-", "")
    qlen = len(ungapped) if qlen is None else qlen
    nident = sum(1 for a, b in zip(qseq, sseq) if a == b and a != "-")
    return Hit(
        query=query,
        subject=subject,
        pident=100.0 * nident / max(1, len(qseq)),
        length=len(qseq),
        nident=nident,
        qstart=qstart,
        qend=qstart + len(ungapped) - 1,
        qlen=qlen,
        slen=len(sseq.replace("-", "")),
        evalue=1e-50,
        bitscore=bitscore,
        qseq_gapped=qseq,
        sseq_gapped=sseq,
    )


# ------------------------------------------------------- the mutation grammar


#: ProteinGym is a downloaded benchmark, staged per host and not distributed with the code.
needs_proteingym = pytest.mark.skipif(
    not PROTEINGYM_ROOT.is_dir(),
    reason=f'{PROTEINGYM_ROOT} is absent: the ProteinGym benchmark is staged per host and '
           'not distributed with the code repository')


@needs_proteingym
def test_parse_mutant_is_the_one_grammar_and_agrees_with_the_blosum_channel():
    assay_name = available_assays()[0]
    assay = load_assay(assay_name, n=32, seed=7)
    by_hand = np.array(
        [sum(BLOSUM62[(t[0], t[-1])] for t in m.split(":")) for m in assay.mutants],
        dtype=np.float64,
    )
    through_parser = np.array(
        [sum(BLOSUM62[(w, m)] for w, _, m in subs) for subs in assay.substitutions],
        dtype=np.float64,
    )
    assert np.array_equal(by_hand, through_parser)


@pytest.mark.parametrize("bad", ["", "A", "AG", "AxG", "A12G:", "A12G:XG"])
def test_parse_mutant_refuses_a_string_that_is_not_a_substitution(bad):
    with pytest.raises(ValueError):
        parse_mutant(bad)


@needs_proteingym
def test_wildtype_of_agrees_with_the_full_loader_it_shortcuts():
    name = available_assays()[0]
    assert wildtype_of(name) == load_assay(name, n=4, seed=1).wildtype


def test_available_assays_reads_the_directory_and_refuses_an_empty_one(tmp_path):
    with pytest.raises((FileNotFoundError, RuntimeError)):
        available_assays(tmp_path / "absent")
    (tmp_path / "empty").mkdir()
    with pytest.raises(RuntimeError):
        available_assays(tmp_path / "empty")


# ------------------------------------------------------- the DIAMOND contract


def test_the_default_field_list_parses_exactly_as_it_did_before(tmp_path):
    row = "q0\ts0\t100.0\t10\t10\t1\t10\t10\t10\t1e-9\t42.0\n"
    path = tmp_path / "hits.tsv"
    path.write_text(row)
    hit = parse_hits(path)[0]
    assert (hit.query, hit.subject, hit.nident, hit.qlen, hit.bitscore) == (
        "q0",
        "s0",
        10,
        10,
        42.0,
    )
    assert hit.qseq_gapped is None and hit.sseq_gapped is None


def test_the_alignment_field_list_carries_the_gapped_strings(tmp_path):
    row = "q0\ts0\t90.0\t10\t9\t1\t9\t9\t10\t1e-9\t42.0\tAC-DEFGHIK\tACWDEFGHIK\n"
    path = tmp_path / "hits.tsv"
    path.write_text(row)
    hit = parse_hits(path, fields=ALIGNMENT_FIELDS)[0]
    assert hit.qseq_gapped == "AC-DEFGHIK"
    assert len(hit.qseq_gapped) == len(hit.sseq_gapped)


def test_a_field_list_missing_a_required_column_is_refused(tmp_path):
    path = tmp_path / "hits.tsv"
    path.write_text("q0\ts0\n")
    with pytest.raises(ValueError, match="required by every consumer"):
        parse_hits(path, fields=("qseqid", "sseqid"))


def test_a_field_with_no_home_on_a_hit_is_refused(tmp_path):
    path = tmp_path / "hits.tsv"
    path.write_text("")
    with pytest.raises(ValueError, match="no place on a Hit"):
        parse_hits(path, fields=(*DIAMOND_FIELDS, "stitle"))


def _cohort(records):
    from src.capability.core.arms import Cohort

    return Cohort(
        name="wt",
        kind="protein",
        records=list(records),
        min_symbols=min(len(r) for r in records),
        max_symbols=max(len(r) for r in records),
        metadata={"repeats": [(0, 0, len(r)) for r in records]},
    )


def _masked_hit(query, *, qlen, slen, nident, qstart, qend, pident=100.0):
    return Hit(
        query=query,
        subject="UniRef50_X",
        pident=pident,
        length=qend - qstart + 1,
        nident=nident,
        qstart=qstart,
        qend=qend,
        qlen=qlen,
        slen=slen,
        evalue=0.0,
        bitscore=1000.0,
    )


def test_the_exp_r2_061_truncation_still_stops_the_run_under_both_rules():
    """732 residues, 607 aligned, a length-matched subject: the measured failure.

    Its observed identity is 82.9 in ``id70_to_95`` and its potential is 100.0 in
    ``ge95_near_duplicate``, so repairing it changes the stratum and it must stop
    the run whichever rule is in force.
    """

    from src.capability.context.homology import assign_homology, truncation_raises_stratum

    hit = _masked_hit("q00000", qlen=732, slen=732, nident=607, qstart=1, qend=607)
    assert truncation_raises_stratum(hit, hit.identity_over_query) is True
    for rule in ("any", "stratum_changing"):
        with pytest.raises(RuntimeError, match="truncated alignment"):
            assign_homology(
                _cohort(["A" * 732]), ["q00000"], [hit], truncation_rule=rule
            )


def test_a_terminally_offset_exact_relative_stops_only_the_strict_rule():
    """Human calmodulin, measured 2026-08-07 with ``--masking 0`` throughout.

    Eleven of 22399 alignments were flagged, every one against this query, while
    its own verbatim corpus record was found at 100% identity over all 149
    residues in the same search. Nothing was truncated; the flag's premise that
    such a relationship "does not describe any biological relationship" is false
    for a hyper-conserved protein with variable termini.
    """

    from src.capability.context.homology import assign_homology, truncation_raises_stratum

    verbatim = _masked_hit("q00003", qlen=149, slen=149, nident=149, qstart=1, qend=149)
    offset = _masked_hit("q00003", qlen=149, slen=150, nident=139, qstart=11, qend=149)
    offset = dataclasses.replace(offset, subject="UniRef50_A0A4W5PXE1")
    assert truncated_alignment(offset) is True
    assert truncation_raises_stratum(offset, 100.0) is False

    cohort = _cohort(["A" * 149])
    with pytest.raises(RuntimeError, match="truncated alignment"):
        assign_homology(cohort, ["q00003"], [verbatim, offset], truncation_rule="any")
    assignments = assign_homology(
        cohort, ["q00003"], [verbatim, offset], truncation_rule="stratum_changing"
    )
    assert assignments[0].max_identity_over_query == pytest.approx(100.0)
    assert assignments[0].stratum == "ge95_near_duplicate"


def test_an_unknown_truncation_rule_is_refused():
    from src.capability.context.homology import assign_homology

    with pytest.raises(ValueError, match="unknown truncation rule"):
        assign_homology(_cohort(["AAA"]), ["q0"], [], truncation_rule="ignore")


def test_the_potential_identity_is_the_most_generous_repair():
    from src.capability.context.homology import potential_identity_over_query

    hit = _masked_hit("q0", qlen=100, slen=100, nident=80, qstart=1, qend=80)
    assert potential_identity_over_query(hit) == pytest.approx(100.0)
    partial = _masked_hit("q0", qlen=100, slen=100, nident=40, qstart=1, qend=80)
    assert potential_identity_over_query(partial) == pytest.approx(60.0)


def test_the_masking_flag_is_still_in_the_search_command():
    """Appendix B 0.05: repeat masking is what retracted EXP-R2-050."""

    source = (REPO_ROOT / "src" / "capability" / "context" / "homology.py").read_text()
    body = source.split("def run_diamond_blastp")[1].split("\ndef ")[0]
    assert '"--masking",' in body and '"0",' in body


# ------------------------------------------------------------- the corpus scan


@pytest.mark.parametrize("chunk", [4, 7, 16, 64, 1 << 20])
def test_scan_corpus_finds_a_verbatim_member_at_every_chunk_boundary(tmp_path, chunk):
    """``count_fasta_records`` lost records on a block that ended on a newline.

    The same scan shape is used here, so the same boundary case is tested at
    several chunk sizes rather than at the default one that happens to work.
    """

    fasta = tmp_path / "corpus.fasta"
    fasta.write_text(">a desc\nACDEF\nGHIKL\n>b\nMNPQR\n>c\nACDEFGHIKLM\n")
    scan = P.scan_corpus(fasta, ["ACDEFGHIKL", "WWWWW"], chunk=chunk)
    assert scan.records == 3
    assert scan.verbatim == frozenset({"ACDEFGHIKL"})
    assert scan.residues == 10 + 5 + 11


def test_scan_corpus_background_counts_only_the_alphabet(tmp_path):
    fasta = tmp_path / "corpus.fasta"
    fasta.write_text(">a\nAAAAC\n>b\nXXXXX\n")
    scan = P.scan_corpus(fasta, [], chunk=8)
    background = scan.background
    assert background["A"] == pytest.approx(0.8)
    assert background["C"] == pytest.approx(0.2)
    assert sum(background.values()) == pytest.approx(1.0)
    assert scan.record()["non_alphabet_residues"] == 5


# ----------------------------------------------------------------- the profile


def test_a_gapped_alignment_puts_each_residue_at_its_own_query_column():
    wildtype = "ACDEFG"
    # Subject has an insertion relative to the query at column 3, so a parser
    # that walked the ungapped strings would shift every later column by one.
    hits = [_hit(qseq="ACD-EFG"[:6] + "G", sseq="ACDWEFG", qlen=6)]
    hits = [_hit(qseq="ACD-EFG", sseq="ACDWEFG", qlen=6)]
    profile = P.build_profile(wildtype, "q0", hits, max_sequences=10)
    for position, residue in enumerate(wildtype):
        assert profile.frequencies[position, P.AA20.index(residue)] == pytest.approx(1.0)


def test_ungapped_strings_of_different_length_are_refused_not_trimmed():
    with pytest.raises(ValueError, match="qseq_gapped"):
        P.build_profile(
            "ACDEFG",
            "q0",
            [_hit(qseq="ACDEFG", sseq="ACDWEFG", qlen=6)],
            max_sequences=10,
        )


def test_a_hit_below_the_coverage_floor_contributes_no_column():
    wildtype = "ACDEFGHIKL"
    partial = _hit(qseq="ACDE", sseq="ACDE", qstart=1, qlen=10)
    profile = P.build_profile(wildtype, "q0", [partial], max_sequences=10)
    assert profile.n_sequences == 0
    assert profile.column_weight.sum() == 0.0
    assert profile.neff == 0.0
    assert profile.log10_neff == 0.0


def test_one_subject_reported_twice_contributes_one_sequence():
    wildtype = "ACDEFGHIKL"
    strong = _hit(qseq=wildtype, sseq=wildtype, qlen=10, bitscore=200.0)
    weak = _hit(qseq=wildtype, sseq="ACDEFGHIKW", qlen=10, bitscore=100.0)
    profile = P.build_profile(wildtype, "q0", [strong, weak], max_sequences=10)
    assert profile.n_sequences == 1
    assert profile.frequencies[9, P.AA20.index("L")] == pytest.approx(1.0)


def test_eighty_percent_reweighting_collapses_near_duplicates():
    identical = np.zeros((5, 10), dtype=np.uint8)
    weights = P.sequence_weights(identical)
    assert weights == pytest.approx(np.full(5, 0.2))

    distinct = np.stack(
        [np.full(10, 0, dtype=np.uint8), np.full(10, 1, dtype=np.uint8)]
    )
    assert P.sequence_weights(distinct) == pytest.approx(np.ones(2))


def test_reweighting_is_invariant_to_the_block_size_it_is_computed_in():
    rng = np.random.default_rng(3)
    rows = rng.integers(0, 5, size=(40, 25)).astype(np.uint8)
    reference = P.sequence_weights(rows, budget=1 << 26)
    for budget in (1, 200, 5000):
        assert P.sequence_weights(rows, budget=budget) == pytest.approx(reference)


def test_a_column_only_gaps_see_falls_back_to_the_background_alone():
    wildtype = "ACDEFGHIKL"
    # Subject aligns over the whole query but is a deletion at position 5.
    hit = _hit(qseq=wildtype, sseq="ACDE-GHIKL", qlen=10)
    profile = P.build_profile(wildtype, "q0", [hit], max_sequences=10)
    assert profile.column_weight[4] == 0.0
    background = np.full(20, 0.05)
    codes = P.substitution_codes([("F", 5, "W")])
    assert P.lookup_score(profile.frequencies, background, codes) == pytest.approx(0.0)


def test_the_lookup_score_is_the_declared_log_ratio():
    frequencies = np.zeros((1, 20))
    frequencies[0, P.AA20.index("A")] = 0.75
    frequencies[0, P.AA20.index("C")] = 0.25
    background = np.full(20, 0.05)
    codes = P.substitution_codes([("A", 1, "C")])
    expected = np.log(0.25 + 0.05 * 0.05) - np.log(0.75 + 0.05 * 0.05)
    assert P.lookup_score(frequencies, background, codes) == pytest.approx(expected)
    assert expected < 0.0


def test_a_substitution_outside_the_profile_is_an_error_not_a_silent_zero():
    frequencies = np.zeros((3, 20))
    background = np.full(20, 0.05)
    with pytest.raises(IndexError):
        P.lookup_score(frequencies, background, P.substitution_codes([("A", 9, "C")]))


def test_scoring_a_variant_of_a_different_protein_is_refused_by_default():
    wildtype = "ACDEFGHIKL"
    profile = P.build_profile(
        wildtype, "q0", [_hit(qseq=wildtype, sseq=wildtype, qlen=10)], max_sequences=4
    )
    background = np.full(20, 0.05)
    with pytest.raises(ValueError, match="wild type carries"):
        P.profile_scores(profile, background, [[("W", 1, "C")]])
    # The mismatched-profile control asks for exactly this, and says so.
    assert P.profile_scores(
        profile, background, [[("W", 1, "C")]], check_wildtype=False
    ).shape == (1,)


def test_free_baselines_are_free_of_the_corpus_and_the_labels():
    background = np.full(20, 0.05)
    values = P.free_baselines([[("A", 10, "W")], [("W", 20, "A")]], background, wildtype_length=100)
    assert values["position_index"] == pytest.approx([0.1, 0.2])
    assert values["hydropathy_change"] == pytest.approx(
        [P.KYTE_DOOLITTLE["W"] - P.KYTE_DOOLITTLE["A"], P.KYTE_DOOLITTLE["A"] - P.KYTE_DOOLITTLE["W"]]
    )
    # A flat background makes the composition baseline exactly zero, which is
    # the property that makes it the degenerate limit of the lookup channel.
    assert values["background_composition"] == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------- the clusters


def _self_hit(query, subject, *, nident, qlen, slen, aligned):
    return Hit(
        query=query,
        subject=subject,
        pident=100.0 * nident / aligned,
        length=aligned,
        nident=nident,
        qstart=1,
        qend=aligned,
        qlen=qlen,
        slen=slen,
        evalue=1e-9,
        bitscore=100.0,
    )


def test_clustering_is_single_linkage_at_the_declared_identity():
    lengths = {"a": 100, "b": 100, "c": 100, "d": 100}
    hits = [
        _self_hit("a", "b", nident=60, qlen=100, slen=100, aligned=100),
        _self_hit("b", "c", nident=55, qlen=100, slen=100, aligned=100),
        # Below the identity floor: d stays on its own.
        _self_hit("c", "d", nident=30, qlen=100, slen=100, aligned=100),
    ]
    clusters = P.cluster_by_identity(["a", "b", "c", "d"], lengths, hits)
    assert clusters["a"] == clusters["b"] == clusters["c"]
    assert clusters["d"] != clusters["a"]


def test_a_high_identity_hit_over_too_little_of_the_shorter_sequence_is_not_a_family():
    lengths = {"a": 100, "b": 100}
    hits = [_self_hit("a", "b", nident=60, qlen=100, slen=100, aligned=60)]
    clusters = P.cluster_by_identity(["a", "b"], lengths, hits)
    assert clusters["a"] != clusters["b"]


def test_a_mismatched_donor_is_never_from_the_recipients_own_family():
    identifiers = ["a", "b", "c"]
    lengths = {"a": 100, "b": 110, "c": 105}
    neff = {"a": 2.0, "b": 2.0, "c": 2.0}
    clusters = {"a": 0, "b": 0, "c": 1}
    donors = P.mismatched_donors(identifiers, lengths, neff, clusters)
    assert donors["a"]["donor"] == "c"
    assert donors["c"]["donor"] == "b"
    # A donor is always at least as long, so every substituted position has a
    # column and the control never degenerates into the background baseline.
    for identifier, record in donors.items():
        if record["donor"] is not None:
            assert lengths[record["donor"]] >= lengths[identifier]


def test_a_wild_type_with_no_eligible_donor_is_recorded_rather_than_dropped():
    donors = P.mismatched_donors(
        ["a", "b"], {"a": 500, "b": 100}, {"a": 1.0, "b": 1.0}, {"a": 0, "b": 1}
    )
    assert donors["a"]["donor"] is None and "reason" in donors["a"]
    assert donors["b"]["donor"] == "a"


# -------------------------------------------------------------- the statistics


def test_cluster_means_make_a_repeated_family_one_unit():
    means, labels = P.cluster_means([1.0, 1.0, 1.0, 5.0], [0, 0, 0, 1])
    assert means == pytest.approx([1.0, 5.0])
    assert labels.tolist() == [0, 1]


def test_a_bootstrap_below_the_unit_floor_publishes_no_interval():
    record = P.cluster_bootstrap([1.0, 2.0, 3.0], [0, 1, 2], resamples=64, seed=1)
    assert record["degenerate"] is True
    assert record["interval"] is None
    assert "unit floor" in record["degenerate_reason"]


def test_a_bootstrap_above_the_floor_separates_a_clearly_positive_sample():
    values = [0.4 + 0.01 * index for index in range(12)]
    record = P.cluster_bootstrap(values, list(range(12)), resamples=2000, seed=1)
    assert record["degenerate"] is False
    assert record["interval"][0] > 0.0 and record["excludes_zero"] is True
    centred = [value - 0.45 for value in values]
    assert P.cluster_bootstrap(centred, list(range(12)), resamples=2000, seed=1)[
        "excludes_zero"
    ] is False


def test_the_bin_sweep_reports_an_ordering_only_when_every_partition_agrees():
    axis = np.linspace(0.0, 100.0, 40)
    rising = P.bin_sweep(axis, axis / 100.0, P.IDENTITY_EDGE_SWEEP)
    assert rising["ordering_invariant"] is True
    assert set(rising["signs"]) <= {1, 0}

    # A gradient that reverses inside the axis: the coarse cut and the fine cut
    # disagree, which is exactly the case Appendix B rule 17 exists for.
    folded = np.abs(axis - 50.0) / 100.0
    mixed = P.bin_sweep(axis, folded, ((0.0, 60.0, 101.0), (0.0, 20.0, 101.0)))
    assert mixed["ordering_invariant"] is False


def test_a_bin_below_the_unit_floor_is_marked_and_excluded_from_the_ordering():
    axis = np.array([1.0, 2.0, 3.0] + [90.0] * 12)
    values = np.array([0.0, 0.0, 0.0] + [1.0] * 12)
    sweep = P.bin_sweep(axis, values, ((0.0, 50.0, 101.0),))
    low = sweep["partitions"][0]["bins"][0]
    assert low["n_units"] == 3 and low["below_unit_floor"] is True
    assert sweep["signs"] == [0]


def test_the_retrieval_share_is_resampled_as_one_ratio():
    """Both halves are measured on the same assays, so the draw has to be shared."""

    rng = np.random.default_rng(9)
    clusters = list(range(16))
    denominator = 0.06 + 0.005 * rng.standard_normal(16)
    numerator = 0.5 * denominator
    record = P.share_bootstrap(
        numerator, denominator, clusters, resamples=2000, seed=1
    )
    assert record["share"] == pytest.approx(0.5, abs=0.02)
    # A ratio of two exactly proportional vectors is constant draw for draw, so
    # the interval collapses -- which a pair of independent bootstraps could not
    # reproduce and is the property this function exists for.
    assert record["interval"][1] - record["interval"][0] < 1e-6


def test_the_share_is_withheld_when_its_denominator_can_vanish():
    rng = np.random.default_rng(4)
    clusters = list(range(16))
    denominator = 0.3 * rng.standard_normal(16)
    record = P.share_bootstrap(
        np.ones(16) * 0.05, denominator, clusters, resamples=500, seed=2
    )
    assert record["share"] is None or abs(record["share"]) < 1e9
    if record["share"] is None:
        assert "withheld_reason" in record

    zero = P.share_bootstrap(np.ones(16), np.zeros(16), clusters, resamples=64, seed=2)
    assert zero["share"] is None and "zero on the full sample" in zero["withheld_reason"]


def test_the_share_respects_the_same_unit_floor():
    record = P.share_bootstrap([1.0] * 4, [2.0] * 4, list(range(4)), resamples=64, seed=1)
    assert record["degenerate"] is True and record["share"] is None


def test_kendall_tau_is_undefined_rather_than_zero_when_one_side_is_constant():
    assert P.kendall_tau([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])["undefined"] is True
    assert P.kendall_tau([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])["tau"] == pytest.approx(1.0)


def test_the_difficulty_control_is_fitted_out_of_fold_and_cannot_absorb_the_signal():
    """A covariate that only identifies the training clusters must remove nothing.

    This is the property a partial correlation does not have. The covariate below
    is the cluster's own index, which explains the outcome perfectly *within* any
    training set and carries no information about a family it has never seen; a
    within-sample adjustment would drive the residual to zero and manufacture a
    null.
    """

    rng = np.random.default_rng(5)
    clusters = np.repeat(np.arange(20), 2)
    effect = 0.3 + 0.02 * rng.standard_normal(clusters.size)
    covariates = clusters.reshape(-1, 1).astype(float)
    control = P.out_of_fold_difficulty_residual(
        effect, covariates, clusters, n_splits=5, seed=1
    )
    assert control["out_of_fold_r2"] < 0.3
    assert float(np.mean(control["residual"])) == pytest.approx(0.0, abs=0.05)


def test_the_difficulty_control_removes_a_covariate_that_really_predicts():
    rng = np.random.default_rng(6)
    clusters = np.repeat(np.arange(20), 2)
    covariate = rng.standard_normal(clusters.size)
    effect = 0.5 * covariate + 0.01 * rng.standard_normal(clusters.size)
    control = P.out_of_fold_difficulty_residual(
        effect, covariate.reshape(-1, 1), clusters, n_splits=5, seed=1
    )
    assert control["out_of_fold_r2"] > 0.9
    assert float(np.std(control["residual"])) < float(np.std(effect))


def test_the_difficulty_control_refuses_a_non_finite_input():
    with pytest.raises(ValueError):
        P.out_of_fold_difficulty_residual(
            [1.0, 2.0, np.nan, 4.0] * 3,
            np.zeros((12, 1)),
            np.arange(12),
            n_splits=3,
            seed=1,
        )


# ----------------------------------------------------------------- the verdict


def _boot(point, interval):
    return {"point": point, "interval": interval, "degenerate": False}


def test_the_three_pre_registered_outcomes():
    total = _boot(0.0647, [0.0386, 0.0909])
    assert P.equivalence_verdict(_boot(0.05, [0.02, 0.08]), total)["verdict"] == "acquired"
    assert (
        P.equivalence_verdict(_boot(0.001, [-0.01, 0.01]), total)["verdict"]
        == "retrieval_bounded"
    )
    assert (
        P.equivalence_verdict(_boot(0.005, [-0.09, 0.10]), total)["verdict"]
        == "indeterminate"
    )


def test_no_equivalence_is_claimed_when_there_is_no_advantage_to_partition():
    total = _boot(0.004, [-0.02, 0.03])
    verdict = P.equivalence_verdict(_boot(0.0, [-0.001, 0.001]), total)
    assert verdict["verdict"] == "indeterminate"
    assert "not itself separable" in verdict["reason"]


def test_a_degenerate_interval_cannot_produce_a_verdict():
    degenerate = {"point": 0.2, "interval": None, "degenerate": True,
                  "degenerate_reason": "too few units"}
    assert (
        P.equivalence_verdict(degenerate, _boot(0.06, [0.03, 0.09]))["verdict"]
        == "indeterminate"
    )


# ------------------------------------------------------------------ the anchor


def test_a_verbatim_corpus_member_binned_below_the_top_stratum_fails_the_anchor():
    wildtypes = {"q0": "ACDEF", "q1": "GHIKL"}
    passing = P.verbatim_anchor_check(["ACDEF"], wildtypes, {"q0": 100.0, "q1": 40.0})
    assert passing["passes"] is True and passing["anchors"] == 1

    failing = P.verbatim_anchor_check(["ACDEF"], wildtypes, {"q0": 82.9, "q1": 40.0})
    assert failing["passes"] is False
    assert "--masking 0" in failing["message"]
    assert failing["failures"][0]["query_id"] == "q0"


def test_the_anchor_refuses_a_verbatim_member_it_was_not_given():
    with pytest.raises(KeyError):
        P.verbatim_anchor_check(["WWWWW"], {"q0": "ACDEF"}, {"q0": 100.0})


# ------------------------------------------------------------------- the stage


def test_the_stage_declares_its_corpora_from_the_panel_not_by_hand():
    """Rule 12: a second spelling of a pretraining corpus is a second fact."""

    stage = _stage()
    for arm in ("protgpt2", "progen2-medium"):
        assert stage.ARM_CORPUS[arm]["declared"] == PANEL[arm].pretraining_corpus
    # ProGen3 is not a panel member and its card states no corpus; the stage must
    # say so rather than attribute one to it.
    assert stage.ARM_CORPUS["progen3-112m"]["declared"] == "undeclared"
    assert stage.ARM_CORPUS["protgpt2"]["identification"] == "exact"
    for arm in ("progen2-medium", "progen3-112m"):
        assert "lower bound" in stage.ARM_CORPUS[arm]["identification"]


def _lookup_payload(n=12, *, donor_missing=1):
    """A minimal `lookup.json` body: one cluster per assay, declared channels."""

    rng = np.random.default_rng(2)
    rows = []
    for index in range(n):
        spearman = {
            "lookup": 0.40 + 0.01 * index,
            "blosum62": 0.20 + 0.005 * index,
            "background_composition": 0.05,
            "position_index": 0.03,
            "wt_hydropathy": -0.04,
            "hydropathy_change": 0.06,
        }
        if index >= donor_missing:
            spearman["mismatched_profile"] = 0.02
        rows.append(
            {
                "assay": f"assay{index}",
                "wildtype_id": f"q{index:05d}",
                "cluster": index,
                "n_variants": 100,
                "mutant_digest": f"d{index}",
                "spearman": spearman,
                "shuffled_label_spearman": {k: 0.01 * rng.standard_normal() for k in spearman},
                "alpha_sweep": {"0.05": spearman["lookup"]},
                "difficulty": {
                    "log10_wildtype_length": 2.0,
                    "log10_variants": 2.0,
                    "multi_substitution_fraction": 0.0,
                    "mean_substitutions": 1.0,
                    "dms_score_sd": 1.0,
                },
            }
        )
    return {"assays": rows, "donors": {f"q{i:05d}": {"donor": None if i < donor_missing else "x",
                                                     "length_matched": True,
                                                     "neff_matched": True} for i in range(n)}}


def test_one_wild_type_without_a_donor_does_not_delete_the_mismatch_control():
    """It did. The longest wild type has nothing longer to borrow columns from.

    An intersection over every channel's assay set dropped ``mismatched_profile``
    entirely, and the gate reported ``measured: false`` -- a negative control
    silently absent from a run whose whole point is that its controls fired.
    """

    import argparse

    stage = _stage()
    args = argparse.Namespace(bootstrap=500, seed=1)
    controls = stage._channel_controls(_lookup_payload(), args)
    mismatch = controls["mismatched_profile"]
    assert mismatch["measured"] is True
    assert mismatch["assays_with_a_donor"] == 11 and mismatch["assays_total"] == 12
    assert mismatch["passes"] is True
    assert controls["positive_control"]["passes"] is True
    # A whole-benchmark reproduction claim is withheld on a subset.
    assert controls["positive_control"]["blosum62_reproduces_frozen"] is None


def test_the_mismatch_gate_reports_a_defect_when_the_donor_profile_wins():
    import argparse

    stage = _stage()
    payload = _lookup_payload(donor_missing=0)
    for row in payload["assays"]:
        row["spearman"]["mismatched_profile"] = 0.9
    controls = stage._channel_controls(payload, argparse.Namespace(bootstrap=500, seed=1))
    assert controls["mismatched_profile"]["passes"] is False


def test_the_stage_never_truncates_a_variant_to_fit_a_context():
    """Truncation would score a sequence that need not contain the mutation."""

    source = (REPO_ROOT / "scripts" / "capability" / "stages/retrieval_bound.py").read_text()
    assert "the rendered variant exceeds this arm's context" in source


def test_the_positive_control_floor_is_a_locally_measured_number():
    assert P.FROZEN_BLOSUM62_MEAN_SPEARMAN == pytest.approx(0.2098)
    assert P.FROZEN_MODEL_MINUS_BLOSUM == pytest.approx(0.0647)
    assert 0.0 < P.EQUIVALENCE_FRACTION < 1.0


def test_the_label_shuffle_gate_reads_a_correlation_against_its_own_assay_size():
    """The defect: a raw maximum measures the smallest assay, not calibration.

    A Spearman correlation under permuted labels has null scale 1/sqrt(n-1), and
    this cohort's assays span n = 63 to 1000. The real run's largest shuffled
    correlation, 0.2467, came from its single smallest assay (n=63, null scale
    0.127) and is 1.9 of its own standard deviations -- nothing at all. Compared
    to a constant 0.2 it failed the gate and would have blocked a correctly
    calibrated run.
    """

    import argparse

    stage = _stage()
    args = argparse.Namespace(bootstrap=200, seed=1)

    payload = _lookup_payload()
    small = payload["assays"][0]
    small["n_variants"] = 63
    small["shuffled_label_spearman"]["lookup"] = 0.2467

    controls = stage._channel_controls(payload, args)
    shuffle = controls["label_shuffle"]

    # The raw quantity that produced the spurious failure is still reported...
    assert shuffle["max_abs_spearman"] == pytest.approx(0.2467)
    assert shuffle["max_abs_spearman"] > 0.2
    # ...and standardising it by the assay's own null scale clears the gate.
    assert shuffle["max_abs_z"] == pytest.approx(0.2467 * math.sqrt(62), rel=1e-6)
    assert shuffle["max_abs_z"] < shuffle["critical_z"]
    assert shuffle["passes"] is True


def test_the_label_shuffle_gate_still_fails_a_channel_that_reads_permuted_labels():
    """The guard must keep its teeth: a large assay carrying real signal fails."""

    import argparse

    stage = _stage()
    args = argparse.Namespace(bootstrap=200, seed=1)

    payload = _lookup_payload()
    loud = payload["assays"][0]
    loud["n_variants"] = 1000
    loud["shuffled_label_spearman"]["lookup"] = 0.30

    shuffle = stage._channel_controls(payload, args)["label_shuffle"]
    assert shuffle["max_abs_z"] > shuffle["critical_z"]
    assert shuffle["passes"] is False


def _interval(point, low, high):
    return {"point": point, "interval": [low, high]}


def test_a_lookup_that_beats_the_model_is_resolved_not_indeterminate():
    """ProGen3-112M's measured case, which the original vocabulary had no room for.

    MODEL - LOOKUP -0.0808 [-0.1139, -0.0481] against an equivalence bound of
    0.0340. It fell through to `indeterminate`, whose reason asserted the
    interval spanned zero -- it does not -- and whose documented meaning is that
    the cohort could not resolve the question, when it had.
    """

    verdict = P.equivalence_verdict(
        _interval(-0.0808, -0.11387776, -0.04812502),
        _interval(0.0681, 0.03671601, 0.09638934),
    )
    assert verdict["verdict"] == "retrieval_dominated"
    assert "wholly below" in verdict["reason"]
    assert "spans zero" not in verdict["reason"]


def test_the_three_original_readings_are_unchanged():
    """The added category must not have moved any boundary that already existed."""

    bounded = P.equivalence_verdict(
        _interval(-0.0143, -0.03808299, 0.01014614),
        _interval(0.1343, 0.10558444, 0.16073318),
    )
    assert bounded["verdict"] == "retrieval_bounded"

    acquired = P.equivalence_verdict(_interval(0.09, 0.05, 0.13), _interval(0.10, 0.08, 0.12))
    assert acquired["verdict"] == "acquired"

    # Spans zero and leaves the bound: genuinely unresolved, and still says so.
    unresolved = P.equivalence_verdict(_interval(0.0, -0.30, 0.30), _interval(0.10, 0.08, 0.12))
    assert unresolved["verdict"] == "indeterminate"
    assert "spans zero" in unresolved["reason"]

    # No separable advantage to partition: withheld, with its own reason.
    no_base = P.equivalence_verdict(_interval(-0.5, -0.6, -0.4), _interval(0.01, -0.05, 0.07))
    assert no_base["verdict"] == "indeterminate"
    assert "not itself separable" in no_base["reason"]


def test_an_interval_below_zero_but_astride_the_bound_is_still_indeterminate():
    """The gap between 'equivalent' and 'resolved against' must not be papered over."""

    verdict = P.equivalence_verdict(
        _interval(-0.05, -0.09, -0.01), _interval(0.10, 0.08, 0.12)
    )
    assert verdict["verdict"] == "indeterminate"
    assert "not wholly below" in verdict["reason"]
