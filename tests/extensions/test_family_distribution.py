"""Lossless native strings, structure-preserving nulls, splits and contrasts.

Every fixture is synthetic, so these tests need neither the corpus TSV nor a GPU.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pytest

from src.capability.context.homology import Hit
from src.capability.extensions import family_distribution as fd
from src.capability.extensions.family_distribution import (
    NativeError,
    cluster_rows,
    column_null,
    composition_null,
    contrast,
    duplicate_fragment_groups,
    markov_model,
    markov_score,
    native_row,
    per_residue_spearman,
    profile_batch_audit,
    profile_frequencies,
    profile_score,
    qualify,
    read_config,
    rebuild_gapped,
    select_family_wildtypes,
    split_clusters,
    verify_column_null,
)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = read_config(ROOT / "configs/family_distribution.json")
WILDTYPE = "ACDEFGHIKLMNPQRSTVWY" * 3


def make_hit(columns, *, query="q00000", subject="UniRef50_TEST", wildtype=WILDTYPE,
             bitscore=100.0, **overrides) -> Hit:
    query_gapped = "".join(query_residue for query_residue, _ in columns)
    subject_gapped = "".join(subject_residue for _, subject_residue in columns)
    identity = sum(1 for left, right in columns if left == right and left != "-")
    values: dict = dict(
        query=query, subject=subject, pident=100.0 * identity / max(len(query_gapped), 1),
        length=len(query_gapped), nident=identity, qstart=1,
        qend=len(query_gapped.replace("-", "")), qlen=len(wildtype),
        slen=len(subject_gapped), evalue=0.0, bitscore=bitscore,
        qseq_gapped=query_gapped, sseq_gapped=subject_gapped,
    )
    values.update(overrides)
    return Hit(**values)


def matched(wildtype=WILDTYPE, mutations=None):
    mutations = mutations or {}
    return [(residue, mutations.get(position, residue)) for position, residue in enumerate(wildtype)]


def with_insertion(columns, at, inserted="GG"):
    return columns[:at] + [("-", residue) for residue in inserted] + columns[at:]


def with_deletion(columns, positions):
    """Delete the subject residue at these 0-based wild-type positions."""

    out = []
    position = 0
    for query, subject in columns:
        if query != "-":
            if position in positions:
                subject = "-"
            position += 1
        out.append((query, subject))
    return out


def row(columns, *, row_id="q00000#0000", wildtype=WILDTYPE, subject="UniRef50_TEST", bitscore=100.0):
    return native_row(row_id, wildtype, make_hit(columns, wildtype=wildtype, subject=subject,
                                                 bitscore=bitscore))


# ------------------------------------------------------------ native strings


def test_native_roundtrip_with_insertions_and_deletions():
    columns = with_deletion(with_insertion(matched(), 5), {30, 31})
    hit = make_hit(columns)
    native = native_row("q00000#0000", WILDTYPE, hit)
    assert rebuild_gapped(native, WILDTYPE) == (hit.qseq_gapped, hit.sseq_gapped)
    assert len(native.native) == len(WILDTYPE) - 2 + 2
    inserted = [index for index, position in enumerate(native.query_index) if position < 0]
    assert [native.native[index] for index in inserted] == ["G", "G"]
    assert native.query_index[5] == -1 and native.query_index[7] == 5
    covered = {int(position) for position in native.query_index if position >= 0}
    assert covered == set(range(len(WILDTYPE))) - {30, 31}
    assert sum(stop - start for start, stop in native.fragments) == len(native.native)
    assert len(native.fragments) == 2
    # The query residues that survived are the wild type's own, verbatim.
    for index, position in enumerate(native.query_index):
        if position >= 0 and position not in (30, 31) and native.native[index] == WILDTYPE[position]:
            continue
        assert position < 0 or native.native[index] == WILDTYPE[position]


def test_native_refusals_are_explicit():
    with pytest.raises(NativeError, match="qlen_mismatch"):
        native_row("q00000#0000", WILDTYPE, make_hit(matched(), qlen=len(WILDTYPE) + 1))
    shifted = [("W", subject) for _, subject in matched()]
    with pytest.raises(NativeError, match="query_residues_not_wildtype"):
        native_row("q00000#0000", WILDTYPE, make_hit(shifted))
    noncanonical = matched()
    noncanonical[0] = (noncanonical[0][0], "X")
    with pytest.raises(NativeError, match="noncanonical_subject"):
        native_row("q00000#0000", WILDTYPE, make_hit(noncanonical))
    short = "ACDEFGHIKLMNPQRSTVWY"
    with pytest.raises(NativeError, match="length_out_of_range"):
        native_row("q00000#0000", short, make_hit(matched(short), wildtype=short))
    # A deletion column followed by an insertion column cannot be rebuilt in the
    # same order from the native string, so it is refused instead of reordered.
    columns = matched()[:30] + [(WILDTYPE[30], "-"), ("-", "G")] + matched()[31:]
    with pytest.raises(NativeError, match="roundtrip_mismatch"):
        native_row("q00000#0000", WILDTYPE, make_hit(columns))


# --------------------------------------------------------------------- nulls


def four_rows():
    first = row(matched(), row_id="q00000#0000", subject="A")
    second = row(with_insertion(matched(mutations={7: "W", 20: "A"}), 10),
                 row_id="q00000#0001", subject="B")
    third = row(with_deletion(matched(mutations={3: "K", 15: "D", 40: "R"}), {25}),
                row_id="q00000#0002", subject="C")
    fourth = row(matched(mutations={0: "H", 31: "F"}), row_id="q00000#0003", subject="D")
    return [first, second, third, fourth]


def test_column_null_preserves_lengths_insertions_gaps_and_fragments():
    rows = four_rows()
    shuffled = column_null(rows, 20261106)
    verify_column_null(rows, shuffled)
    for original, value in zip(rows, shuffled):
        assert len(value) == len(original.native)
        assert set(value) <= set(fd.AA20)
        assert np.array_equal(original.query_index, original.query_index)
        for index, position in enumerate(original.query_index):
            if position < 0:
                assert value[index] == original.native[index]
    assert any(value != original.native for original, value in zip(rows, shuffled))
    # Shuffling twice with the same seed is the same null, and one query column
    # keeps its residue multiset even across variable-length rows.
    assert column_null(rows, 20261106) == shuffled
    assert column_null(rows, 20261107) != shuffled


def test_column_null_verification_rejects_a_moved_insertion():
    rows = four_rows()
    shuffled = column_null(rows, 20261106)
    tampered = list(shuffled)
    index = next(index for index, position in enumerate(rows[1].query_index) if position < 0)
    characters = list(tampered[1])
    characters[index] = "A" if characters[index] != "A" else "C"
    tampered[1] = "".join(characters)
    with pytest.raises(RuntimeError, match="inserted residue"):
        verify_column_null(rows, tampered)


def test_composition_null_preserves_composition_and_length():
    original = four_rows()[1]
    shuffled = composition_null(original, 20261106)
    assert len(shuffled) == len(original.native)
    assert sorted(shuffled) == sorted(original.native)
    assert shuffled != original.native


# ------------------------------------------------------- independent-site work


def test_profile_pseudocount_and_additive_batches():
    rows = four_rows()
    profile = profile_frequencies(rows, pseudocount=0.05)
    for distribution in profile.values():
        assert abs(float(distribution.sum()) - 1.0) < 1e-12
    # One row carries H at column 0 instead of the wild-type A, so the declared
    # 1/20 pseudocount is added to each of the twenty residue counts and the
    # column is then renormalised.
    assert profile[0][fd.AA_INDEX["A"]] == pytest.approx((3 + 0.05) / (4 + 1))
    assert profile[0][fd.AA_INDEX["H"]] == pytest.approx((1 + 0.05) / (4 + 1))
    assert profile[0][fd.AA_INDEX["C"]] == pytest.approx(0.05 / (4 + 1))
    assert profile_batch_audit(rows, profile) == 0.0


def test_profile_zero_invariant_breaks_only_under_per_sequence_normalisation():
    rows = four_rows()
    lengths = np.asarray([len(original.native) for original in rows], dtype=float)
    assert len(set(lengths)) > 1, "the fixture must vary in length"
    profile = profile_frequencies(rows, pseudocount=0.05)
    native = np.asarray([profile_score(original, profile) for original in rows])
    permuted = []
    for seed in CONFIG["null"]["column_shuffle_seeds"]:
        shuffled = column_null(rows, seed)
        permuted.append([profile_score(row, profile) for row in
                         (fd.replace(original, native=value) for original, value in zip(rows, shuffled))])
    total = contrast(-native, -np.asarray(permuted), lengths)
    assert abs(total["nats_per_residue"]) < 1e-12
    # The wrong reduction -- divide each sequence first, then average -- is not zero.
    null_mean_per_row = np.asarray(permuted).mean(axis=0)
    wrong = float(np.mean((null_mean_per_row - native) / lengths))
    assert abs(wrong) > 1e-6


def test_markov_is_fitted_on_fragments_and_finite():
    rows = four_rows()
    model = markov_model(rows, pseudocount=0.05)
    assert abs(float(model["start"].sum()) - 1.0) < 1e-12
    assert np.allclose(model["transitions"].sum(axis=1), 1.0)
    assert np.all(model["transitions"] > 0)
    for original in rows:
        assert np.isfinite(markov_score(original, model))


# --------------------------------------------------------------- queue protocol


def test_qualification_and_split_are_declared_and_disjoint():
    assert qualify(15, CONFIG) == (False, "fewer_than_16_clusters")
    assert qualify(16, CONFIG) == (True, "qualified")
    rng = np.random.default_rng(CONFIG["clustering"]["split_seed"])
    split = split_clusters(16, CONFIG, rng)
    assert split["test_clusters"] == 8 and split["train_clusters"] == 8
    assert not set(split["test"]) & set(split["train"])
    assert len(set(split["test"]) | set(split["train"])) == 16
    large = split_clusters(80, CONFIG, np.random.default_rng(1))
    assert large["test_clusters"] == CONFIG["clustering"]["max_test_clusters"]
    with pytest.raises(ValueError, match="training clusters"):
        split_clusters(12, CONFIG, np.random.default_rng(1))


def test_family_selection_prefers_depth_and_excludes_unsupported_families():
    by_family = {1: ["q1", "q2"], 2: ["q3"], 3: ["q4", "q5"]}
    clusters = {"q1": (20, True), "q2": (30, True), "q3": (5, False),
                "q4": (16, True), "q5": (16, True)}
    selected, selection = select_family_wildtypes(by_family, clusters)
    assert selected == ["q2", "q4"]
    reasons = {entry["family_cluster"]: entry for entry in selection}
    assert reasons[1]["reason"] == "most_qualified_clusters"
    assert reasons[2]["selected"] is None
    assert reasons[2]["reason"] == "no_qualified_wildtype_in_family"
    assert reasons[3]["reason"] == "most_qualified_clusters"


def test_cluster_rows_keeps_insertion_variants_together():
    first = row(matched(), row_id="q00000#0000", subject="A")
    second = row(with_insertion(matched(mutations={5: "W", 9: "Y", 14: "R", 44: "T"}), 10),
                 row_id="q00000#0001", subject="B")
    # Half of a covering fragment's residues differ, so it stays its own cluster
    # even though its insertions and query span would merge it under a rule that
    # treated insertion agreement as evidence.
    varied = [(residue, "G" if index % 2 == 0 else residue)
              for index, residue in enumerate(WILDTYPE[:35])]
    fragment = row(with_insertion(varied, 20, inserted="SSSSS"),
                   row_id="q00000#0002", subject="C")
    labels = cluster_rows([first, second, fragment], len(WILDTYPE), CONFIG)
    assert labels[0] == labels[1]
    assert labels[2] not in (labels[0],)
    assert len(fragment.native) >= 40


def test_repeated_native_fragment_merges_units():
    shared = "ACDEFGHIKLMNPQRSTVWY" * 2 + "ACDEFG"
    longer = shared + "YYWWKKLLMMNNPPQQ"
    unrelated = "YVTSRQPNMLKIHGFEDCA" * 3
    groups, evidence = duplicate_fragment_groups(
        {"u001": [shared], "u002": [longer], "u003": [unrelated]}, config=CONFIG)
    assert sorted(sorted(members) for members in groups) == [["u001", "u002"], ["u003"]]
    assert evidence and evidence[0]["kind"] == "contained_native"


# ------------------------------------------------------------------- contrasts


def test_contrast_uses_the_nats_sum_sign_convention():
    # log w = -nll_sum_nats, so a native sum of 20 against a null mean of 24 is
    # +4 nats over a total length of 5, and a native sum of 24 is zero.
    result = contrast([10.0, 10.0], [[12.0, 12.0]] * 4, [2.0, 3.0])
    assert result["nats_per_residue"] == pytest.approx(4.0 / 5.0)
    assert result["nulls"] == 4 and result["total_length"] == 5.0
    equal = contrast([14.0, 10.0], [[12.0, 12.0]] * 4, [2.0, 3.0])
    assert equal["nats_per_residue"] == pytest.approx(0.0)


def test_contrast_refuses_nonfinite_and_misshapen_input():
    with pytest.raises(ValueError, match="finite"):
        contrast([float("nan")], [[1.0]], [1.0])
    with pytest.raises(ValueError, match="finite"):
        contrast([1.0], [[float("inf")]], [1.0])
    with pytest.raises(ValueError, match="matching lengths"):
        contrast([1.0, 2.0], [[1.0, 2.0]], [1.0])
    with pytest.raises(ValueError, match="native vector"):
        contrast(np.asarray([[1.0]]), np.asarray([[1.0]]), np.asarray([1.0]))


def test_stored_scores_reject_identity_mismatch_duplicates_and_nonfinite():
    rows = four_rows()
    stored = {
        "row_ids": [original.row_id for original in rows],
        "lengths": [len(original.native) for original in rows],
        "native_nll": [1.0] * 4,
        "column_null_nll": [[1.0] * 4] * 4,
        "composition_null_nll": [1.0] * 4,
        "scored_tokens": {"native": [2] * 4, "column_null": [[2] * 4] * 4,
                          "composition_null": [2] * 4},
    }
    fd._stored_scores(stored, rows, CONFIG, "arm")
    wrong_ids = dict(stored, row_ids=["q00000#9999"] + stored["row_ids"][1:])
    with pytest.raises(ValueError, match="frozen queue rows"):
        fd._stored_scores(wrong_ids, rows, CONFIG, "arm")
    duplicate = dict(stored, row_ids=[stored["row_ids"][0]] * 4)
    with pytest.raises(ValueError, match="repeats a row id"):
        fd._stored_scores(duplicate, rows, CONFIG, "arm")
    nonfinite = dict(stored, native_nll=[1.0, 1.0, 1.0, float("nan")])
    with pytest.raises(ValueError, match="not finite"):
        fd._stored_scores(nonfinite, rows, CONFIG, "arm")
    misshapen = dict(stored, column_null_nll=[[1.0] * 4] * 3)
    with pytest.raises(ValueError, match="wrong shape"):
        fd._stored_scores(misshapen, rows, CONFIG, "arm")
    no_tokens = {key: value for key, value in stored.items() if key != "scored_tokens"}
    with pytest.raises(ValueError, match="no scored-token counts"):
        fd._stored_scores(no_tokens, rows, CONFIG, "arm")
    short_tokens = dict(stored, scored_tokens=dict(stored["scored_tokens"], native=[2] * 3))
    with pytest.raises(ValueError, match="wrong shape"):
        fd._stored_scores(short_tokens, rows, CONFIG, "arm")


def test_scored_token_diagnostic_sign_and_scale():
    lengths = np.asarray([10.0, 30.0])
    equal = fd.token_compression([5, 15], [[5, 15]] * 4, lengths)
    assert equal["tokens_per_residue"] == pytest.approx(0.0)
    more_targets_in_null = fd.token_compression([5, 15], [[6, 16]] * 4, lengths)
    assert more_targets_in_null["tokens_per_residue"] == pytest.approx(2.0 / 40.0)
    assert more_targets_in_null["total_residues"] == 40.0
    fewer = fd.token_compression([5, 15], [[4, 14]] * 4, lengths)
    assert fewer["tokens_per_residue"] == pytest.approx(-2.0 / 40.0)
    with pytest.raises(ValueError, match="finite"):
        fd.token_compression([5, 15], [[float("nan"), 15]] * 4, lengths)


def test_per_residue_spearman_tracks_rank_and_is_absent_when_degenerate():
    lengths = np.full(40, 100.0)
    reference = np.linspace(-8.0, 8.0, 40)
    aligned = -reference * lengths            # nll shares the reference's order reversed
    assert per_residue_spearman(aligned, reference, lengths) == pytest.approx(1.0)
    independent = np.random.default_rng(7).normal(size=40)
    independent_rho = per_residue_spearman(independent, reference, lengths)
    assert independent_rho is not None and abs(independent_rho) < 0.5
    assert per_residue_spearman(np.zeros(40), reference, lengths) is None
    assert per_residue_spearman(np.zeros(3), reference[:3], lengths[:3]) is None

# ------------------------------------------------------------- analysis smoke


def synthetic_queue(units=8, rows_per_wildtype=8):
    """A tiny frozen queue whose only purpose is to exercise the analysis path."""

    rows = four_rows()
    # Coupled adjacent heldout sites make the synthetic Markov contrast
    # non-additive; isolated single-site changes alone would cancel exactly.
    rows[2] = fd.replace(rows[2], native="WW" + rows[2].native[2:])
    rows[3] = fd.replace(rows[3], native="HH" + rows[3].native[2:])
    extra = [row(matched(mutations={index: "V" for index in (2, 12, 22)}),
                 row_id=f"q00000#{index:04d}", subject=f"E{index}") for index in range(2)]
    pool = rows + extra
    units_payload = []
    for index in range(units):
        payloads = []
        for position in range(rows_per_wildtype):
            original = pool[position % len(pool)]
            payload = original.payload()
            payload["row_id"] = f"q{index:05d}#{position:04d}"
            payload["cluster"] = f"c{position // 2 + (position >= 3):03d}"
            payload["split"] = "test" if position < 4 else "train"
            payload["representative"] = position not in (1, 5, 7)
            payloads.append(payload)
        units_payload.append({
            "unit_id": f"u{index:03d}",
            "wildtype_ids": [f"q{index:05d}"],
            "family_clusters": [index],
            "partition": "evaluation",
            "selection_rank": index,
            "wildtypes": [{"wildtype_id": f"q{index:05d}", "wildtype": WILDTYPE,
                           "rows": payloads}],
        })
    return {"schema": fd.SCHEMA_QUEUE, "units": units_payload}


def freeze_queue(root, config, queue):
    queue["protocol"] = config
    fd.scientific_inputs(root, config, queue, create=True)
    fd._atomic_json(root / config["output"] / "queue.json", queue)


def spell(sequence: str) -> int:
    """A stand-in tokenizer: one target per residue except on a repeated pair."""

    return len(sequence) - sum(1 for left, right in zip(sequence, sequence[1:]) if left == right)


def synthetic_scores(queue, config, arm, nll_of):
    """One score document whose nulls are the declared ones, scored by a stand-in."""

    from src.capability.extensions.family_distribution import score_config

    document = {"schema": fd.SCHEMA_SCORES, "units": {},
                "scientific_inputs": fd.INPUT_SNAPSHOT, "dtype": "torch.float32",
                "checkpoint": {"arm": arm, "config": {"model_type": "synthetic_fixture"},
                               "tensor_files": [{"name": "fixture", "bytes": 0}]},
                "config": score_config(config, max_len=config["scoring"]["max_len"],
                                       partition="evaluation", arm=arm)}
    for unit in queue["units"]:
        for wildtype in unit["wildtypes"]:
            rows = fd.evaluation_rows(wildtype)
            shuffled = [column_null(rows, seed) for seed in config["null"]["column_shuffle_seeds"]]
            compositions = [composition_null(original, 11) for original in rows]
            document["units"].setdefault(unit["unit_id"], {"wildtypes": {}})["wildtypes"][
                wildtype["wildtype_id"]] = {
                "row_ids": [original.row_id for original in rows],
                "lengths": [len(original.native) for original in rows],
                "native_nll": nll_of(rows),
                "column_null_nll": [nll_of([fd.replace(original, native=value)
                                            for original, value in zip(rows, null)])
                                    for null in shuffled],
                "composition_null_nll": nll_of([fd.replace(original, native=value)
                                                for original, value in zip(rows, compositions)]),
                # A stand-in tokenizer that merges runs of identical residues, so
                # the native and null spellings are scored over different supports.
                "scored_tokens": {
                    "native": [spell(original.native) for original in rows],
                    "column_null": [[spell(value) for value in null] for null in shuffled],
                    "composition_null": [spell(original.native) for original in rows],
                },
            }
    return document


def test_analyse_end_to_end_reports_the_declared_families(tmp_path):
    import json

    config = dict(CONFIG)
    config["output"] = "results/tmp_family_distribution"
    config["statistics"] = dict(CONFIG["statistics"], bootstrap_draws=200)
    config["scoring"] = dict(CONFIG["scoring"], arms=["adjacent", "noise"])
    root = tmp_path
    output = root / config["output"]
    queue = synthetic_queue()
    (output / "scores" / "adjacent" ).mkdir(parents=True)
    (output / "scores" / "noise").mkdir(parents=True)
    freeze_queue(root, config, queue)

    rows = fd.evaluation_rows(queue["units"][0]["wildtypes"][0])
    true_markov = markov_model(rows, pseudocount=0.05)
    adjacent = synthetic_scores(queue, config, "adjacent",
                                lambda given: [-markov_score(original, true_markov) for original in given])
    rng = np.random.default_rng(3)
    noise = synthetic_scores(queue, config, "noise", lambda given: list(rng.normal(size=len(given))))
    (output / "scores" / "adjacent" / "evaluation.json").write_text(json.dumps(adjacent))
    (output / "scores" / "noise" / "evaluation.json").write_text(json.dumps(noise))

    summary = fd.analyse(root, config)
    evaluation = summary["partitions"]["evaluation"]
    assert evaluation["status"] == "complete"
    assert evaluation["units"] == 8 and evaluation["hypothesis_family"] == 2
    assert summary["arms"] == ["adjacent", "noise"]
    assert evaluation["controls"]["profile_zero_audit_max_abs_nats_per_residue"] < 1e-10
    assert evaluation["controls"]["profile_batch_additivity_max_abs"] < 1e-10
    points = {entry["arm"]: entry["point"] for entry in evaluation["inference"]["primary"]["hypotheses"]}
    # The adjacent-residue score improves when the column shuffle breaks adjacency;
    # pure noise averages out around zero on the same samples.
    assert points["adjacent"] > 0.0
    assert points["adjacent"] > points["noise"]
    assert evaluation["composition_inference"]["primary"]["family_size"] == 2
    # The scored-target diagnostic rides beside the contrast and is descriptive.
    diagnostic = evaluation["scored_token_diagnostic"]
    assert diagnostic["status"] == "descriptive"
    assert set(diagnostic["per_arm"]) == {"adjacent", "noise"}
    assert "no significance family" in diagnostic["role"]

    # Changing only the stored target counts moves the diagnostic and leaves the
    # main contrast untouched, which is the declared separation of the two.
    unit_id = queue["units"][0]["unit_id"]
    wildtype_id = queue["units"][0]["wildtypes"][0]["wildtype_id"]
    edited = json.loads(json.dumps(noise))
    counts = edited["units"][unit_id]["wildtypes"][wildtype_id]["scored_tokens"]
    counts["native"] = [value + 5 for value in counts["native"]]
    (output / "scores" / "noise" / "evaluation.json").write_text(json.dumps(edited))
    updated = fd.analyse(root, config)
    after = updated["partitions"]["evaluation"]
    assert (after["scored_token_diagnostic"]["per_arm"]["noise"]
            < diagnostic["per_arm"]["noise"])
    after_points = {entry["arm"]: entry["point"]
                    for entry in after["inference"]["primary"]["hypotheses"]}
    assert after_points["noise"] == pytest.approx(points["noise"])
    assert after_points["adjacent"] == pytest.approx(points["adjacent"])

    # An unscored wild type is refused, not imputed.
    broken = json.loads(json.dumps(adjacent))
    del broken["units"][queue["units"][0]["unit_id"]]
    (output / "scores" / "adjacent" / "evaluation.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="unscored"):
        fd.analyse(root, config)


def test_analyse_is_descriptive_below_the_inference_floor(tmp_path):
    import json

    config = dict(CONFIG)
    config["output"] = "results/tmp_family_small"
    config["scoring"] = dict(CONFIG["scoring"], arms=["noise"])
    root = tmp_path
    output = root / config["output"]
    queue = synthetic_queue(units=6)
    (output / "scores" / "noise").mkdir(parents=True)
    freeze_queue(root, config, queue)
    rng = np.random.default_rng(5)
    document = synthetic_scores(queue, config, "noise", lambda given: list(rng.normal(size=len(given))))
    (output / "scores" / "noise" / "evaluation.json").write_text(json.dumps(document))
    summary = fd.analyse(root, config)
    evaluation = summary["partitions"]["evaluation"]
    assert evaluation["status"] == "descriptive_only"
    assert evaluation["unit_floor"]["degenerate"] is True
    assert "inference" not in evaluation


def test_analyse_refuses_to_narrow_the_declared_family_silently(tmp_path):
    import json

    config = dict(CONFIG)
    config["output"] = "results/tmp_family_roster"
    config["statistics"] = dict(CONFIG["statistics"], bootstrap_draws=200)
    config["scoring"] = dict(CONFIG["scoring"], arms=["noise", "adjacent"])
    root = tmp_path
    output = root / config["output"]
    queue = synthetic_queue()
    (output / "scores" / "noise").mkdir(parents=True)
    freeze_queue(root, config, queue)
    rng = np.random.default_rng(11)
    document = synthetic_scores(queue, config, "noise", lambda given: list(rng.normal(size=len(given))))
    (output / "scores" / "noise" / "evaluation.json").write_text(json.dumps(document))
    # One scored column is not the declared two-column family, and a narrower
    # family would be a different, undeclared one.
    summary = fd.analyse(root, config)
    evaluation = summary["partitions"]["evaluation"]
    assert evaluation["status"] == "descriptive_only"
    assert "incomplete" in evaluation["reason"]
    assert evaluation["roster"]["missing"] == ["adjacent"]
    assert "inference" not in evaluation


def test_declared_roster_and_pilot_are_the_retained_ones():
    from src.capability.interactions.pairwise_epistasis import ARM_DTYPE, ROSTER

    assert sorted(CONFIG["scoring"]["arms"]) == sorted(ROSTER)
    assert len(CONFIG["scoring"]["arms"]) == 33
    assert set(CONFIG["scoring"]["lineage_arms"]) <= set(ROSTER)
    pilot = CONFIG["queue"]["pilot_models"]
    assert pilot == ["gpt2", "protgpt2", "progen2-medium", "prollama"]
    assert set(pilot) <= set(ROSTER)
    assert ARM_DTYPE.get("progen3-112m") == "bfloat16"
    assert ARM_DTYPE.get("protgpt2", "float32") == "float32"


def test_arm_partition_declaration_is_enforced():
    assert fd.validate_arm_partition("protgpt2", "pilot", CONFIG) is None
    assert fd.validate_arm_partition("prollama", "evaluation", CONFIG) is None
    with pytest.raises(ValueError, match="declared pilot models"):
        fd.validate_arm_partition("gpt2-large", "pilot", CONFIG)
    with pytest.raises(ValueError, match="outside the declared"):
        fd.validate_arm_partition("zymctrl", "evaluation", CONFIG)
    with pytest.raises(ValueError, match="unknown partition"):
        fd.validate_arm_partition("gpt2", "heldout", CONFIG)


def test_containment_evidence_order_is_stable():
    from src.capability.extensions.family_distribution import _containment

    first = "ACDEFGHIKLMNPQRSTVWY" * 2 + "AC"
    second = "ACDEFGHIKLMNPQRSTVWY" * 2 + "AD"
    container = "MM" + second + "RR" + first + "TT"
    # Set iteration order over equal-length strings varies between processes, so
    # the rule fixes it: shorter rows first, then the string itself.
    assert _containment([second, first], [container]) == (first, container)


def test_analyse_reads_the_pilot_alone_and_reports_the_rest_as_unscored(tmp_path):
    import json

    config = dict(CONFIG)
    config["output"] = "results/tmp_family_pilot_only"
    config["statistics"] = dict(CONFIG["statistics"], bootstrap_draws=200)
    config["scoring"] = dict(CONFIG["scoring"], arms=["noise"])
    root = tmp_path
    output = root / config["output"]
    queue = synthetic_queue(units=8)
    for index, unit in enumerate(queue["units"]):
        unit["partition"] = "pilot" if index < 4 else "evaluation"
    (output / "scores" / "noise").mkdir(parents=True)
    freeze_queue(root, config, queue)
    for partition in ("pilot",):
        document = synthetic_scores(queue, config, "noise",
                                    lambda given: list(np.random.default_rng(7).normal(size=len(given))))
        document["config"] = fd.score_config(config, max_len=config["scoring"]["max_len"],
                                             partition=partition, arm="noise")
        (output / "scores" / "noise" / f"{partition}.json").write_text(json.dumps(document))
    summary = fd.analyse(root, config)
    assert summary["partitions"]["pilot"]["status"] == "descriptive_only"
    assert "evaluation" in summary["unscored_partitions"]
    assert "no score document" in summary["unscored_partitions"]["evaluation"]


# ------------------------------------------------------ repaired boundaries


def test_low_complexity_containment_and_shared_kmer_false_positive():
    groups, evidence = duplicate_fragment_groups({
        "a": ["A" * 40], "b": ["C" + "A" * 40 + "D"],
        "c": ["A" * 12 + "C" * 28],
    }, config=CONFIG)
    assert groups == [["a", "b"], ["c"]]
    assert len(evidence) == 1 and evidence[0]["contained_length"] == 40


def scored_fixture(tmp_path, monkeypatch):
    """Real score/resume/analysis flow with no weights and a deterministic CPU score."""
    from copy import deepcopy
    from types import SimpleNamespace
    from torch.nn import Linear

    config = deepcopy(CONFIG)
    config["output"] = "run"
    config["scoring"]["arms"] = ["gpt2"]
    config["statistics"]["bootstrap_draws"] = 100
    queue = synthetic_queue()
    freeze_queue(tmp_path, config, queue)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    raw = {"model_type": "gpt2", "n_layer": 12, "n_embd": 768, "vocab_size": 20}
    fd._atomic_json(checkpoint / "config.json", raw)
    (checkpoint / "model.bin").write_bytes(b"synthetic fixture, not model weights")
    model: Any = Linear(1, 1)
    model.config = SimpleNamespace(to_dict=lambda: dict(raw))
    handle = SimpleNamespace(name="gpt2", spec=SimpleNamespace(path=checkpoint), model=model)
    loads, calls = [], []

    def load(name, *, device):
        loads.append(device)
        return handle

    def score(handle, sequences, **kwargs):
        calls.append(list(sequences))
        return ([float(sum(left != right for left, right in zip(value, value[1:])))
                 for value in sequences], [len(value) for value in sequences])

    monkeypatch.setattr(fd, "load_causal", load)
    monkeypatch.setattr(fd, "_nll_and_tokens", score)
    args: dict[str, Any] = dict(arm="gpt2", partition="evaluation", device="cpu", batch_size=1, max_len=1024)
    document = fd.score_partition(tmp_path, config, queue, **args)
    return config, queue, document, args, handle, loads, calls


def test_mixed_queue_scores_only_heldout_representatives_end_to_end(tmp_path, monkeypatch):
    config, queue, document, args, handle, loads, calls = scored_fixture(tmp_path, monkeypatch)
    assert loads == [None, "cpu"]
    for index, unit in enumerate(queue["units"]):
        wildtype = unit["wildtypes"][0]
        heldout = fd.evaluation_rows(wildtype)
        assert 1 < len(heldout) <= 32
        assert len(heldout) < len(wildtype["rows"])
        stored = document["units"][unit["unit_id"]]["wildtypes"][wildtype["wildtype_id"]]
        assert stored["row_ids"] == [original.row_id for original in heldout]
        native, *rest = calls[6 * index:6 * (index + 1)]
        assert native == [original.native for original in heldout]
        assert rest[:4] == [column_null(heldout, seed) for seed in config["null"]["column_shuffle_seeds"]]
        for null in rest[:4]:
            verify_column_null(heldout, null)
        statistics = fd._wildtype_statistics(wildtype, config)
        training = [fd.NativeRow.of(payload) for payload in wildtype["rows"]
                    if payload["split"] == "train" and payload["representative"]]
        expected = profile_frequencies(training, pseudocount=0.05)
        np.testing.assert_allclose(statistics["profile_native"],
                                   [profile_score(original, expected) for original in heldout])
        expected_markov = markov_model(training, pseudocount=0.05)
        np.testing.assert_allclose(statistics["markov_native"],
                                   [markov_score(original, expected_markov) for original in heldout])
        assert abs(statistics["profile"]["nats_per_residue"]) < 1e-12
    summary = fd.analyse(tmp_path, config)
    assert summary["partitions"]["evaluation"]["status"] == "complete"
    assert summary["partitions"]["evaluation"]["controls"]["profile_zero_audit_max_abs_nats_per_residue"] < 1e-12
    assert fd.score_partition(tmp_path, config, queue, **args) == document
    assert loads == [None, "cpu", None]  # resume does not load weights
    # Both readers use the same loaded-identity/dtype gate.
    path = tmp_path / config["output"] / "scores/gpt2/evaluation.json"
    document["dtype"] = "torch.float16"
    fd._atomic_json(path, document)
    for read in (lambda: fd.score_partition(tmp_path, config, queue, **args),
                 lambda: fd.analyse(tmp_path, config)):
        with pytest.raises(ValueError, match="identity/dtype"):
            read()


@pytest.mark.parametrize("field", ["native", "query_index", "split", "seed", "missing_snapshot"])
def test_same_ids_and_lengths_do_not_authorize_changed_inputs(tmp_path, monkeypatch, field):
    config, queue, document, args, *_ = scored_fixture(tmp_path, monkeypatch)
    payload = queue["units"][0]["wildtypes"][0]["rows"][0]
    if field == "native":
        payload["native"] = "W" + payload["native"][1:]
    elif field == "query_index":
        payload["query_index"][0] = -1
    elif field == "split":
        payload["split"] = "train"
    elif field == "seed":
        config["null"]["column_shuffle_seeds"][0] += 100
    else:
        (tmp_path / config["output"] / fd.INPUT_SNAPSHOT).unlink()
    fd._atomic_json(tmp_path / config["output"] / "queue.json", queue)
    for read in (lambda: fd.score_partition(tmp_path, config, queue, **args),
                 lambda: fd.analyse(tmp_path, config),
                 lambda: fd.scientific_inputs(tmp_path, config, queue, create=True)):
        with pytest.raises(ValueError, match="scientific input"):
            read()


def test_checkpoint_mismatch_cannot_resume_or_load_silently(tmp_path, monkeypatch):
    from types import SimpleNamespace
    config, queue, document, args, handle, *_ = scored_fixture(tmp_path, monkeypatch)
    declared = fd.checkpoint_identity(handle)
    wrong = dict(declared["config"], n_layer=36, n_embd=1280)
    handle.model.config = SimpleNamespace(to_dict=lambda: wrong)
    with pytest.raises(ValueError, match="loaded model config"):
        fd.verify_loaded(handle, declared)
    fd._atomic_json(handle.spec.path / "config.json", wrong)
    with pytest.raises(ValueError, match="checkpoint identity differs"):
        fd.score_partition(tmp_path, config, queue, **args)


@pytest.mark.parametrize("arm", CONFIG["scoring"]["arms"])
def test_all_33_use_the_native_readout_loader_dispatch(monkeypatch, arm):
    from types import SimpleNamespace
    from src.capability.interactions.pairwise_epistasis import ARM_DTYPE
    from src.capability.models import text_aa_fitness as text
    from src.capability.readouts.readout_extraction import text_readout_names, text_boundary
    from src.capability.core.arms import TEXT_MODEL_BASE, TEXT_MODEL_ROOT
    stage = fd.native_stage()
    seen = []

    def native(name, **kwargs):
        seen.append((name, kwargs))
        assert name in stage.ch.ARMS and name not in stage.ch.EXCLUDED_ARMS
        stage.ch.packing_of(name)  # real declaration resolves, not a generic causal handle
        return SimpleNamespace(name=name)

    def literal(path, **kwargs):
        seen.append((path, kwargs))
        return SimpleNamespace(model=object(), tokenizer=object())

    monkeypatch.setattr(stage, "load_scorable_arm", native)
    monkeypatch.setattr(fd, "native_stage", lambda: stage)
    monkeypatch.setattr(text, "load_text_aa_scorer", literal)
    loaded = fd.load_causal(arm, device="cpu")
    assert loaded.name == arm and len(seen) == 1
    if arm in text_readout_names():
        assert text_boundary(loaded) is not None
        if arm == "gpt2":
            assert loaded.spec.path == TEXT_MODEL_BASE / "gpt2"
        if arm == "gpt2-large":
            assert loaded.spec.path == TEXT_MODEL_ROOT
        if arm == "qwen2.5-0.5b-instruct":
            assert loaded.spec.path == TEXT_MODEL_BASE / "Qwen2.5-0.5B-Instruct"
    else:
        assert seen[0] == (arm, {"device": "cpu", "dtype": ARM_DTYPE.get(arm, "float32")})


def test_native_likelihood_uses_scored_span_without_hidden_hooks(monkeypatch):
    from types import SimpleNamespace
    import torch
    from src.capability.readouts import readout_extraction as native
    stage = fd.native_stage()
    monkeypatch.setattr(fd, "native_stage", lambda: stage)
    # Actual native text packer; one marker and three targets. Pooling span is
    # irrelevant to this path and no representation block is accessed.
    arm = SimpleNamespace(name="gpt2", device="cpu",
                          serving_provenance={"text_aa_boundary": SimpleNamespace(conditioning_id=0)},
                          tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=0))
    monkeypatch.setattr(native, "text_ids", lambda handle, value: [0, 1, 2, 3])
    logits = torch.zeros(1, 4, 7)
    logits[0, 0, 1] = 2
    logits[0, 1, 2] = 3
    logits[0, 2, 3] = 4
    arm.model = lambda **kwargs: SimpleNamespace(logits=logits)
    values, tokens = fd._nll_and_tokens(arm, ["ACD"], batch_size=1, max_len=4)
    expected = stage._target_nll(logits, torch.tensor([[0, 1, 2, 3]]), 1, 4)
    assert values == [expected["nll_sum"]] and tokens == [3]
    with pytest.raises(ValueError, match="budget"):
        fd._nll_and_tokens(arm, ["ACD"], batch_size=1, max_len=3)
    with pytest.raises(ValueError, match="batch_size=1"):
        fd._nll_and_tokens(arm, ["ACD"], batch_size=2, max_len=4)


def test_bootstrap_bands_and_pointwise_use_identical_family_means():
    matrix = np.column_stack((np.arange(8.0), np.arange(8.0)[::-1] ** 2, np.full(8, 2.0)))
    result = fd._family_inference(matrix, ["a", "b", "constant"], draws=113, seed=7)
    counts = np.random.default_rng(7).multinomial(8, np.full(8, 1 / 8), size=113)
    means = counts @ matrix / 8
    np.testing.assert_allclose(result["auxiliary_pointwise_95"]["interval"],
                               np.quantile(means, [.025, .975], axis=0).T)
    point = matrix.mean(axis=0)
    se = matrix.std(axis=0, ddof=1) / np.sqrt(8)
    critical = np.quantile(np.max(np.abs((means[:, :2] - point[:2]) / se[:2]), axis=1), .95)
    assert result["primary"]["critical_value"] == pytest.approx(critical)
    assert result["degenerate_arms"] == ["constant"]
    assert result["primary"]["hypotheses"][2]["status"] == "empirical_bootstrap_degenerate"
    assert "not certain capability or equivalence" in result["degeneracy_note"]
