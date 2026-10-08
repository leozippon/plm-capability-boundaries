"""Invariants of the identity-binned homologous-context design (E09/E10/E11).

No checkpoint, no corpus and no GPU: everything here is a property of the frozen
declaration, the banding, the matched control, the copying endpoint or the
independence floor, and each of them has to hold before any number is produced.
"""
import pytest

from src.capability.context import homology_context as H
from src.capability.context.homology import Hit


def hit(**overrides):
    fields = dict(
        query="q00000",
        subject="UniRef90_X",
        pident=80.0,
        length=20,
        nident=16,
        qstart=1,
        qend=20,
        qlen=20,
        slen=20,
        evalue=1e-20,
        bitscore=100.0,
        qseq_gapped="A" * 20,
        sseq_gapped="A" * 20,
    )
    fields.update(overrides)
    return Hit(**fields)


def tokens(sequence):
    """One token per residue: the residue-level arms' own tokenisation."""
    return len(sequence)


# ------------------------------------------------------------------ the bins


def test_identity_bins_are_half_open_and_partition_the_range():
    assert H.assign_identity_bin(100.0) == "id_90_100"
    assert H.assign_identity_bin(90.0) == "id_90_100"
    assert H.assign_identity_bin(89.99999) == "id_70_90"
    assert H.assign_identity_bin(70.0) == "id_70_90"
    assert H.assign_identity_bin(69.99999) == "id_50_70"
    assert H.assign_identity_bin(50.0) == "id_50_70"
    assert H.assign_identity_bin(30.0) == "id_30_50"
    assert H.assign_identity_bin(29.99999) == "id_lt_30"
    assert H.assign_identity_bin(0.0) == "id_lt_30"
    # Contiguous and without overlap, from 0 up to and including 100.
    edges = [(low, high) for _, low, high in H.IDENTITY_BINS][::-1]
    assert edges[0][0] == 0.0
    assert edges[-1][1] > 100.0
    for (_, high), (low, _) in zip(edges, edges[1:]):
        assert high == low


def test_identity_outside_the_range_is_refused():
    for value in (-0.1, 100.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            H.assign_identity_bin(value)


def test_coverage_floor_and_verbatim_target_keep_a_bin_honest():
    wildtype = "ACDEFGHIKLMNPQRSTVWY"
    hits = [
        hit(subject="good", nident=16, sseq_gapped="ACDEFGHIKLMNPQRSTVWW", bitscore=200.0),
        hit(subject="partial", qend=10, nident=10, bitscore=150.0),
        hit(subject="noncanonical", sseq_gapped="ACDEFGHIKLMNPQRSTVWX", bitscore=140.0),
        hit(subject="verbatim", nident=20, pident=100.0, sseq_gapped=wildtype, bitscore=300.0),
        hit(subject="duplicate", nident=16, sseq_gapped="ACDEFGHIKLMNPQRSTVWW", bitscore=120.0),
    ]
    bins, refusals = H.bin_candidates(hits, wildtype=wildtype)
    assert [row["subject"] for row in bins["id_70_90"]] == ["good"]
    assert refusals["coverage_below_floor"] == 1
    assert refusals["non_canonical_segment"] == 1
    assert refusals["verbatim_target"] == 1
    assert refusals["duplicate_segment"] == 1
    # The verbatim record is the declared ceiling, not a member of the top bin.
    assert bins["id_90_100"] == []
    copy = H.self_copy_candidate(hits, wildtype=wildtype)
    assert copy["subject"] == "verbatim" and copy["sequence"] == wildtype


def test_max_identity_ignores_hits_below_the_coverage_floor():
    covered = hit(subject="far", nident=8, bitscore=50.0)
    shallow = hit(subject="shallow", qend=6, nident=6, pident=100.0, bitscore=90.0)
    assert H.max_identity_over_query([covered, shallow]) == pytest.approx(40.0)
    assert H.max_identity_over_query([shallow]) == 0.0
    assert H.max_identity_over_query([]) == 0.0


def test_a_hit_without_its_alignment_cannot_become_a_context_item():
    with pytest.raises(ValueError):
        H.hit_segment(hit(sseq_gapped=None))


# ------------------------------------------------- the matched unrelated control


def test_donor_pool_screens_overlap_and_the_targets_own_hits():
    target = "ACDEFGHIKLMNPQRSTVWY"
    candidates = [
        {"subject": "own", "sequence": "MMMMMMMMMMMMMMMMMMMM"},
        {"subject": "overlapping", "sequence": "ACDEFGHIKLMNPQRSTVWY"},
        {"subject": "clean", "sequence": "MKVLAAGIVGLNLQWRTYSD"},
        {"subject": "clean_again", "sequence": "MKVLAAGIVGLNLQWRTYSD"},
        {"subject": "too_short", "sequence": "MKVLAAGIVGL"},
        {"subject": "too_long", "sequence": "MKVLAAGIVGLNLQWRTYSDMKVLAAGIVGLNLQ"},
    ]
    pool = H.donor_pool(target, candidates=candidates, excluded_subjects={"own"})
    subjects = [row["subject"] for row in pool]
    assert "too_short" not in subjects and "too_long" not in subjects, (
        "the length band is applied before the cap, or the control starves"
    )
    assert "own" not in subjects, "a donor from the target's own hit list is not unrelated"
    assert "overlapping" not in subjects, "a verbatim run of the target is not unrelated"
    assert subjects == ["clean"], "a repeated segment must not fill the pool twice"
    assert pool[0]["max_lcs_to_target"] < H.UNRELATED_MAX_LCS
    assert pool[0]["composition_distance"] > 0.0


def test_unrelated_control_is_length_matched_and_composition_ordered():
    target = "ACDEFGHIKLMNPQRSTVWY"
    homolog = {"subject": "h1", "sequence": "ACDEFGHIKLMNPQRSTVWW", "bitscore": 200.0, "identity": 80.0}
    donors = [
        # Same length as the target, composition far from it.
        {"subject": "far", "sequence": "MMMMMMMMMMMMMMMMMMMM", "composition_distance": 0.9},
        # Same length, composition close to it: the one the rule must pick.
        {"subject": "near", "sequence": "YWVTSRQPNMLKIHGFEDCA", "composition_distance": 0.0},
        # Composition identical but far too short for the window.
        {"subject": "short", "sequence": "ACDEF", "composition_distance": 0.0},
    ]
    plans, budget = H.plan_conditions(
        wildtype=target,
        bins={"id_70_90": [homolog]},
        donors=donors,
        self_copy=None,
        token_length=tokens,
        referent=tokens(target),
        room=25,
    )
    assert budget["item_count"] == 1, "one item of the window's width is all this room holds"
    chosen = plans[H.UNRELATED]
    assert chosen.status == "present"
    assert [item["subject"] for item in chosen.items] == ["near"]
    # The control and the homologue carry the same token budget: the contrast
    # varies content and not length.
    assert chosen.record()["context_tokens"] == plans["id_70_90"].record()["context_tokens"]


def test_item_count_is_a_property_of_the_target_not_of_a_bin():
    assert H.item_count(room=0, referent=20) == 0
    # The per-item window's upper bound, not the referent, is what has to fit.
    assert H.item_count(room=24, referent=20) == 0
    assert H.item_count(room=25, referent=20) == 1
    assert H.item_count(room=10_000, referent=20) == H.CONTEXT_ITEMS_MAX
    assert H.item_token_window(100) == (75, 125)
    assert H.item_token_window(10) == (8, 12), "the floor applies to short targets"
    assert H.total_token_window(referent=100, count=4) == (380, 420)
    assert H.total_token_window(referent=5, count=1) == (3, 7), "the floor applies here too"


def test_every_condition_of_one_target_carries_the_same_item_count():
    target = "ACDEFGHIKLMNPQRSTVWY"
    many = [
        {"subject": f"h{index}", "sequence": "ACDEFGHIKLMNPQRSTVW" + residue, "bitscore": 100.0 - index, "identity": 80.0}
        for index, residue in enumerate("WYFLM")
    ]
    donors = [
        {"subject": f"d{index}", "sequence": "YWVTSRQPNMLKIHGFEDC" + residue, "composition_distance": index / 10}
        for index, residue in enumerate("AYFLM")
    ]
    plans, budget = H.plan_conditions(
        wildtype=target,
        bins={"id_70_90": many},
        donors=donors,
        self_copy=None,
        token_length=tokens,
        referent=tokens(target),
        room=200,
    )
    assert budget["item_count"] == H.CONTEXT_ITEMS_MAX
    for condition in ("id_70_90", H.UNRELATED):
        assert len(plans[condition].items) == budget["item_count"]


# --------------------------------------------------------------- absence is data


def test_an_empty_bin_is_recorded_absent_and_the_target_is_kept():
    target = "ACDEFGHIKLMNPQRSTVWY"
    plans, _ = H.plan_conditions(
        wildtype=target,
        bins={"id_70_90": [{"subject": "h", "sequence": "ACDEFGHIKLMNPQRSTVWW", "bitscore": 1.0, "identity": 80.0}]},
        donors=[{"subject": "d", "sequence": "YWVTSRQPNMLKIHGFEDCA", "composition_distance": 0.0}],
        self_copy=None,
        token_length=tokens,
        referent=tokens(target),
        room=200,
    )
    # Every declared condition is present as a key, so no target can leave the
    # inventory by having an empty bin.
    assert set(plans) == set(H.CONDITIONS) | {H.CEILING_CONDITION}
    for name in ("id_90_100", "id_50_70", "id_30_50", "id_lt_30"):
        assert plans[name].status == "absent"
        assert plans[name].reason and plans[name].reason.startswith("0 of ")
        assert plans[name].record()["n_items"] == 0
    assert plans[H.SELF_COPY].status == "absent"
    assert "not verbatim in the corpus" in plans[H.SELF_COPY].reason
    assert plans[H.NO_CONTEXT].status == "present"


def test_a_target_too_long_for_one_item_loses_every_context_condition():
    target = "ACDEFGHIKLMNPQRSTVWY"
    plans, budget = H.plan_conditions(
        wildtype=target,
        bins={"id_70_90": [{"subject": "h", "sequence": target[:-1] + "W", "bitscore": 1.0, "identity": 80.0}]},
        donors=[],
        self_copy=None,
        token_length=tokens,
        referent=tokens(target),
        room=3,
    )
    assert budget["item_count"] == 0
    assert plans[H.NO_CONTEXT].status == "present"
    assert all(plans[name].status == "absent" for name in H.CURVE_ORDER)
    assert "room for 0 items" in plans[H.UNRELATED].reason


# ------------------------------------------------------------------ the copying


def test_copy_statistics_find_a_verbatim_run_and_a_reshuffled_duplicate():
    context = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    copied = "MKV" + context[:H.COPY_LCS_RESIDUES] + "GGG"
    statistics = H.copy_statistics(copied, [context])
    assert statistics["max_lcs_to_context"] == H.COPY_LCS_RESIDUES
    assert H.copy_verdict(statistics)["rules"]["long_verbatim_run"] is True
    assert H.copy_verdict(statistics)["is_copy"] is True

    # A product stitched from the context's own k-mers has no single long run but
    # is still not a new protein.
    stitched = context[20:] + context[:20]
    stitched_statistics = H.copy_statistics(stitched, [context])
    assert stitched_statistics["max_kmer_containment"] >= H.COPY_KMER_CONTAINMENT
    assert H.copy_verdict(stitched_statistics)["is_copy"] is True


def test_a_clean_product_is_not_a_copy_and_the_identity_rule_abstains():
    statistics = H.copy_statistics("MKVLAAGIVGLNLQWRTYSDKPHE", ["ACDEFGHIKLMNPQRSTVWY"])
    verdict = H.copy_verdict(statistics)
    assert verdict["is_copy"] is False
    assert verdict["rules"]["alignment_identity"] is None
    assert verdict["identity_rule_evaluated"] is False
    # Once an aligner has spoken, the same product can still be a copy.
    aligned = H.copy_verdict(statistics, identity_percent=H.COPY_IDENTITY_PERCENT)
    assert aligned["is_copy"] is True
    assert aligned["fired"] == ["alignment_identity"]


def test_kmer_containment_is_directional_and_bounded():
    assert H.kmer_containment("A" * 10, "A" * 100) == 1.0
    assert H.kmer_containment("A" * 100, "A" * 10) == 1.0
    assert H.kmer_containment("ACDEF", "ACDEFGHIK") == 0.0, "shorter than k has no k-mers"
    assert 0.0 <= H.kmer_containment("ACDEFGHIKLMN", "MNPQRSTVWYAC") <= 1.0


def test_an_empty_product_is_measurable_rather_than_an_error():
    statistics = H.copy_statistics("", ["ACDEFGHIK"])
    assert statistics["max_lcs_to_context"] == 0
    assert H.copy_verdict(statistics)["is_copy"] is False


# ------------------------------------------------- the floor, power and the curve


def test_bin_admission_enforces_the_independence_floor():
    support = {name: H.GROUP_FLOOR for name in H.BIN_NAMES}
    support["id_lt_30"] = H.GROUP_FLOOR - 1
    admitted = H.admitted_bins(support)
    assert "id_lt_30" not in admitted
    assert set(admitted) == set(H.BIN_NAMES) - {"id_lt_30"}
    assert admitted == tuple(name for name in H.CURVE_ORDER if name in admitted)
    with pytest.raises(ValueError):
        H.admitted_bins({"id_90_100": 10})


def test_balanced_panel_requires_every_admitted_bin_and_both_controls():
    full = {name: "present" for name in H.CONDITIONS}
    thin = dict(full, id_lt_30="absent")
    no_control = dict(full, unrelated="absent")
    statuses = {"t1": full, "t2": thin, "t3": no_control}
    assert H.balanced_targets(statuses) == ["t1"]
    assert H.balanced_targets(statuses, bins=["id_90_100", "id_70_90"]) == ["t1", "t2"]
    with pytest.raises(ValueError):
        H.balanced_targets(statuses, bins=["not_a_condition"])


def test_thin_support_is_unresolved_even_when_the_interval_excludes_zero():
    record = H.power_record(
        point=0.1, standard_error=0.01, groups=H.GROUP_FLOOR - 1, interval=(0.05, 0.15), critical=2.0
    )
    assert record["resolution"] == "unresolved_thin_support"
    assert record["direction"] == "above_zero"
    resolved = H.power_record(
        point=0.1, standard_error=0.01, groups=H.GROUP_FLOOR, interval=(0.05, 0.15), critical=2.0
    )
    assert resolved["resolution"] == "resolved"


def test_required_groups_separates_no_effect_from_no_power():
    assert H.required_groups(0.01, 0.01, groups=100, critical=2.0) == 400
    assert H.required_groups(0.04, 0.01, groups=100, critical=2.0) == 25
    assert H.required_groups(0.0, 0.01, groups=100, critical=2.0) is None
    assert H.required_groups(0.01, 0.0, groups=100, critical=2.0) is None
    assert H.required_groups(float("nan"), 0.01, groups=100, critical=2.0) is None


def test_vanishing_point_scans_downward_and_names_why_it_stopped():
    def record(name, resolved, above=True):
        return H.bin_power_record(
            bin_name=name,
            point=0.1 if above else -0.1,
            standard_error=0.01,
            groups=20,
            interval=(0.05, 0.15) if above else (-0.15, -0.05),
            critical=2.0,
        ) | ({} if resolved else {"resolution": "unresolved_interval_crosses_zero", "direction": None})

    curve = [
        record("id_90_100", True),
        record("id_70_90", True),
        record("id_50_70", False),
        record("id_30_50", False),
    ]
    reading = H.vanishing_point(curve)
    assert reading["resolved_positive_bins"] == ["id_90_100", "id_70_90"]
    assert reading["lowest_resolved_positive_bin"] == "id_70_90"
    assert reading["vanishing_identity_percent"] == 70.0
    assert reading["scan_stopped_at_bin"] == "id_50_70"
    assert reading["scan_stopped_because"] == "unresolved_interval_crosses_zero"
    assert reading["no_bin_resolved_positive"] is False

    none_resolved = H.vanishing_point([record("id_90_100", False)])
    assert none_resolved["no_bin_resolved_positive"] is True
    assert none_resolved["vanishing_identity_percent"] is None

    # A resolved *negative* bin does not extend the curve, and must not be
    # reported as an unresolved one either: the gain reversed, which is a finding.
    negative = H.vanishing_point([record("id_90_100", True, above=False)])
    assert negative["no_bin_resolved_positive"] is True
    assert negative["scan_stopped_because"] == "resolved below_zero"
    assert negative["resolved_negative_bins"] == ["id_90_100"]

    exhausted = H.vanishing_point([record("id_90_100", True), record("id_70_90", True)])
    assert exhausted["scan_stopped_at_bin"] is None
    assert exhausted["scan_stopped_because"] == "every scanned bin resolved above zero"


# -------------------------------------------------------------- the declaration


def test_the_declaration_travels_with_every_artefact_and_is_refused_if_it_differs():
    digest = H.declaration_digest()
    assert len(digest) == 64
    H.require_declaration({"declaration_sha256": digest})
    with pytest.raises(ValueError):
        H.require_declaration({"declaration_sha256": "0" * 64})
    with pytest.raises(ValueError):
        H.require_declaration({})


def test_the_declaration_records_what_the_curve_is_read_on():
    declared = H.declaration()
    assert declared["primary_referent"] == H.UNRELATED
    assert declared["secondary_referent"] == H.NO_CONTEXT
    assert declared["inference"]["primary_panel"] == H.PRIMARY_PANEL
    assert declared["inference"]["group_floor"] == 8
    assert declared["conditions"][0] == H.NO_CONTEXT
    assert declared["curve_order"] == ["id_lt_30", "id_30_50", "id_50_70", "id_70_90", "id_90_100"]
    assert declared["search"]["coverage_floor"] == H.COVERAGE_FLOOR
    assert declared["search"]["max_target_seqs"] == H.MAX_TARGET_SEQS


# ------------------------------------------------- the structural comparison (E11)


def attempt(identifier, condition, residues):
    return {"attempt_id": identifier, "condition": condition, "residues": residues}


def test_structure_length_bands_are_the_generation_lanes_own():
    from src.capability.generation import generation_evidence as ge

    assert H.STRUCTURE_LENGTH_BANDS == tuple(
        tuple(band) for band in ge.POLICY["length_strata"]
    ), "the structural bands must not drift from the generation lane's strata"
    assert H.structure_length_band(16) == "len_16_128"
    assert H.structure_length_band(128) == "len_16_128"
    assert H.structure_length_band(129) == "len_129_256"
    assert H.structure_length_band(15) is None
    assert H.structure_length_band(2048) is None


def test_structure_selection_draws_an_equal_count_per_condition_in_each_band():
    conditions = ["unrelated", "close_homolog"]
    attempts = []
    # Band one: five unrelated products, two close ones.
    attempts += [attempt(f"u{i}", "unrelated", 100) for i in range(5)]
    attempts += [attempt(f"c{i}", "close_homolog", 100) for i in range(2)]
    # Band two: only the close condition reaches it, so the band is unusable.
    attempts += [attempt(f"c{i}", "close_homolog", 300) for i in range(5, 9)]
    selected, record = H.select_structure_products(attempts, conditions=conditions)
    first = next(row for row in record["bands"] if row["band"] == "len_16_128")
    assert first["drawn_per_condition"] == 2, "the per-band minimum across conditions"
    assert first["status"] == "used"
    second = next(row for row in record["bands"] if row["band"] == "len_257_512")
    assert second["drawn_per_condition"] == 0
    assert second["status"] == "unused"
    assert "supplies no product" in second["reason"]
    assert record["selected_per_condition"] == {"unrelated": 2, "close_homolog": 2}
    assert len(selected) == 4


def test_structure_selection_excludes_products_too_short_to_measure():
    conditions = ["unrelated", "close_homolog"]
    attempts = [
        attempt("short_u", "unrelated", H.MIN_PRODUCT_RESIDUES - 1),
        attempt("short_c", "close_homolog", H.MIN_PRODUCT_RESIDUES - 1),
        attempt("long_u", "unrelated", 120),
        attempt("long_c", "close_homolog", 120),
    ]
    selected, record = H.select_structure_products(attempts, conditions=conditions)
    assert selected == {"long_u", "long_c"}
    assert record["min_product_residues"] == H.MIN_PRODUCT_RESIDUES


def test_structure_selection_is_capped_and_deterministic():
    conditions = ["unrelated", "close_homolog"]
    attempts = [attempt(f"u{i}", "unrelated", 100) for i in range(50)]
    attempts += [attempt(f"c{i}", "close_homolog", 100) for i in range(50)]
    first, record = H.select_structure_products(attempts, conditions=conditions, samples_per_band=4)
    second, _ = H.select_structure_products(attempts, conditions=conditions, samples_per_band=4)
    assert first == second, "the same seed must select the same products"
    assert record["selected_per_condition"] == {"unrelated": 4, "close_homolog": 4}
    third, _ = H.select_structure_products(
        attempts, conditions=conditions, samples_per_band=4, seed=H.DRAW_SEED + 1
    )
    assert third != first, "a different declared seed must give a different draw"


def test_a_scope_only_checks_the_sections_its_input_depends_on():
    digests = H.declaration_digests()
    assert set(digests) == {
        "search_sha256",
        "retrieval_sha256",
        "context_sha256",
        "declaration_sha256",
    }
    # A retrieval artefact passes the retrieval scope even when a section it never
    # read has changed, so a corpus scan is not invalidated by an unrelated edit.
    stale = H.declaration()
    stale["structure"] = {"samples_per_band": 999}
    stale["inference"] = dict(stale["inference"], bootstrap_draws=1)
    record = {"declaration": stale}
    H.require_declaration(record, scope="search")
    H.require_declaration(record, scope="retrieval")
    with pytest.raises(ValueError, match="context declaration"):
        H.require_declaration(record, scope="context")
    with pytest.raises(ValueError, match="full declaration"):
        H.require_declaration(record, scope="full")
    # A screen governs the binned artefact, not the hit table a search produced.
    screened = H.declaration()
    screened["screens"] = dict(screened["screens"], donor_pool_cap=1)
    H.require_declaration({"declaration": screened}, scope="search")
    with pytest.raises(ValueError, match="retrieval declaration"):
        H.require_declaration({"declaration": screened}, scope="retrieval")
    # A changed bin edge is in every scope.
    moved = H.declaration()
    moved["identity_bins"] = [["id_90_100", 85.0, 100.000001]]
    with pytest.raises(ValueError, match="retrieval declaration"):
        H.require_declaration({"declaration": moved}, scope="retrieval")
    H.require_declaration({"declaration": moved}, scope="search")
    # A changed token budget is in the context scope and not in the retrieval one.
    budget = H.declaration()
    budget["budget"] = dict(budget["budget"], context_items_max=1)
    H.require_declaration({"declaration": budget}, scope="retrieval")
    with pytest.raises(ValueError, match="context declaration"):
        H.require_declaration({"declaration": budget}, scope="context")
    with pytest.raises(KeyError):
        H.require_declaration({"declaration": H.declaration()}, scope="nonsense")


def folded(condition, residues, plddt, *, copy=False, ptm=0.5):
    return {
        "condition": condition,
        "residues": residues,
        "plddt": plddt,
        "ptm": ptm,
        "copy_verdict": {"is_copy": copy},
    }


def test_structure_comparison_refuses_an_unbalanced_band():
    conditions = ["unrelated", "close_homolog"]
    products = [
        folded("unrelated", 100, 0.5),
        folded("unrelated", 110, 0.5),
        folded("close_homolog", 120, 0.9),
    ]
    out = H.structure_comparison(products, conditions=conditions)
    first = out["len_16_128"]
    assert first["status"] == "not comparable"
    assert first["equal_count_across_conditions"] is False
    assert "equal number" in first["reason"]
    # The numbers are still reported, with the length they were measured at.
    assert first["per_condition"]["close_homolog"]["mean_plddt"] == 0.9
    assert first["per_condition"]["unrelated"]["mean_residues"] == 105.0


def test_structure_comparison_is_within_a_band_and_excludes_copies_separately():
    conditions = ["unrelated", "close_homolog"]
    products = [
        folded("unrelated", 100, 0.40),
        folded("close_homolog", 100, 0.60),
        # A long product in another band must not lift the short band's mean.
        folded("unrelated", 300, 0.95),
        folded("close_homolog", 300, 0.95),
        # A copy folds like the real relative it copied; it is reported apart.
        folded("unrelated", 110, 0.20),
        folded("close_homolog", 110, 0.98, copy=True),
    ]
    out = H.structure_comparison(products, conditions=conditions)
    short = out["len_16_128"]
    assert short["status"] == "comparable"
    assert short["per_condition"]["unrelated"]["mean_plddt"] == pytest.approx(0.30)
    assert short["per_condition"]["close_homolog"]["mean_plddt"] == pytest.approx(0.79)
    # With the copy excluded the apparent advantage disappears.
    assert short["per_condition"]["close_homolog"]["non_copy"]["mean_plddt"] == pytest.approx(0.60)
    assert short["per_condition"]["close_homolog"]["non_copy"]["folded"] == 1
    assert short["per_condition"]["unrelated"]["non_copy"]["folded"] == 2
    long_band = out["len_257_512"]
    assert long_band["status"] == "comparable"
    assert long_band["per_condition"]["unrelated"]["mean_residues"] == 300.0
    # A band nothing reached is reported, not omitted.
    assert out["len_513_1024"]["status"] == "not comparable"
    assert out["len_513_1024"]["per_condition"]["unrelated"]["folded"] == 0


def test_the_declaration_names_an_evolutionary_profile_referent():
    """A context gain must be readable against a count model of homologous sequence.

    The phenotype programme on this same cohort saw apparent ranking gains of
    about +0.10 fall to +0.008 and go unresolved once an evolutionary-profile
    control entered, so the profile is a declared referent here rather than a
    discussion point.
    """
    declared = H.declaration()
    assert H.PROFILE_REFERENT in H.REFERENTS
    assert declared["referents"] == [H.UNRELATED, H.NO_CONTEXT, H.PROFILE_REFERENT]
    assert H.PROFILE_REFERENT in declared["condition_purpose"]
    # It is a referent, never a condition the decoder is scored under.
    assert H.PROFILE_REFERENT not in H.CONDITIONS
    assert any("not a nested increment" in line for line in declared["limitations"])


def test_the_curve_is_read_against_the_empty_context():
    """A contrast against the matched control alone cannot price a prefix itself.

    Reading only bin-minus-unrelated conflates "homologous context helps less
    than unrelated context" with "any prefix hurts". The empty context is
    therefore the reading referent, and the matched-unrelated condition's own
    contrast against it is what prices the general prefix cost.
    """
    declared = H.declaration()
    assert H.READING_REFERENT == H.NO_CONTEXT
    assert declared["inference"]["reading_referent"] == H.NO_CONTEXT
    assert "any prefix" in declared["inference"]["reading_referent_reason"]
    assert "READING_REFERENT" in declared["inference"]["vanishing_point_rule"]
    # The matched control stays declared as the content control it is.
    assert H.PRIMARY_REFERENT == H.UNRELATED
    # A reading rule must not invalidate a scored record: it is not in any
    # digest scope, so a stage can change how it reads without rescoring.
    for scope in ("search", "retrieval", "context"):
        assert "inference" not in (H.DIGEST_SCOPES[scope] or ())


def test_scoring_is_one_row_per_forward_with_a_structural_repeat_gate():
    """The endpoint is a difference of two scored states, so batch extent matters.

    At eight rows per forward the mutant-minus-wild differences moved by 1.5e-3
    and 4.6e-3 nats on the two larger ProGen2 rungs, against a 1e-3 check -- exact
    binary fractions, i.e. rounding, not a logic error. The answer is one row per
    forward, which makes the repeat gate structural, not a widened tolerance that
    would also admit a real defect of the same size.
    """
    declared = H.declaration()
    assert H.SCORING_ROWS_PER_FORWARD == 1
    assert H.REPEAT_TOLERANCE_NATS == 0.0
    inference = declared["inference"]
    assert inference["rows_per_forward"] == 1
    assert inference["repeat_tolerance_nats"] == 0.0
    assert inference["batch_extent_probe_rows"] == H.BATCH_EXTENT_PROBE_ROWS
    assert "difference of two scored states" in inference["numerics_reason"]
    assert any("rows-per-forward" in line for line in declared["limitations"])
    # A numerics protocol is a reading of the same inputs, not a different cohort,
    # so it must not invalidate a retrieval artefact.
    for scope in ("search", "retrieval", "context"):
        assert "inference" not in (H.DIGEST_SCOPES[scope] or ())
