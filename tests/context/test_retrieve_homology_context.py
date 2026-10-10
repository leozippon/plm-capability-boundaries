"""Invariants of the retrieval stage: reconstruction, re-banding and refusals.

The two things this stage must not get wrong are searching a protein that was
never measured, and reporting a band migration that is an artefact of a second
banding convention rather than of the corpus change.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/capability/context/retrieve_homology_context.py"
spec = importlib.util.spec_from_file_location("retrieve_homology_context", PATH)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)

from src.capability.context.homology import Hit  # noqa: E402


def hit(query, *, nident, qlen, qend=None, bitscore=100.0, subject="s"):
    return Hit(
        query=query,
        subject=subject,
        pident=100.0 * nident / qlen,
        length=qlen,
        nident=nident,
        qstart=1,
        qend=qlen if qend is None else qend,
        qlen=qlen,
        slen=qlen,
        evalue=1e-10,
        bitscore=bitscore,
        qseq_gapped="A" * qlen,
        sseq_gapped="A" * qlen,
    )


def background(name, group, wildtype, *, band, stratum):
    """A cohort background whose variants differ from it at one declared site."""
    variants = []
    for position in range(1, min(len(wildtype), 9) + 1):
        mutated = list(wildtype)
        mutated[position - 1] = "W" if wildtype[position - 1] != "W" else "A"
        variants.append(
            {
                "mutant": mutated[position - 1],
                "position": position,
                "sequence": "".join(mutated),
                "state": 1,
                "rows": 1,
                "target": -0.5,
            }
        )
    return {
        "name": name,
        "group": group,
        "band": band,
        "stratum": stratum,
        "length": len(wildtype),
        "library": "dms7",
        "admitted_variants": len(variants),
        "variants": variants,
    }


def gate_args(tmp_path, payload, *, limit=0):
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(payload))
    return argparse.Namespace(
        no_remote_gate=False, remote_gate_cohort=path, limit=limit
    )


WILDTYPE = "ACDEFGHIKLMNPQRST"


def test_backgrounds_are_reconstructed_and_verified_against_their_own_variants(tmp_path):
    payload = {
        "backgrounds": [
            background("bg1", "g1", WILDTYPE, band="id30_to_70_remote_homology", stratum="remote")
        ]
    }
    panel, support = stage.load_remote_gate(gate_args(tmp_path, payload))
    assert len(panel) == 1
    assert panel[0]["wildtype"] == WILDTYPE
    assert support["backgrounds"] == 1 and support["groups"] == 1
    assert support["prior_stratum_counts"] == {"remote": 1}
    assert "verified variant by variant" in support["reconstruction"]


def test_a_background_whose_variants_do_not_reconstruct_stops_the_run(tmp_path):
    payload = {
        "backgrounds": [
            background("bg1", "g1", WILDTYPE, band="id30_to_70_remote_homology", stratum="remote")
        ]
    }
    # A second substitution smuggled into one variant: the consensus no longer
    # explains it, and searching it would search a protein nothing measured.
    corrupted = list(payload["backgrounds"][0]["variants"][0]["sequence"])
    corrupted[-1] = "W" if corrupted[-1] != "W" else "A"
    payload["backgrounds"][0]["variants"][0]["sequence"] = "".join(corrupted)
    with pytest.raises(SystemExit, match="does not reconstruct"):
        stage.load_remote_gate(gate_args(tmp_path, payload))


def test_disabling_the_gate_panel_is_recorded_rather_than_silent(tmp_path):
    args = gate_args(tmp_path, {"backgrounds": []})
    args.no_remote_gate = True
    panel, support = stage.load_remote_gate(args)
    assert panel == []
    assert support == {"included": False, "reason": "disabled with --no-remote-gate"}


def test_reband_reports_prior_remote_groups_that_are_close_in_this_corpus():
    panel = [
        {
            "target_id": "bg1",
            "wildtype": WILDTYPE,
            "length": len(WILDTYPE),
            "group": "g1",
            "prior_band": "id30_to_70_remote_homology",
            "prior_stratum": "remote",
        },
        {
            "target_id": "bg2",
            "wildtype": WILDTYPE,
            "length": len(WILDTYPE),
            "group": "g2",
            "prior_band": "lt30_no_detectable_homology",
            "prior_stratum": "remote",
        },
    ]
    hits = {
        # 16 of 17 residues identical: close in this corpus, remote in the prior one.
        "bg1": [hit("bg1", nident=16, qlen=len(WILDTYPE))],
        # Still nothing detectable.
        "bg2": [],
    }
    record = stage.reband_remote_gate(panel, hits)
    assert record["prior_group_strata"] == {"remote": 2}
    assert record["new_group_strata"]["close"] == 1
    assert record["new_group_strata"]["remote"] == 1
    assert record["prior_remote_groups_now_close_or_mixed"] == ["g1"]
    assert record["prior_remote_groups_now_close_or_mixed_fraction"] == 0.5
    assert record["group_migration_counts"] == {"remote->close": 1, "remote->remote": 1}
    assert "only the corpus changed" in record["band_rule"]
    # A re-band does not re-read any increment, and says so.
    assert "does not re-read any" in record["does_not_license"]


def test_reband_reports_the_covered_band_beside_the_prior_gates_own_rule():
    panel = [
        {
            "target_id": "bg1",
            "wildtype": WILDTYPE,
            "length": len(WILDTYPE),
            "group": "g1",
            "prior_band": "lt30_no_detectable_homology",
            "prior_stratum": "remote",
        }
    ]
    # Essentially exact over a third of the target: close without a coverage
    # requirement, undetectable with one.
    shallow = hit("bg1", nident=6, qlen=len(WILDTYPE), qend=6)
    record = stage.reband_remote_gate(panel, {"bg1": [shallow]})
    row = record["backgrounds"][0]
    assert row["new_band"] == "id30_to_70_remote_homology"
    assert row["new_band_covered"] == "lt30_no_detectable_homology"
    assert record["new_group_strata"]["remote"] == 1
    assert record["new_group_strata_covered"]["remote"] == 1


def test_a_length_mismatch_between_the_alignment_and_the_background_stops_the_run():
    panel = [
        {
            "target_id": "bg1",
            "wildtype": WILDTYPE,
            "length": len(WILDTYPE),
            "group": "g1",
            "prior_band": "lt30_no_detectable_homology",
            "prior_stratum": "remote",
        }
    ]
    with pytest.raises(SystemExit, match="qlen"):
        stage.reband_remote_gate(panel, {"bg1": [hit("bg1", nident=5, qlen=len(WILDTYPE) + 1)]})


def test_assay_payload_refuses_a_mutation_string_the_wild_type_contradicts():
    cohort = {
        "assays": [
            {
                "assay": "a1",
                "wildtype_id": "q1",
                "wildtype": WILDTYPE,
                "cluster": 3,
                "mutants": ["A1G"],
                "sequences": ["G" + WILDTYPE[1:]],
                "measured": [0.5],
                "mutant_digest": "d",
                "profile_scores": [0.1],
            }
        ]
    }
    targets = [{"target_id": "q1"}]
    assert stage.assay_payload(cohort, targets)[0]["assay"] == "a1"
    # The declared wild-type residue no longer matches the wild type.
    cohort["assays"][0]["mutants"] = ["C1G"]
    with pytest.raises(SystemExit, match="the wild type carries"):
        stage.assay_payload(cohort, targets)
    # The mutation string and the stored sequence disagree.
    cohort["assays"][0]["mutants"] = ["A1G"]
    cohort["assays"][0]["sequences"] = ["W" + WILDTYPE[1:]]
    with pytest.raises(SystemExit, match="differs from the cohort"):
        stage.assay_payload(cohort, targets)


def test_the_variant_payload_carries_no_sequences_only_mutation_strings():
    cohort = {
        "assays": [
            {
                "assay": "a1",
                "wildtype_id": "q1",
                "wildtype": WILDTYPE,
                "cluster": 3,
                "mutants": ["A1G"],
                "sequences": ["G" + WILDTYPE[1:]],
                "measured": [0.5],
                "mutant_digest": "d",
                "profile_scores": [0.1],
            }
        ]
    }
    row = stage.assay_payload(cohort, [{"target_id": "q1"}])[0]
    assert "sequences" not in row
    assert row["mutants"] == ["A1G"] and row["cluster"] == 3


# ------------------------------------------- the copying join (EXP-E11, 2026-10-10)
#
# The identity-to-context rule reported zero for every product of every condition
# because the reference records were keyed by a composite ``target__bin__index``
# identifier while a product records the corpus accessions of the items its prompt
# actually contained. Those two identifier spaces never intersect, so the rule
# could not fire, and the matched-unrelated donors were not in the reference at
# all. Measured after the fix on the real artefact: a verbatim context item
# submitted as a product scores 100.0% identity at coverage 1.0, a 40%-mutated
# copy of one scores 57.6%, and a random sequence of the same length produces no
# alignment -- so the rule is capable of firing, which is what makes the zero on
# the 6144 real products a measurement rather than a join artefact.


def family_entry(target, subject, *, identity, coverage, in_family=True):
    return {
        "record": f"{target}__bin__0",
        "subject": subject,
        "target_id": target,
        "in_family": in_family,
        "identity_over_query": identity,
        "coverage": coverage,
        "bitscore": 100.0,
    }


def generated(target="q0", condition="close_homolog", subjects=("UniRef90_A", "UniRef90_B")):
    return {
        "attempt_id": f"arm|{target}|{condition}|0000",
        "arm": "arm",
        "target_id": target,
        "condition": condition,
        "sequence": "M" * 200,
        "context_subjects": list(subjects),
    }


def test_the_copying_rule_matches_the_accessions_the_prompt_actually_held():
    out = stage.product_annotation(
        generated(),
        index=0,
        family=[family_entry("q0", "UniRef90_B", identity=71.5, coverage=0.4)],
        corpus={},
    )
    assert out["context_alignment_identity"] == pytest.approx(71.5)
    # No coverage floor on the copying rule: a short high-identity run is exactly
    # what it is looking for, and the coverage is published beside it.
    assert out["context_alignment_coverage"] == pytest.approx(0.4)
    assert out["context_items_aligned"] == 1
    assert out["context_items"] == 2


def test_an_alignment_to_another_targets_context_is_not_this_products_copying():
    out = stage.product_annotation(
        generated(),
        index=0,
        family=[family_entry("q9", "UniRef90_B", identity=88.0, coverage=0.9)],
        corpus={},
    )
    assert out["context_alignment_identity"] is None
    assert out["prompt_family_recognised"] is False


def test_no_alignment_is_none_rather_than_zero_identity():
    out = stage.product_annotation(generated(), index=0, family=[], corpus={})
    assert out["context_alignment_identity"] is None
    assert out["context_items_aligned"] == 0
    # The copy verdict must abstain on an unevaluated rule, not pass it.
    from src.capability.context import homology_context as H

    verdict = H.copy_verdict(
        {"max_lcs_to_context": 3, "max_kmer_containment": 0.0},
        identity_percent=out["context_alignment_identity"],
    )
    assert verdict["rules"]["alignment_identity"] is None
    assert verdict["identity_rule_evaluated"] is False


def test_an_unrelated_donor_can_be_copied_without_becoming_family_recognition():
    out = stage.product_annotation(
        generated(condition="unrelated", subjects=("UniRef90_U",)),
        index=0,
        family=[family_entry("q0", "UniRef90_U", identity=100.0, coverage=1.0, in_family=False)],
        corpus={},
    )
    assert out["context_alignment_identity"] == pytest.approx(100.0)
    assert out["prompt_family_recognised"] is False
    assert out["prompt_family_identity"] == 0.0
    assert out["prompt_family_best_coverage"] == 0.0


def test_family_recognition_still_needs_the_coverage_floor():
    from src.capability.context import homology_context as H

    below = stage.product_annotation(
        generated(),
        index=0,
        family=[family_entry("q0", "UniRef90_Z", identity=95.0, coverage=H.FAMILY_COVERAGE_FLOOR / 2)],
        corpus={},
    )
    assert below["prompt_family_recognised"] is False
    assert below["prompt_family_best_coverage"] == pytest.approx(H.FAMILY_COVERAGE_FLOOR / 2)
    at_floor = stage.product_annotation(
        generated(),
        index=0,
        family=[family_entry("q0", "UniRef90_Z", identity=95.0, coverage=H.FAMILY_COVERAGE_FLOOR)],
        corpus={},
    )
    assert at_floor["prompt_family_recognised"] is True
