"""E20 fold recapitulation: the conditions that must hold, and the refusals."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.capability.core.amino_acids import AA20
from src.capability.generation import fold_recapitulation as fr
from src.capability.generation import fold_recapitulation_inference as fri
from src.capability.generation import structure_evidence as se
from src.capability.ladder import fold_geometry as fg

WT = {"D1": "ACDEFGHIKL", "D2": "MNPQRSTVWY"}


def _rec(i, domain="D1", arm="N", seq="ACDEFGHIKL", **kw):
    row = {
        "schema_version": fr.SCHEMA_VERSION, "id": f"r{i}", "domain": domain, "model": "m", "arm": arm,
        "lambda": 0.0 if arm in ("N", "P", "H") else 1.0, "seed": i, "sequence": seq,
        "native_loglik": -10.0, "guidance_score": None if arm in ("N", "P", "H") else 1.0,
        "mean_step_entropy": 2.0, "attempt_budget": 5,
    }
    row.update(kw)
    return row


def _helix(n, shift=0.0):
    t = np.arange(n) * 100 * np.pi / 180
    return np.stack([2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * np.arange(n) + shift], axis=1)


def test_tm_score_self_is_one_and_length_mismatch_refused():
    x = _helix(40)
    assert fg.tm_score(x, x)["tm_score"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="one length"):
        fg.tm_score(x, _helix(41))


def test_amino_acid_orders_are_asserted():
    assert fr.AA != AA20 and sorted(fr.AA) == sorted(AA20)


def test_contract_rejects_missing_field_and_inconsistent_records():
    bad = _rec(1)
    del bad["seed"]
    with pytest.raises(fr.ContractError, match="missing"):
        fr.validate_records([bad], WT)
    with pytest.raises(fr.ContractError, match="lambda"):
        fr.validate_records([_rec(1, arm="C", **{"lambda": 0.0})], WT)
    with pytest.raises(fr.ContractError, match="guidance_score"):
        fr.validate_records([_rec(1, arm="C", guidance_score=None)], WT)
    with pytest.raises(fr.ContractError, match="within cell"):
        fr.validate_records([_rec(1, arm="C"), _rec(2, arm="C", attempt_budget=6)], WT)
    with pytest.raises(fr.ContractError, match="unique"):
        fr.validate_records([_rec(1), _rec(1)], WT)
    assert len(fr.validate_records([_rec(1, domain="D1.pdb")], WT)) == 1


def test_wrong_length_is_refused_not_skipped():
    with pytest.raises(fr.ContractError, match="refused, not skipped"):
        fr.validate_records([_rec(1), _rec(2, seq="ACDEFGHIK")], WT)


def test_controls_match_hand_values():
    assert fr.identity("AAAA", "AABB") == 50.0
    p, q = fr.composition(["AAAA"]), fr.composition(["AACC"])
    assert fr.composition_distance(p, q) == pytest.approx(0.5)
    # rows: identity over non-gap columns, row 3 is a short fragment (coverage < 0.5) and is ignored
    msa = ["AAAA", "AB-B", "A---"]
    assert fr.max_msa_identity("AAAA", msa) == 100.0
    assert fr.max_msa_identity("AACC", ["AAB-", "A---"]) == pytest.approx(200 / 3)
    d = fr.diversity(["AAAA", "AAAB", "CCCC"])
    assert d["mean_pairwise_identity"] == pytest.approx((75 + 0 + 0) / 3)
    assert d["near_duplicate_rate"] == 0.0
    assert fr.diversity(["AAAA", "AAAA", "CCCC"])["near_duplicate_rate"] == pytest.approx(2 / 3)


def _rows(values, n_dom=8, ceiling_copy=None):
    """values[arm][domain_index] -> MIN_ADMISSIBLE_PER_CELL identical rows each; optional copies above ceiling."""
    rows = []
    for arm, per_domain in values.items():
        for k in range(n_dom):
            rows += [{"model": "m", "domain": f"d{k}", "arm": arm, "tm_score": per_domain[k],
                      "plddt_mean": 50.0, "ptm": 0.5, "max_msa_identity": 40.0, "identity_to_wild_type": 10.0}
                     for _ in range(fr.MIN_ADMISSIBLE_PER_CELL)]
    return rows


ROLES = {f"d{k}": "evaluation" for k in range(12)}
NO_DOSE = {"lambda_fields": None, "lambda_coupling": None}
STRENGTHS = [{"model": "m", "arm": a, "frozen_strength": NO_DOSE} for a in fr.ARMS]


def test_factorial_contrasts_values_and_interval():
    v = {"N": [0.2] * 8, "C": [0.4] * 8, "M": [0.3] * 8, "F": [0.7] * 8}
    out = fri.analyse(_rows(v), roles=ROLES, role="evaluation", strengths=STRENGTHS)
    f = out["results"]["m|tm_score|unstratified"]["factorial"]
    assert f["C-N"]["mean_over_domains"] == pytest.approx(0.2)
    assert f["F-M"]["mean_over_domains"] == pytest.approx(0.4)
    assert f["main_couplings"]["mean_over_domains"] == pytest.approx(0.3)
    assert f["main_fields"]["mean_over_domains"] == pytest.approx(0.2)
    assert f["interaction"]["mean_over_domains"] == pytest.approx(0.2)
    assert f["interaction"]["interval"]["n_domains"] == 8
    assert out["refusals"] == []


def test_stratified_contrast_drops_sequences_above_ceiling():
    v = {"N": [0.2] * 8, "C": [0.4] * 8, "M": [0.2] * 8, "F": [0.4] * 8}
    rows = _rows(v)
    # in every domain a homolog copy in arm C scores perfectly
    rows += [{"model": "m", "domain": f"d{k}", "arm": "C", "tm_score": 1.0, "plddt_mean": 50.0,
              "ptm": 0.5, "max_msa_identity": 95.0, "identity_to_wild_type": 10.0} for k in range(8)]
    out = fri.analyse(rows, roles=ROLES, role="evaluation", strengths=STRENGTHS)["results"]
    un = out["m|tm_score|unstratified"]["factorial"]["C-N"]["mean_over_domains"]
    st = out["m|tm_score|novelty_restricted"]["factorial"]["C-N"]["mean_over_domains"]
    assert un == pytest.approx((25 * 0.4 + 1.0) / 26 - 0.2) and st == pytest.approx(0.2)


def test_partial_factorial_coverage_is_excluded_and_floor_refused():
    v = {"N": [0.2] * 8, "C": [0.4] * 8, "M": [0.3] * 8, "F": [0.7] * 8}
    rows = [r for r in _rows(v) if not (r["domain"] == "d7" and r["arm"] == "F")]
    out = fri.analyse(rows, roles=ROLES, role="evaluation", strengths=STRENGTHS)
    block = out["results"]["m|tm_score|unstratified"]
    assert block["factorial_domains_excluded_partial_coverage"] == ["d7"]
    for c in block["factorial"].values():  # same pairings for simple, main and interaction contrasts
        assert "d7" not in c["per_domain"] and c["n_domains"] == 7
        assert c["interval"] is None and "floor" in c["interval_refused"]
    assert out["refusals"]
    # C-N alone could use d7, but the factorial family does not mix domain sets
    assert block["pairwise"]["Cx-N"]["n_domains"] == 0 and not any("Cx-N" in r for r in out["refusals"])


def test_cross_domain_interval_below_eight_refused():
    with pytest.raises(ValueError, match="floor"):
        fri.cross_domain_interval({f"d{k}": 0.1 * k for k in range(7)}, seed=1)
    assert fri.cross_domain_interval({f"d{k}": 0.1 * k for k in range(8)}, seed=1)["n_domains"] == 8


def _pdb(trace):
    return "".join(
        f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C  \n"
        for i, (x, y, z) in enumerate(trace)
    )


def _tree(root, items, signature_extra="a"):
    """A minimal run_structure_evidence tree; items = {sequence: (trace, plddt, ptm)}."""
    contract = {"num_loops": 3, "extra": signature_extra}
    sig = se.contract_signature(contract)
    root.mkdir(parents=True, exist_ok=True)
    (root / "worker-000-of-001.json").write_text(json.dumps({"evaluation_signature": sig, "contract": contract}))
    for sequence, (trace, plddt, ptm) in items.items():
        sha = se.sequence_digest(sequence)
        d = root / "objects" / sha
        d.mkdir(parents=True)
        (d / "prediction.pdb").write_text(_pdb(trace))
        (d / "result.json").write_text(json.dumps({
            "evaluation_signature": sig, "sequence_sha256": sha, "status": "ok",
            "mean_ca_plddt": plddt, "ptm": ptm, "files_sha256": {"prediction.pdb": fr.sha256_file(d / "prediction.pdb")},
        }))
    return sig


def test_measure_records_end_to_end_on_synthetic_trees(tmp_path):
    wt_seq, gen_seq = WT["D1"], "ACDEFGHIKA"
    sig = _tree(tmp_path / "wt", {wt_seq: (_helix(10), 90.0, 0.8)})
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "D1.pdb").write_text(_pdb(_helix(10)))
    manifest = {"schema_version": fr.REFERENCE_SCHEMA, "evaluation_signature": sig,
                "domains": {"D1": {"pdb": "D1.pdb", "pdb_sha256": fr.sha256_file(tmp_path / "refs" / "D1.pdb")}}}
    (tmp_path / "refs" / "m.json").write_text(json.dumps(manifest))
    _tree(tmp_path / "gen", {gen_seq: (_helix(10), 70.0, 0.4)})
    recs = fr.validate_records([_rec(1, seq=gen_seq)], WT)
    msas = {"D1": [wt_seq, "ACDEF-----"]}
    out = fr.measure_records(recs, wild_types=WT, msas=msas, tree=tmp_path / "gen",
                             reference_manifest=tmp_path / "refs" / "m.json")[0]
    assert out["tm_score"] == pytest.approx(1.0)
    assert (out["plddt_mean"], out["ptm"]) == (70.0, 0.4)  # distinct endpoints, never combined
    assert out["identity_to_wild_type"] == 90.0 and out["max_msa_identity"] == 100.0  # the half-length row matches fully
    # a generated tree folded with different instrument settings is refused
    _tree(tmp_path / "gen2", {gen_seq: (_helix(10), 70.0, 0.4)}, signature_extra="b")
    with pytest.raises(ValueError, match="instrument settings"):
        fr.measure_records(recs, wild_types=WT, msas=msas, tree=tmp_path / "gen2",
                           reference_manifest=tmp_path / "refs" / "m.json")
    # a tampered reference structure is refused
    (tmp_path / "refs" / "D1.pdb").write_text(_pdb(_helix(10, 1.0)))
    with pytest.raises(ValueError, match="hash"):
        fr.load_reference_manifest(tmp_path / "refs" / "m.json")


def test_asset_resolution_does_not_depend_on_module_location(tmp_path):
    """The pod failure: data/ is not beside the code. Assets come from the options only."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    fasta = tmp_path / "staged" / "wt.faa"
    fasta.parent.mkdir()
    names = ["1E0L", "1F0M", "1IFY", "1K1V", "1UFM", "1V1C", "2K1B", "2KWH", "2LGW", "2LVN", "3DKM", "3L1X"]
    fasta.write_text("".join(f">{n}.pdb\n{'ACDEFGHIKL'}\n" for n in names))
    stage = root / "scripts/capability/generation/build_e20_fold_cohort.py"
    out = tmp_path / "out"
    run = subprocess.run([sys.executable, str(stage), "--device", "cpu", "--out", str(out), "--wild-type",
                          "--wild-type-fasta", str(fasta)], capture_output=True, text=True, cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    assert len((out / "fold_cohort.jsonl").read_text().splitlines()) == 12
    # a missing asset fails at once and names the option that supplies it
    bad = subprocess.run([sys.executable, str(stage), "--device", "cpu", "--out", str(tmp_path / "o2"), "--wild-type",
                          "--wild-type-fasta", str(tmp_path / "absent.faa")], capture_output=True, text=True, cwd=tmp_path)
    assert bad.returncode != 0 and "--wild-type-fasta" in bad.stderr
    with pytest.raises(FileNotFoundError, match="--msa-dir"):
        fr.load_msa(tmp_path / "nomsa", "D1", 10)
    with pytest.raises(FileNotFoundError, match="--protocol"):
        fr.load_roles(tmp_path / "nope.json")


def test_new_arms_accepted_and_c_minus_pr_emitted():
    recs = [_rec(i, arm=a) for i, a in enumerate(["Pr", "Cd", "Cx"], 1)]
    assert len(fr.validate_records(recs, WT)) == 3
    with pytest.raises(fr.ContractError, match="lambda"):
        fr.validate_records([_rec(1, arm="Pr", **{"lambda": 0.0})], WT)
    v = {"N": [0.2] * 8, "C": [0.5] * 8, "M": [0.3] * 8, "F": [0.7] * 8, "Pr": [0.4] * 8}
    pw = fri.analyse(_rows(v), roles=ROLES, role="evaluation", strengths=STRENGTHS)["results"]["m|tm_score|unstratified"]["pairwise"]
    assert pw["C-Pr"]["mean_over_domains"] == pytest.approx(0.1)
    assert pw["Pr-N"]["mean_over_domains"] == pytest.approx(0.2)


def _pair(i, arm, lf, lc, lam):
    return _rec(i, arm=arm, **{"lambda": lam, "lambda_fields": lf, "lambda_coupling": lc})


def test_two_strengths_of_one_arm_validate_and_cells_are_keyed_by_pair():
    recs = [_pair(1, "M", 0.25, 0.0, 0.25), _pair(2, "M", 0.5, 0.0, 0.5), _pair(3, "F", 0.25, 0.5, 0.5),
            _pair(4, "F", 0.5, 0.5, 0.5)]  # F: equal nominal lambda, distinct pairs
    out = fr.validate_records(recs, WT)
    assert len({fr.cell_key(r) for r in out}) == 4
    # without a pair the old rule stands: one lambda per cell
    with pytest.raises(fr.ContractError, match="within cell"):
        fr.validate_records([_rec(1, arm="M"), _rec(2, arm="M", **{"lambda": 2.0})], WT)
    with pytest.raises(fr.ContractError, match="inconsistent with arm"):
        fr.validate_records([_pair(1, "C", 0.5, 0.5, 0.5)], WT)
    with pytest.raises(fr.ContractError, match="together"):
        fr.validate_records([_rec(1, arm="M", lambda_fields=0.5)], WT)
    with pytest.raises(fr.ContractError, match="nominal"):
        fr.validate_records([_pair(1, "M", 0.5, 0.0, 0.25)], WT)
    # attempt budgets still must agree inside one pair-keyed cell
    with pytest.raises(fr.ContractError, match="within cell"):
        fr.validate_records([_pair(1, "M", 0.5, 0.0, 0.5), _pair(2, "M", 0.5, 0.0, 0.5) | {"attempt_budget": 9}], WT)


def _decl(arm, frozen, doses=None, model="m"):
    pair = lambda p: {"lambda_fields": p[0], "lambda_coupling": p[1]}
    entry = {"model": model, "arm": arm, "frozen_strength": pair(frozen)}
    if doses is not None:
        entry["dose_response_strengths"] = [pair(d) for d in doses]
    return entry


def test_strength_guard_accepts_only_declared_strengths():
    recs = fr.validate_records([_pair(1, "C", 0.0, 0.5, 0.5), _pair(2, "C", 0.0, 2.0, 2.0)], WT)
    ok = [_decl("C", (0.0, 0.5), [(0.0, 2.0)])]
    fr.require_frozen_strength(recs, ok)
    assert [r["id"] for r in fr.frozen_rows(recs, ok)] == ["r1"]  # the dose is structurally not primary
    with pytest.raises(fr.ContractError, match="undeclared strength"):  # dose not pre-declared
        fr.require_frozen_strength(recs, [_decl("C", (0.0, 0.5))])
    with pytest.raises(fr.ContractError, match="both the frozen strength and a dose-response"):
        fr.require_frozen_strength(recs, [_decl("C", (0.0, 0.5), [(0.0, 0.5), (0.0, 2.0)])])
    with pytest.raises(fr.ContractError, match="no declared frozen_strength"):  # arm C has records, entry is for M
        fr.require_frozen_strength(recs, [_decl("M", (0.5, 0.0))])
    with pytest.raises(fr.ContractError, match="keys must be"):  # no frozen_strength key at all
        fr.require_frozen_strength(recs, [{"model": "m", "arm": "C", "dose_response_strengths": []}])
    with pytest.raises(fr.ContractError, match="a strength is an object"):  # several frozen strengths
        fr.require_frozen_strength(recs, [dict(ok[0], frozen_strength=[ok[0]["frozen_strength"]] * 2)])
    with pytest.raises(fr.ContractError, match="twice"):
        fr.require_frozen_strength(recs, ok + ok)
    with pytest.raises(fr.ContractError, match="no records at its frozen_strength"):
        fr.require_frozen_strength(recs[1:], ok)


def test_analyse_reads_only_the_frozen_strength_and_counts_the_dose_rows():
    rows = _rows({"N": [0.2] * 8, "C": [0.4] * 8, "M": [0.3] * 8, "F": [0.7] * 8})
    for r in rows:
        r["lambda_fields"], r["lambda_coupling"] = (None, None)
    rows += [dict(r, arm="C", lambda_fields=0.0, lambda_coupling=4.0, tm_score=0.9) for r in rows if r["arm"] == "C"]
    for r in rows:
        if r["arm"] == "C" and r["lambda_coupling"] is None:
            r["lambda_fields"], r["lambda_coupling"] = 0.0, 1.0
    strengths = [e for e in STRENGTHS if e["arm"] != "C"] + [_decl("C", (0.0, 1.0), [(0.0, 4.0)])]
    out = fri.analyse(rows, roles=ROLES, role="evaluation", strengths=strengths)
    assert out["n_dose_response_rows_excluded"] == 8 * fr.MIN_ADMISSIBLE_PER_CELL
    assert out["results"]["m|tm_score|unstratified"]["factorial"]["C-N"]["mean_over_domains"] == pytest.approx(0.2)
    with pytest.raises(fr.ContractError, match="undeclared strength"):
        fri.analyse(rows, roles=ROLES, role="evaluation", strengths=[e for e in strengths if e["arm"] != "C"] + [_decl("C", (0.0, 1.0))])


def test_identity_stratum_has_a_per_cell_floor_and_reports_admissibility():
    rows = _rows({"N": [0.2] * 8, "C": [0.4] * 8, "M": [0.3] * 8, "F": [0.7] * 8})
    for r in rows:
        if r["domain"] == "d0" and r["arm"] == "C":  # d0/C: only 24 of its 25 below 20% identity
            r["identity_to_wild_type"] = 50.0 if r is [x for x in rows if x["domain"] == "d0" and x["arm"] == "C"][0] else 10.0
    block = fri.analyse(rows, roles=ROLES, role="evaluation", strengths=STRENGTHS)["results"]["m|tm_score|identity_restricted"]
    assert block["cell_admissibility"]["d0|C"] == {"admissible": 24, "depth": 25, "feasibility_rate": 24 / 25,
                                                   "floor": 25, "meets_floor": False}
    assert block["factorial_domains_excluded_partial_coverage"] == ["d0"]
    assert "d0" not in block["factorial"]["C-N"]["per_domain"]
    assert block["factorial"]["C-N"]["n_domains"] == 7


def test_development_selects_a_strength_without_contrasts():
    roles = {f"d{k}": "development" for k in range(4)}
    rows = []
    for k in range(4):
        for lf, v in ((0.25, 0.3), (0.5, 0.6), (1.0, 0.4)):
            rows.append({"model": "m", "domain": f"d{k}", "arm": "M", "lambda_fields": lf, "lambda_coupling": 0.0,
                         "tm_score": v + 0.01 * k, "plddt_mean": 50.0, "ptm": 0.5, "max_msa_identity": 40.0, "identity_to_wild_type": 10.0})
        rows.append({"model": "m", "domain": f"d{k}", "arm": "N", "lambda_fields": 0.0, "lambda_coupling": 0.0,
                     "tm_score": 0.2, "plddt_mean": 50.0, "ptm": 0.5, "max_msa_identity": 40.0, "identity_to_wild_type": 10.0})
    out = fri.analyse_development(rows, roles=roles, component_scale=[{"model": "m"}])
    sel = out["selection"]["unstratified"]
    assert sel["m|M"]["lambda_fields"] == 0.5 and sel["m|N"]["lambda_fields"] == 0.0
    best = [e for e in out["settings"] if e["stratum"] == "unstratified" and e["arm"] == "M" and e["lambda_fields"] == 0.5][0]
    assert best["endpoints"]["tm_score"]["pooled_over_domains"] == pytest.approx(0.6 + 0.015)
    assert set(best["endpoints"]) == {"tm_score", "plddt_mean", "ptm"} and out["component_scale"] == [{"model": "m"}]
    assert "factorial" not in json.dumps(out) and "interval" not in json.dumps(out).replace("cross-domain interval", "")
    with pytest.raises(ValueError, match="evaluation-role only"):
        fri.analyse(rows, roles=roles, role="development", strengths=STRENGTHS)


def test_reference_arms_take_null_language_model_fields_and_no_other_arm_does():
    ref = lambda arm, **kw: _rec(1, arm=arm, model=fr.MODEL_FREE_ARMS[arm], native_loglik=None,
                                 mean_step_entropy=None, guidance_score=None, **kw)
    for arm in ("P", "H"):
        assert fr.validate_records([ref(arm, lambda_fields=None, lambda_coupling=None)], WT)
    with pytest.raises(fr.ContractError, match="native_loglik is null for arm C"):
        fr.validate_records([_rec(1, arm="C", native_loglik=None)], WT)
    with pytest.raises(fr.ContractError, match="mean_step_entropy is null for arm N"):
        fr.validate_records([_rec(1, arm="N", mean_step_entropy=None)], WT)
    with pytest.raises(fr.ContractError, match="reserved"):  # P must be labelled potts-joint
        fr.validate_records([ref("P") | {"model": "natural"}], WT)
    with pytest.raises(fr.ContractError, match="reserved"):  # a language-model arm may not borrow the label
        fr.validate_records([_rec(1, arm="C", model="natural")], WT)
    with pytest.raises(fr.ContractError, match="must be null"):
        fr.validate_records([ref("H", lambda_fields=0.0, lambda_coupling=0.0)], WT)


def test_a_protocol_without_a_strengths_declaration_refuses_evaluation_clearly(tmp_path):
    protocol = tmp_path / "p.json"
    protocol.write_text(json.dumps({"roles": {"development": [], "evaluation": []}}))
    with pytest.raises(fr.ContractError, match="declares no ``strengths``"):
        fr.load_strengths(protocol)
