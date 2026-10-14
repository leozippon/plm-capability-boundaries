"""E20 reference arms P and H: the refusals that keep the primary endpoint's scale honest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.capability.generation import generate_reference_arms as stage
from scripts.capability.generation import merge_e20_cells as merge
from src.capability.context.family_potts import MixingError, save_family_model
from src.capability.core.amino_acids import AA20
from src.capability.generation import fold_recapitulation as fr
from src.capability.generation import reference_arms as ra

ROOT = Path(__file__).resolve().parents[2]
JOINTH = ROOT / "data/pairwise_assets/family_references_jointh_20261013"
MSA_DIR = ROOT / "data/pairwise_assets/same12_provenance_20260924/msa"
L = 6


@pytest.fixture
def world(tmp_path):
    rng = np.random.default_rng(0)
    wt = "".join(rng.choice(list(AA20), size=L))
    save_family_model(tmp_path / "ref", "AAA", wildtype=wt, h=rng.normal(size=(L, 20)) * .3,
                      pairs=rng.normal(size=(L * (L - 1) // 2, 20, 20)) * .05,
                      profile_freq=rng.dirichlet(np.ones(20), size=L), profile_column_weight=np.full(L, 10.0),
                      qualified=True, qualification_failures=[], extra={})
    protocol = {"schema": "coupling_guided_generation_v1",
                "domains": {"AAA": {"L": L, "wt_sha256": hashlib.sha256(wt.encode()).hexdigest()}},
                "seed_streams": [1000, 3000, 5000],
                "reference_arms": {"potts_joint": {"chains": 40, "sweeps": 20, "mixing_tolerance": 0.99},
                                   "natural": {"max_rows": 3, "min_admissible": 2}}}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    return tmp_path, path, wt, protocol


def _run(tmp, protocol, name, arm, stream=0, **extra):
    argv = ["--out", str(tmp / name), "--protocol", str(protocol), "--arm", arm, "--domain", "AAA",
            "--stream-index", str(stream)]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return stage.main(argv)


def test_potts_cell_records_protocol_values_and_null_language_model_fields(world):
    tmp, protocol, wt, _ = world
    _run(tmp, protocol, "p", "P", family_reference_dir=tmp / "ref")
    rows = [json.loads(x) for x in (tmp / "p/sequences.jsonl").read_text().splitlines()]
    assert len(rows) == 40 and {r["model"] for r in rows} == {"potts-joint"} and {r["arm"] for r in rows} == {"P"}
    assert all(r[k] is None for r in rows for k in ("native_loglik", "mean_step_entropy", "guidance_score"))
    assert fr.validate_records(rows, {"AAA": wt})
    done = json.loads((tmp / "p/guided_decoding_pool.json").read_text())
    sampling = done["reference_arm"]["sampling"]
    assert done["reference_arm"]["protocol_values"] == {"chains": 40, "sweeps": 20, "mixing_tolerance": 0.99}
    assert sampling["sweeps"] == 20 and sampling["check_sweeps"] == 10 and 0 <= sampling["mixing_max_tv"] <= 1


def test_failed_mixing_check_fails_the_cell_with_no_records_and_no_retry(world):
    tmp, protocol, _, spec = world
    spec["reference_arms"]["potts_joint"]["mixing_tolerance"] = 1e-6
    protocol.write_text(json.dumps(spec))
    with pytest.raises(MixingError):
        _run(tmp, protocol, "p", "P", family_reference_dir=tmp / "ref")
    assert not (tmp / "p/sequences.jsonl").exists() and not (tmp / "p/guided_decoding_pool.json").exists()
    failure = json.loads((tmp / "p" / stage.FAILURE_FILENAME).read_text())
    assert failure["status"] == "failed" and failure["mixing"]["max_tv"] > 1e-6
    assert failure["mixing"]["sweeps"] == 20  # the declared sweeps; never raised


def test_protocol_values_are_declared_not_defaulted(world):
    tmp, protocol, _, spec = world
    del spec["reference_arms"]["potts_joint"]["sweeps"]
    protocol.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="exactly"):
        _run(tmp, protocol, "p", "P", family_reference_dir=tmp / "ref")


def _msa(rows):
    return [(f"r{i}", s) for i, s in enumerate(rows)]


def test_gapped_rows_are_excluded_never_truncated_or_padded():
    msa = _msa(["ACDE", "AC-E", "ACDF", "ACDG", "ACDH"])
    rows, m = ra.natural_cell(msa, np.array([0, 1, 1, 1, 1], bool), domain="X", length=4, max_rows=10,
                              min_admissible=2, seed=1)
    assert [r["sequence"] for r in rows] == ["ACDF", "ACDG", "ACDH"]
    assert m["held_out_rows"] == 4 and m["gapped_held_out_rows_excluded"] == 1 and m["admissible_rows"] == 3
    with pytest.raises(fr.ContractError, match="canonical"):  # the contract refuses a gapped sequence outright
        fr.validate_records([ra.contract_row("H", "X", 0, 1, "AC-E")], {"X": "ACDE"})


def test_length_mismatch_is_refused():
    with pytest.raises(ValueError, match="columns"):
        ra.natural_cell(_msa(["ACDE", "ACD", "ACDF"]), np.array([0, 1, 1], bool), domain="X", length=4,
                        max_rows=10, min_admissible=1, seed=1)
    with pytest.raises(ValueError, match="columns"):  # a non-residue symbol is not silently a gap
        ra.natural_cell(_msa(["ACDE", "ACDX"]), np.array([0, 1], bool), domain="X", length=4,
                        max_rows=10, min_admissible=1, seed=1)


def test_subsampling_is_deterministic_and_recorded_and_scarcity_is_not_an_error():
    msa = _msa(["ACDE"] + [f"AC{a}{b}" for a in "DEF" for b in "GHIK"])
    held = np.array([0] + [1] * 12, bool)
    kw = dict(domain="X", length=4, max_rows=5, min_admissible=2, seed=7)
    (r1, m1), (r2, m2) = (ra.natural_cell(msa, held, **kw) for _ in range(2))
    assert r1 == r2 and m1["subsampled"] and len(m1["selected"]) == 5 and m1["admissible_rows"] == 12
    rows, m = ra.natural_cell(msa[:3], held[:3], **{**kw, "min_admissible": 25})
    assert len(rows) == 2 and m["below_min_admissible"] and m["admissible_rows"] == 2
    rows, m = ra.natural_cell(msa[:1], held[:1], **kw)
    assert rows == [] and m["admissible_rows"] == 0


@pytest.mark.skipif(not (JOINTH / "qualification.json").exists(), reason="data/ not staged")
def test_natural_cell_reproduces_the_recorded_holdout_on_a_real_domain(tmp_path):
    wt = fr.read_fasta(MSA_DIR / "1E0L.pdb.fasta")[0][1]
    protocol = {"schema": "coupling_guided_generation_v1", "seed_streams": [11],
                "domains": {"1E0L": {"L": len(wt), "wt_sha256": hashlib.sha256(wt.encode()).hexdigest()}},
                "reference_arms": {"natural": {"max_rows": 100, "min_admissible": 25}}}
    (tmp_path / "p.json").write_text(json.dumps(protocol))
    qual = JOINTH / "qualification.json"
    argv = ["--out", str(tmp_path / "h"), "--protocol", str(tmp_path / "p.json"), "--arm", "H", "--domain", "1E0L",
            "--stream-index", "0", "--msa-dir", str(MSA_DIR), "--qualification", str(qual)]
    stage.main(argv)
    done = json.loads((tmp_path / "h/guided_decoding_pool.json").read_text())["reference_arm"]["sampling"]
    assert done["held_out_rows"] == done["qualification_held_out_rows"] == 117
    rows = [json.loads(x) for x in (tmp_path / "h/sequences.jsonl").read_text().splitlines()]
    assert len(rows) == min(100, done["admissible_rows"]) and all("-" not in r["sequence"] for r in rows)
    # a qualification whose recorded split differs from the fit script's is refused, not trusted
    tampered = json.loads(qual.read_text())
    tampered["split"]["seed"] += 1
    (tmp_path / "q.json").write_text(json.dumps(tampered))
    argv[-1], argv[1] = str(tmp_path / "q.json"), str(tmp_path / "h2")
    with pytest.raises(ValueError, match="split parameters"):
        stage.main(argv)
    tampered = json.loads(qual.read_text())
    tampered["domains"]["1E0L.pdb"]["held_out_rows"] += 1
    (tmp_path / "q2.json").write_text(json.dumps(tampered))
    argv[-1], argv[1] = str(tmp_path / "q2.json"), str(tmp_path / "h3")
    with pytest.raises(ValueError, match="recomputed holdout"):
        stage.main(argv)


def test_reference_cells_merge_with_the_guided_cells_unchanged(world):
    tmp, protocol, wt, spec = world
    _run(tmp, protocol, "p", "P", family_reference_dir=tmp / "ref")
    strengths = [{"model": "potts-joint", "arm": "P",
                  "frozen_strength": {"lambda_fields": None, "lambda_coupling": None}}]
    out = tmp / "merged"
    summary = merge.merge_records([tmp / "p"], out, {"AAA": wt}, {"AAA": "evaluation"}, "evaluation", 1, strengths)
    assert summary == {"cells": 1, "records": 40}
    with pytest.raises(ValueError, match="strengths declaration"):
        merge.merge_records([tmp / "p"], tmp / "m2", {"AAA": wt}, {"AAA": "evaluation"}, "evaluation", 1)
