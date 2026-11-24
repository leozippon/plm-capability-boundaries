"""Tests for unified 62-model R3 contact discrimination & bidirectional yardstick."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.capability.core.io import sha256_file
from src.capability.unified.contacts_yardstick import (
    DEFAULT_LEGACY_EXTRACTION_ROOTS,
    _parse_upstream_tolerance,
    canonicalize_arm_id,
    discover_e01_extractions,
    evaluate_bidirectional_panel_fast,
    run_contacts_unified_62,
    two_sample_depth_contrast_bands,
)
from src.capability.unified.roster import (
    REPO_ROOT,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_OUTPUT_DIR,
)


def _assert_finite(obj: object, path: str = "root") -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float at {path}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_finite(v, f"{path}[{i}]")


def test_canonicalize_arm_id_and_prefix_tolerance_rules() -> None:
    """Directory names normalize cleanly and tier-3 prefix violations are refused."""
    assert canonicalize_arm_id("posresp_extract_dplm_650m_gpu0") == "dplm-650m"
    assert canonicalize_arm_id("posrespg_extract_esm2_3b_gpu0") == "esm2-3b"
    assert canonicalize_arm_id("posrespx_extract_rita_xl_gpu0") == "rita-xl"
    assert canonicalize_arm_id("posrespe01b_extract_llama_2_7b_gpu0") == "llama-2-7b"
    assert canonicalize_arm_id("posresp_extract_qwen2_5_7b_gpu0") == "qwen2.5-7b"
    assert canonicalize_arm_id("posresp_extract_galactica_1_3b_gpu0") == "galactica-1.3b"
    assert canonicalize_arm_id("esm3-1.4b") == "esm3-1.4b"
    with pytest.raises(ValueError):
        canonicalize_arm_id("not_a_real_model_arm")

    tol1, tier1, ref1, cav1 = _parse_upstream_tolerance(
        {"prefix_invariant": {"tier": 1, "admitted_tolerance_nats": 0.0, "upstream_max_abs_nats": 0.0}},
        "causal_next_token",
    )
    assert tol1 == 0.0 and tier1 == 1 and ref1 is None and cav1 is None

    tol2, tier2, ref2, cav2 = _parse_upstream_tolerance(
        {
            "prefix_invariant": {
                "tier": 2,
                "admitted_tolerance_nats": 7.5e-5,
                "upstream_max_abs_nats": 2.5e-5,
            }
        },
        "causal_next_token",
    )
    assert tol2 == pytest.approx(7.5e-5) and tier2 == 2 and ref2 is None and cav2 is None

    tol_sub, tier_sub, ref_sub, cav_sub = _parse_upstream_tolerance(
        {
            "prefix_invariant": {
                "tier": 3,
                "admitted_tolerance_nats": 0.0,
                "repeat_max_abs_nats": 0.0,
                "upstream_max_abs_nats": 8.94e-6,
            }
        },
        "causal_next_token",
    )
    assert ref_sub is None and tier_sub == 3 and cav_sub is not None and tol_sub < 1e-4

    _, tier3, ref3, _ = _parse_upstream_tolerance(
        {
            "prefix_invariant": {
                "tier": 3,
                "admitted_tolerance_nats": 0.0,
                "upstream_max_abs_nats": 5.0e-3,
            }
        },
        "causal_next_token",
    )
    assert tier3 == 3 and ref3 is not None


def test_discovery_dedup_and_refused_moe_receipt(tmp_path: Path) -> None:
    """Discovery skips _verify_* dirs, prefers float32 over bf16, and loads refused_moe_layout.json."""
    r1 = tmp_path / "run1"
    r2 = tmp_path / "run2"
    (r1 / "_verify_esm2_650m").mkdir(parents=True)
    (r1 / "_verify_esm2_650m" / "position_likelihood.json").write_text(
        json.dumps({"identity": {"arm": "esm2-650m", "paradigm": "masked_marginal_pseudolikelihood"}, "assays": []}),
        encoding="utf-8",
    )
    # r1 has float32 for esm2-8m; r2 has bfloat16 for esm2-8m -> float32 in r1 must win!
    (r1 / "posresp_extract_esm2-8m").mkdir(parents=True)
    (r1 / "posresp_extract_esm2-8m" / "position_likelihood.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "identity": {
                    "arm": "esm2-8m",
                    "paradigm": "masked_marginal_pseudolikelihood",
                    "dtype": "float32",
                },
                "assays": [],
            }
        ),
        encoding="utf-8",
    )
    (r2 / "posresp_extract_esm2-8m").mkdir(parents=True)
    (r2 / "posresp_extract_esm2-8m" / "position_likelihood.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "identity": {
                    "arm": "esm2-8m",
                    "paradigm": "masked_marginal_pseudolikelihood",
                    "dtype": "bfloat16",
                },
                "assays": [],
            }
        ),
        encoding="utf-8",
    )
    (r2 / "refused_moe_layout.json").write_text(
        json.dumps(
            {
                "schema": "e01_refused_moe_layout_v1",
                "refused_arms": [
                    {"arm": "progen3-1b", "status": "refused_moe_layout", "upstream_max_abs_nats": 0.01}
                ],
            }
        ),
        encoding="utf-8",
    )

    found, refused = discover_e01_extractions(extraction_roots=[r1, r2])
    assert set(found.keys()) == {"esm2-8m"}
    assert found["esm2-8m"].dtype_realised == "float32"
    assert found["esm2-8m"].directory == r1 / "posresp_extract_esm2-8m"
    assert {"progen3-3b", "protgpt3-1.3b", "progen3-1b"}.issubset(refused.keys())


def test_two_sample_depth_contrast_bands_symmetry_and_ordering() -> None:
    """Shallow-minus-Deep and Deep-minus-Shallow must be exact negatives with simultaneous >= marginal."""
    rng = np.random.default_rng(12345)
    shallow = np.column_stack(
        [
            rng.normal(loc=0.06, scale=0.01, size=15),
            rng.normal(loc=0.00, scale=0.01, size=15),
            rng.normal(loc=0.03, scale=0.01, size=15),
        ]
    )
    deep = np.column_stack(
        [
            rng.normal(loc=0.01, scale=0.01, size=13),
            rng.normal(loc=0.00, scale=0.01, size=13),
            rng.normal(loc=0.10, scale=0.01, size=13),
        ]
    )
    res = two_sample_depth_contrast_bands(
        shallow, deep, ["arm_a", "arm_b", "arm_c"], draws=2000, seed=20261106
    )
    assert res["n_shallow_groups"] == 15
    assert res["n_deep_groups"] == 13
    assert res["columns"][0]["shallow_minus_deep"]["verdict"] == "positive"
    assert res["columns"][2]["deep_minus_shallow"]["verdict"] == "positive"
    for row in res["columns"]:
        sd = row["shallow_minus_deep"]
        ds = row["deep_minus_shallow"]
        assert sd["point"] == pytest.approx(-ds["point"], abs=1e-12)
        assert sd["interval"][0] == pytest.approx(-ds["interval"][1], abs=1e-12)
        assert sd["marginal_interval"][0] == pytest.approx(-ds["marginal_interval"][1], abs=1e-12)
        sim_w = sd["interval"][1] - sd["interval"][0]
        mrg_w = sd["marginal_interval"][1] - sd["marginal_interval"][0]
        assert sim_w >= mrg_w - 1e-9


def test_uc_memoized_evaluator_matches_reference_script_bit_exact() -> None:
    """evaluate_bidirectional_panel_fast must reproduce fit_contact_bidirectional.evaluate_panel to < 1e-12."""
    import fit_contact_bidirectional as fcb

    rng = np.random.default_rng(20261124)
    n_refs = 10
    blocks, support = {}, {}
    model: dict[str, dict[str, dict[str, np.ndarray]]] = {"arm_a": {}, "arm_b": {}, "arm_c": {}}
    for idx in range(n_refs):
        assay = f"REF_{idx:02d}"
        n_pairs = 40
        label = np.zeros(n_pairs, dtype=np.float64)
        label[:6] = 1.0
        rng.shuffle(label)
        blocks[assay] = {
            "geom": rng.normal(size=(n_pairs, 7)),
            "prof": rng.normal(size=(n_pairs, 4)),
            "couplings": rng.normal(size=(n_pairs, 1)) + 0.8 * label[:, None],
            "label": label,
        }
        support[assay] = {"family": f"FAM_{idx:02d}"}
        rows_full = np.arange(n_pairs, dtype=np.int64)
        rows_sub = np.arange(0, n_pairs, 2, dtype=np.int64)
        if len(set(label[rows_sub].tolist())) < 2:
            rows_sub = rows_full
        # arm_a and arm_b share identical support (exercises U_C memoization); arm_c has distinct support.
        model["arm_a"][assay] = {"rows": rows_full, "column": rng.normal(size=(len(rows_full), 1)) + 0.5 * label[:, None]}
        model["arm_b"][assay] = {"rows": rows_full, "column": rng.normal(size=(len(rows_full), 1))}
        model["arm_c"][assay] = {"rows": rows_sub, "column": rng.normal(size=(len(rows_sub), 1))}

    panel_arms = [
        {"arm": name, "references": sorted(blocks.keys())}
        for name in ("arm_a", "arm_b", "arm_c")
    ]
    ref = fcb.evaluate_panel(panel_arms, blocks, model, support)
    fast = evaluate_bidirectional_panel_fast(panel_arms, blocks, model, support, workers=2)

    assert fast["complete"] == ref["complete"]
    assert fast["shared"] == ref["shared"]
    for stratum in fcb.STRATA:
        assert fast["strata"][stratum]["native"]["critical_value"] == pytest.approx(
            ref["strata"][stratum]["native"]["critical_value"], abs=1e-12
        )
        for c_fast, c_ref in zip(fast["strata"][stratum]["columns"], ref["strata"][stratum]["columns"]):
            assert c_fast["point"] == pytest.approx(c_ref["point"], abs=1e-12)
            assert c_fast["interval"] == pytest.approx(c_ref["interval"], abs=1e-12)
            assert c_fast["marginal_interval"] == pytest.approx(c_ref["marginal_interval"], abs=1e-12)
        for c_fast, c_ref in zip(
            fast["strata"][stratum]["pure_couplings_increment"]["columns"],
            ref["strata"][stratum]["pure_couplings_increment"]["columns"],
        ):
            assert c_fast["point"] == pytest.approx(c_ref["point"], abs=1e-12)


def test_legacy_23_arm_smoke_and_exact_parity_against_20261120(tmp_path: Path) -> None:
    """Running on the 23 legacy local E01 arms reproduces 2026-11-20 discrimination & bidirectional exactly."""
    legacy_roots = [REPO_ROOT / rel for rel in DEFAULT_LEGACY_EXTRACTION_ROOTS]
    res = run_contacts_unified_62(
        root=REPO_ROOT,
        out_dir=tmp_path,
        extraction_roots=legacy_roots,
        workers=4,
        write_outputs=True,
    )
    master = res["master"]
    _assert_finite(master)

    acct = master["roster_accounting"]
    assert acct["n_roster_total"] == 62
    assert acct["n_accounted_total"] == 62
    assert acct["n_admitted_full_alphabet"] == 16
    assert acct["n_admitted_thin_bpe"] == 6
    assert acct["n_refused_below_8_refs"] == 1
    assert acct["n_refused_moe_prefix_layout"] == 2
    assert acct["n_not_yet_extracted"] == 37

    # Verify all 7 JSON + .sha256 sidecars exist and match
    expected_files = [
        "contacts_unified_62.json",
        "contact_discrimination_unified_62.json",
        "contact_discrimination_unified_62_deep.json",
        "contact_discrimination_unified_62_shallow.json",
        "contact_bidirectional_unified_62.json",
        "contact_bidirectional_unified_62_deep.json",
        "contact_bidirectional_unified_62_shallow.json",
    ]
    for fname in expected_files:
        fpath = tmp_path / fname
        spath = tmp_path / f"{fname}.sha256"
        assert fpath.is_file() and spath.is_file()
        assert sha256_file(fpath) == spath.read_text(encoding="utf-8").strip().split()[0]

    # Exact parity check against frozen 2026-11-20 corrected refit (all 3 strata!)
    ref_disc_dir = REPO_ROOT / "results/extensions/contact_discrimination_20261120"
    ref_bidir_dir = REPO_ROOT / "results/extensions/contact_bidirectional_20261120"
    for skey, suffix in (("all", ""), ("deep", "_deep"), ("shallow", "_shallow")):
        ref_d = json.loads((ref_disc_dir / f"contact_discrimination{suffix}.json").read_text(encoding="utf-8"))
        new_d = res["discrimination"][skey]
        assert new_d["references_complete"] == ref_d["references_complete"]
        assert new_d["full_alphabet_secondary"]["references_complete"] == ref_d["full_alphabet_secondary"]["references_complete"]
        ref_fa_by_arm = {
            (canonicalize_arm_id(c["readout"].split("propagation:", 1)[1]) if c["kind"] == "model" else c["readout"]): c["point"]
            for c in ref_d["full_alphabet_secondary"]["columns"]
        }
        new_fa_by_arm = {
            (c["arm_id"] if c["kind"] == "model" else c["readout"]): c["point"]
            for c in new_d["full_alphabet_secondary"]["columns"]
        }
        assert set(new_fa_by_arm) == set(ref_fa_by_arm)
        for k in ref_fa_by_arm:
            assert new_fa_by_arm[k] == pytest.approx(ref_fa_by_arm[k], abs=1e-12)

        ref_b = json.loads((ref_bidir_dir / f"contact_bidirectional{suffix}.json").read_text(encoding="utf-8"))
        new_b = res["bidirectional"][skey]
        assert new_b["full_alphabet_secondary"]["references_offered"] == ref_b["full_alphabet_secondary"]["references_offered"]
        for stratum in ("contact_ranking", "contact_squared_error", "contact_squared_error_clipped"):
            ref_cols = {
                (canonicalize_arm_id(c["arm"]), c["rung"], c["direction"]): c["point"]
                for c in ref_b["full_alphabet_secondary"]["strata"][stratum]["columns"]
            }
            new_cols = {
                (c["arm"], c["rung"], c["direction"]): c["point"]
                for c in new_b["full_alphabet_secondary"]["strata"][stratum]["columns"]
            }
            assert set(new_cols) == set(ref_cols)
            for k in ref_cols:
                assert new_cols[k] == pytest.approx(ref_cols[k], abs=1e-12)


def test_production_unified_contacts_artifacts_integrity() -> None:
    """Validate final unified 62-model contacts artifacts & .sha256 sidecars once generated."""
    out_dir = REPO_ROOT / UNIFIED_OUTPUT_DIR
    master_path = out_dir / "contacts_unified_62.json"
    if not master_path.is_file():
        pytest.skip("Production contacts_unified_62.json not yet generated (awaiting H200 extraction pull)")

    expected_files = [
        "contacts_unified_62.json",
        "contact_discrimination_unified_62.json",
        "contact_discrimination_unified_62_deep.json",
        "contact_discrimination_unified_62_shallow.json",
        "contact_bidirectional_unified_62.json",
        "contact_bidirectional_unified_62_deep.json",
        "contact_bidirectional_unified_62_shallow.json",
    ]
    for fname in expected_files:
        fpath = out_dir / fname
        spath = out_dir / f"{fname}.sha256"
        assert fpath.is_file(), f"Missing {fpath}"
        assert spath.is_file(), f"Missing {spath}"
        assert sha256_file(fpath) == spath.read_text(encoding="utf-8").strip().split()[0]

    doc = json.loads(master_path.read_text(encoding="utf-8"))
    _assert_finite(doc)
    assert doc["schema"] == "contacts_unified_62_v1"
    acct = doc["roster_accounting"]
    assert acct["n_roster_total"] == 62
    assert acct["n_accounted_total"] == 62
    assert set(doc["per_arm"].keys()) == set(UNIFIED_62_SINGLE_SEQUENCE_ARMS)
    assert doc["depth_strata"]["all"]["n_complete_full_alphabet"] == 28
    assert doc["depth_strata"]["deep"]["n_complete_full_alphabet"] == 13
    assert doc["depth_strata"]["shallow"]["n_complete_full_alphabet"] == 15
    assert "full_alphabet" in doc["depth_contrasts_shallow_vs_deep"]
    assert "auc_minus_half" in doc["depth_contrasts_shallow_vs_deep"]["full_alphabet"]
    assert "forward_model_given_gpc" in doc["depth_contrasts_shallow_vs_deep"]["full_alphabet"]["contact_ranking"]
    assert "reverse_couplings_given_gpM" in doc["depth_contrasts_shallow_vs_deep"]["full_alphabet"]["contact_ranking"]
