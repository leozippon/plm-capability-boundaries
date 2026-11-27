"""Tests for the reusable external capability benchmark interface (positive, negative & smoke/full decoupling)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.capability.benchmark import (
    ALL_SUITES,
    AdapterValidationError,
    BaseProteinBenchmarkAdapter,
    CapabilityFlags,
    CapabilityNotSupportedError,
    ModelMetadata,
    compute_rank_and_percentile,
    load_custom_adapter,
    load_reference_leaderboards,
    pack_mid_and_last_hidden_states,
    run_benchmark,
    validate_adapter,
)
from src.capability.benchmark.protocol import AA20_INDEX
from src.capability.benchmark.runner import (
    R2_MEGASCALE_COHORT_REL,
    R3_EPISTASIS_PLAN_REL,
    R4_FAMILY_COND_REL,
    R5_CONTEXT_COHORT_REL,
)
from src.capability.core.amino_acids import AA20

REPO_ROOT = Path(__file__).resolve().parents[2]


class _LeakyMockAdapter(BaseProteinBenchmarkAdapter):
    """Deliberately leaks target residue identity into site_conditional_logprobs."""

    def __init__(self) -> None:
        super().__init__(
            ModelMetadata(
                model_id="leaky-mock",
                display_name="LeakyMock",
                paradigm="masked_lm",
                modality="protein",
                n_params_millions=1.0,
                max_positions=512,
            ),
            CapabilityFlags(site_conditional_logprobs=True),
        )

    def site_conditional_logprobs(self, sequence, positions=None, *, coords=None) -> np.ndarray:
        pos_list = list(range(len(sequence))) if positions is None else [int(p) for p in positions]
        out = np.zeros((len(pos_list), 20), dtype=np.float32)
        for i, p in enumerate(pos_list):
            logits = np.zeros(20, dtype=np.float32)
            logits[AA20_INDEX[sequence[p]]] = 4.0  # LEAKS target residue!
            logits -= np.logaddexp.reduce(logits)
            out[i] = logits
        return out


class _MinimalScoringOnlyAdapter(BaseProteinBenchmarkAdapter):
    """Declares ONLY score_mutations to test graceful not_supported reporting on other suites."""

    def __init__(self) -> None:
        super().__init__(
            ModelMetadata(
                model_id="scoring-only-mock",
                display_name="ScoringOnlyMock",
                paradigm="masked_lm",
                modality="protein",
                n_params_millions=5.0,
                max_positions=512,
            ),
            CapabilityFlags(score_mutations=True),
        )

    def score_mutations(self, wildtype, mutations, *, coords=None) -> np.ndarray:
        return np.linspace(-1.0, 1.0, len(mutations), dtype=np.float64)


class _FullMockAdapter(BaseProteinBenchmarkAdapter):
    """Clean zero-leakage mock adapter implementing all scoring, readout & generation hooks."""

    def __init__(self) -> None:
        super().__init__(
            ModelMetadata(
                model_id="full-mock-v1",
                display_name="FullMock-10M",
                paradigm="masked_lm",
                modality="protein",
                conditioning_regime="sequence_only",
                input_alignment_format="unaligned_concat",
                n_params_millions=10.0,
                max_positions=512,
            ),
            CapabilityFlags(
                score_mutations=True,
                site_conditional_logprobs=True,
                score_sequence_nll=True,
                extract_representations_1024=True,
                generate_unconditional=True,
                infill_local=True,
                sample_family_guided=True,
                continue_from_prefix=True,
            ),
        )

    def site_conditional_logprobs(self, sequence, positions=None, *, coords=None) -> np.ndarray:
        pos_list = list(range(len(sequence))) if positions is None else [int(p) for p in positions]
        out = np.empty((len(pos_list), 20), dtype=np.float32)
        for i, p in enumerate(pos_list):
            left = ord(sequence[p - 1]) if p > 0 else 65
            right = ord(sequence[p + 1]) if p + 1 < len(sequence) else 65
            logits = np.linspace(-0.5, 0.5, 20, dtype=np.float32) + 0.01 * ((left + right) % 7)
            out[i] = logits - np.logaddexp.reduce(logits)
        return out

    def score_mutations(self, wildtype, mutations, *, coords=None) -> np.ndarray:
        return np.array([0.1 * (i + 1) for i in range(len(mutations))], dtype=np.float64)

    def score_sequence_nll(self, sequence, *, context_items=(), coords=None, context_coords=()) -> dict[str, float]:
        ctx = "".join(context_items)
        bonus = 0.25 if (ctx and ctx[:4] == sequence[:4]) else 0.0
        nll_per = max(0.5, 2.5 - bonus)
        return {"nll_sum": float(nll_per * len(sequence)), "nll_per_residue": float(nll_per)}

    def extract_representations_1024(self, sequences, *, coords_list=None) -> np.ndarray:
        rows = []
        for s in sequences:
            rng = np.random.default_rng(len(s) + ord(s[0]))
            h_mid = rng.standard_normal((len(s), 48)).astype(np.float32)
            h_last = rng.standard_normal((len(s), 48)).astype(np.float32) + 0.5
            rows.append(pack_mid_and_last_hidden_states(h_mid, h_last))
        return np.stack(rows, axis=0)

    def generate_unconditional(self, lengths, *, seed=20261124) -> list[str]:
        rng = np.random.default_rng(int(seed))
        return ["".join(AA20[int(k)] for k in rng.integers(0, 20, size=int(L))) for L in lengths]

    def infill_local(self, template, mask_positions, *, coords=None, seed=20261124) -> str:
        rng = np.random.default_rng(int(seed))
        chars = list(template)
        for p in mask_positions:
            chars[int(p)] = AA20[int(rng.integers(0, 20))]
        return "".join(chars)

    def sample_family_guided(self, length, *, n_samples, profile_log_bias=None, guidance_scale=0.0, coords=None, seed=20261124) -> list[str]:
        rng = np.random.default_rng(int(seed))
        return ["".join(AA20[int(k)] for k in rng.integers(0, 20, size=int(length))) for _ in range(int(n_samples))]

    def continue_from_prefix(self, context_items, target_length, *, coords=None, seed=20261124) -> str:
        rng = np.random.default_rng(int(seed))
        return "".join(AA20[int(k)] for k in rng.integers(0, 20, size=int(target_length)))


def test_validate_adapter_catches_target_leakage_and_bad_metadata() -> None:
    leaky = _LeakyMockAdapter()
    with pytest.raises(AdapterValidationError, match="Target-residue leakage"):
        validate_adapter(leaky)

    with pytest.raises(ValueError, match="conditioning_regime"):
        ModelMetadata(
            model_id="bad",
            display_name="Bad",
            paradigm="masked_lm",
            modality="protein",
            conditioning_regime="not_a_regime",
        )

    minimal = _MinimalScoringOnlyAdapter()
    with pytest.raises(CapabilityNotSupportedError):
        minimal.generate_unconditional([10])


def test_pack_mid_and_last_hidden_states_deterministic_and_shapes() -> None:
    rng = np.random.default_rng(7)
    h_mid = rng.standard_normal((12, 64)).astype(np.float32)
    h_last = rng.standard_normal((12, 64)).astype(np.float32) + 1.0

    v1 = pack_mid_and_last_hidden_states(h_mid, h_last)
    v2 = pack_mid_and_last_hidden_states(h_mid, h_last)
    assert v1.shape == (1024,) and v1.dtype == np.float32
    assert np.array_equal(v1, v2)
    assert not np.allclose(v1[:512], v1[512:])

    batch = pack_mid_and_last_hidden_states(np.stack([h_mid, h_mid]), np.stack([h_last, h_last]))
    assert batch.shape == (2, 1024) and np.allclose(batch[0], v1, atol=1e-6)


def test_reference_leaderboards_and_rank_percentile() -> None:
    boards = load_reference_leaderboards()
    assert "r1_dms" in boards and len(boards["r1_dms"].scores_by_arm) >= 62
    assert "r2_biophysical" in boards and len(boards["r2_biophysical"].scores_by_arm) >= 62
    assert "r3_contacts" in boards and len(boards["r3_contacts"].scores_by_arm) >= 60
    assert "r5_context" in boards and len(boards["r5_context"].scores_by_arm) >= 62

    top = compute_rank_and_percentile(0.999, boards["r1_dms"])
    bot = compute_rank_and_percentile(-0.999, boards["r1_dms"])
    assert top["rank"] == 1 and top["percentile"] == 100.0
    assert bot["rank"] == top["n_reference_models"] + 1 and bot["percentile"] == 0.0


def test_smoke_mode_never_attaches_reference_comparison_and_prints_banner(tmp_path: Path) -> None:
    adapter = _FullMockAdapter()
    out_dir = tmp_path / "smoke_out"
    rep = run_benchmark(adapter, suites=ALL_SUITES, mode="smoke", out_dir=out_dir)
    assert rep["evaluation_scope"] == "contract_smoke_only"
    assert rep["suites"]["validate"]["status"] == "ok"
    for suite in ALL_SUITES:
        if suite == "validate":
            continue
        sres = rep["suites"][suite]
        assert sres["status"] == "smoke_ok", f"{suite} expected smoke_ok, got {sres}"
        assert sres["evaluation_scope"] == "contract_smoke_only"
        assert "reference_comparison" not in sres, f"{suite} must NOT attach reference_comparison in smoke mode!"

    md_text = (out_dir / "full-mock-v1_benchmark_summary.md").read_text(encoding="utf-8")
    assert "Contract / Smoke Validation Only" in md_text


def test_full_mode_fails_closed_not_configured_on_empty_root(tmp_path: Path) -> None:
    adapter = _FullMockAdapter()
    rep = run_benchmark(adapter, suites=ALL_SUITES, mode="full", root=tmp_path)
    assert rep["evaluation_scope"] == "formal_full_cohort"
    assert rep["suites"]["validate"]["status"] == "ok"
    for suite in ALL_SUITES:
        if suite == "validate":
            continue
        sres = rep["suites"][suite]
        assert sres["status"] == "not_configured", f"{suite} must fail closed on missing cohort, got {sres}"
        assert sres["evaluation_scope"] == "formal_full_cohort"
        assert len(sres["missing_paths"]) >= 1
        assert "reference_comparison" not in sres


def test_full_mode_executes_real_schema_cohort_fixtures_and_attaches_ranks(tmp_path: Path) -> None:
    # Build minimal real-schema fixtures for R2 MegaScale, R3 Epistasis, R4 Family Memory, and R5 Context
    wt = "MKTAYIAKQRQISFVK"

    # R2 MegaScale cohort
    p_r2 = tmp_path / R2_MEGASCALE_COHORT_REL
    p_r2.parent.mkdir(parents=True, exist_ok=True)
    p_r2.write_text(
        json.dumps(
            {
                "backgrounds": [
                    {
                        "name": "1TEST.pdb",
                        "group": "g0",
                        "wildtype": wt,
                        "variants": [
                            {"position": 1, "mutant": "A", "ddg": -0.5},
                            {"position": 2, "mutant": "E", "ddg": -0.1},
                            {"position": 3, "mutant": "S", "ddg": 0.2},
                            {"position": 4, "mutant": "G", "ddg": 0.6},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # R3 Epistasis extraction_plan.json with measured cycle targets
    p_r3 = tmp_path / R3_EPISTASIS_PLAN_REL
    p_r3.parent.mkdir(parents=True, exist_ok=True)
    s_a = "AKTAYIAKQRQISFVK"
    s_b = "METAYIAKQRQISFVK"
    s_ab = "AETAYIAKQRQISFVK"
    s_c = "MKSAYIAKQRQISFVK"
    s_ac = "AKSAYIAKQRQISFVK"
    s_bc = "MESAYIAKQRQISFVK"
    p_r3.write_text(
        json.dumps(
            {
                "backgrounds": [
                    {
                        "name": "1TEST.pdb",
                        "group": "g0",
                        "wildtype": wt,
                        "sequences": [wt, s_a, s_b, s_ab, s_c, s_ac, s_bc],
                        "cycles": [
                            {"positions": [0, 1], "states": [0, 1, 2, 3], "epistasis": 0.1},
                            {"positions": [0, 2], "states": [0, 1, 4, 5], "epistasis": 0.2},
                            {"positions": [1, 2], "states": [0, 2, 4, 6], "epistasis": 0.3},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # R4 Family Memory conditional-inputs.json
    p_r4 = tmp_path / R4_FAMILY_COND_REL
    p_r4.parent.mkdir(parents=True, exist_ok=True)
    p_r4.write_text(
        json.dumps(
            {
                "units": [
                    {
                        "unit_id": "u000",
                        "partition": "evaluation",
                        "sites": [
                            {
                                "query_position": 2,
                                "contexts": [{"native": wt, "native_index": 2, "observed": wt[2]}],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # R5 Context cohort.json (real schema consumed by r4_scoring_yardstick.build_unit_condition_inputs)
    p_r5 = tmp_path / R5_CONTEXT_COHORT_REL
    p_r5.parent.mkdir(parents=True, exist_ok=True)
    p_r5.write_text(
        json.dumps(
            {
                "protein": {
                    "records": [wt, "MKTAYIAKQRQISFVR", "GGGGPPPPWWWWCCCC", "YYYYFFFFHHHHDDDD"],
                    "related": {"0": [1]},
                    "filler": 3,
                    "units": [
                        {
                            "key": "u0",
                            "stratum": "retained",
                            "band": "id_30_50",
                            "target": 0,
                            "partners": [1],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    adapter = _FullMockAdapter()
    rep = run_benchmark(
        adapter,
        suites=["r2_biophysical", "r3_epistasis", "r4_family_memory", "r5_context"],
        mode="full",
        root=tmp_path,
    )
    for suite in ("r2_biophysical", "r3_epistasis", "r4_family_memory", "r5_context"):
        sres = rep["suites"][suite]
        assert sres["status"] == "ok", f"{suite} failed: {sres}"
        assert sres["evaluation_scope"] == "formal_full_cohort"
        assert "reference_comparison" in sres and sres["reference_comparison"]["rank"] is not None


def test_dynamic_custom_plugin_cli_end_to_end_and_sha256(tmp_path: Path) -> None:
    plugin_file = tmp_path / "custom_model.py"
    plugin_file.write_text(
        """
from __future__ import annotations
import numpy as np
from src.capability.benchmark import (
    BaseProteinBenchmarkAdapter,
    CapabilityFlags,
    ModelMetadata,
    pack_mid_and_last_hidden_states,
)
from src.capability.core.amino_acids import AA20

class MyModel(BaseProteinBenchmarkAdapter):
    def __init__(self) -> None:
        super().__init__(
            ModelMetadata(
                model_id="custom-demo-v1",
                display_name="CustomDemo-10M",
                paradigm="masked_lm",
                modality="protein",
                n_params_millions=10.0,
                max_positions=512,
            ),
            CapabilityFlags(
                score_mutations=True,
                site_conditional_logprobs=True,
                score_sequence_nll=True,
                extract_representations_1024=True,
                generate_unconditional=True,
                infill_local=True,
                sample_family_guided=True,
                continue_from_prefix=True,
            ),
        )

    def site_conditional_logprobs(self, sequence, positions=None, *, coords=None):
        pos_list = list(range(len(sequence))) if positions is None else [int(p) for p in positions]
        out = np.empty((len(pos_list), 20), dtype=np.float32)
        for i, p in enumerate(pos_list):
            left = ord(sequence[p - 1]) if p > 0 else 65
            right = ord(sequence[p + 1]) if p + 1 < len(sequence) else 65
            logits = np.linspace(-0.5, 0.5, 20, dtype=np.float32) + 0.01 * ((left + right) % 7)
            out[i] = logits - np.logaddexp.reduce(logits)
        return out

    def score_mutations(self, wildtype, mutations, *, coords=None):
        return np.array([0.1 * (i + 1) for i in range(len(mutations))], dtype=np.float64)

    def score_sequence_nll(self, sequence, *, context_items=(), coords=None, context_coords=()):
        ctx = "".join(context_items)
        bonus = 0.25 if (ctx and ctx[:4] == sequence[:4]) else 0.0
        nll_per = max(0.5, 2.5 - bonus)
        return {"nll_sum": float(nll_per * len(sequence)), "nll_per_residue": float(nll_per)}

    def extract_representations_1024(self, sequences, *, coords_list=None):
        rows = [pack_mid_and_last_hidden_states(np.ones((len(s), 32)), np.ones((len(s), 32)) * 2.0) for s in sequences]
        return np.stack(rows, axis=0)

    def generate_unconditional(self, lengths, *, seed=20261124):
        rng = np.random.default_rng(int(seed))
        return ["".join(AA20[int(k)] for k in rng.integers(0, 20, size=int(L))) for L in lengths]

    def infill_local(self, template, mask_positions, *, coords=None, seed=20261124):
        rng = np.random.default_rng(int(seed))
        chars = list(template)
        for p in mask_positions:
            chars[int(p)] = AA20[int(rng.integers(0, 20))]
        return "".join(chars)

    def sample_family_guided(self, length, *, n_samples, profile_log_bias=None, guidance_scale=0.0, coords=None, seed=20261124):
        rng = np.random.default_rng(int(seed))
        return ["".join(AA20[int(k)] for k in rng.integers(0, 20, size=int(length))) for _ in range(int(n_samples))]

    def continue_from_prefix(self, context_items, target_length, *, coords=None, seed=20261124):
        rng = np.random.default_rng(int(seed))
        return "".join(AA20[int(k)] for k in rng.integers(0, 20, size=int(target_length)))
""",
        encoding="utf-8",
    )

    loaded = load_custom_adapter(f"{plugin_file}:MyModel")
    assert loaded.metadata.model_id == "custom-demo-v1"

    out_dir = tmp_path / "bench_out"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/capability/unified/run_capability_benchmark.py"),
        "--adapter",
        f"{plugin_file}:MyModel",
        "--mode",
        "smoke",
        "--suites",
        "all",
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    summary_stdout = json.loads(proc.stdout)
    assert summary_stdout["model_id"] == "custom-demo-v1"
    assert summary_stdout["suites"]["validate"] == "ok"
    assert all(summary_stdout["suites"][s] == "smoke_ok" for s in ALL_SUITES if s != "validate")

    report_path = out_dir / "custom-demo-v1_benchmark_report.json"
    sha_path = out_dir / "custom-demo-v1_benchmark_report.json.sha256"
    md_path = out_dir / "custom-demo-v1_benchmark_summary.md"
    assert report_path.is_file() and sha_path.is_file() and md_path.is_file()

    expected_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert sha_path.read_text(encoding="utf-8").strip().startswith(expected_sha)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert all("reference_comparison" not in report["suites"][s] for s in ALL_SUITES)
    assert "Contract / Smoke Validation Only" in md_path.read_text(encoding="utf-8")


def test_partial_capability_adapter_reports_not_supported_honestly() -> None:
    rep = run_benchmark(_MinimalScoringOnlyAdapter(), suites=ALL_SUITES, mode="smoke")
    assert rep["suites"]["validate"]["status"] == "ok"
    assert rep["suites"]["r1_dms"]["status"] == "smoke_ok"
    assert rep["suites"]["r3_contacts"]["status"] == "not_supported"
    assert rep["suites"]["r4_generation"]["status"] == "not_supported"
    assert rep["suites"]["r5_context"]["status"] == "not_supported"
