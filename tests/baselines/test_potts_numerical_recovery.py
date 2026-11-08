"""Real small PLM runs, with compile-time-only failure injection for lifecycle tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capability.baselines import recover_potts_fit as R  # noqa: E402

VENDOR = ROOT / "data/pairwise_assets/plmc/debbiemarkslab-plmc-18c9e55"


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    before = {str(p): R.digest(p) for p in VENDOR.rglob("*") if p.is_file()}
    root = tmp_path_factory.mktemp("recovery-build")
    for name, testing in (("production", False), ("testing", True)):
        R.build(VENDOR, root / name, testing=testing)
    yield root
    R.check_files(before)


def run_fixture(tmp_path, builds, *, cap=5000, testing=False, **injection):
    out = tmp_path / "run"
    out.mkdir()
    fasta = out / "full.fa"
    weights = out / "full.weights"
    # Same zero-weight focus convention and genuine gapped observations.
    fasta.write_text(">focus\nACD\n>h0\nACD\n>h1\nCCD\n>h2\nA-E\n>h3\nC-D\n")
    weights.write_text("0.000000\n1.000000\n0.750000\n0.500000\n0.250000\n")
    binary = builds / ("testing" if testing else "production") / "plmc-recovery"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PLMC_TEST_")}
    env.update(OMP_NUM_THREADS="4", OMP_DYNAMIC="FALSE", CUDA_VISIBLE_DEVICES="", **injection)
    cmd = [str(binary), "-o", str(out / "final.params"), "-f", "focus", "-le", "80",
           "-lh", ".01", "-m", str(cap), "-g", "-n", "4", "-w", str(weights), str(fasta)]
    p = subprocess.run(cmd, cwd=out, env=env, capture_output=True, text=True, timeout=30)
    (out / "fit.log").write_text(p.stdout + p.stderr)
    events = R.read_events(p.stdout + p.stderr)
    receipt = R.classify(p.returncode, events, cap)
    return out, events, receipt


def doubles(path):
    raw = path.read_bytes()
    assert raw[:8] == b"PLM64LE\n"
    length, q, n = struct.unpack_from("<iii", raw, 8)
    x = np.frombuffer(raw, dtype="<f8", offset=20)
    assert len(x) == n == length * q + length * (length - 1) // 2 * q * q
    return length, q, x


def independent_objective_gradient(x, length, q):
    """Direct scalar reference in vendor internal layout, not patched C code."""
    rows = np.array([[0, 1, 2], [1, 1, 2], [0, -1, 3], [1, -1, 2]])
    weights = [1., .75, .5, .25]
    g = np.zeros_like(x)
    lam = np.full(len(x), 80.)
    lam[:length * q] = .01
    f = np.sum(lam * x * x)
    g += 2 * lam * x

    def ij(i, j, a, b):
        if i > j:
            i, j, a, b = j, i, b, a
        return length * q + (j * (j - 1) // 2 + i) * q * q + b * q + a

    for row, w in zip(rows, weights, strict=True):
        for i in range(length):
            if row[i] < 0:
                continue
            potential = np.array([x[i + length * a] for a in range(q)])
            for j in range(length):
                if j != i and row[j] >= 0:
                    potential += [x[ij(i, j, a, row[j])] for a in range(q)]
            scale = potential.max()
            probability = np.exp(potential - scale)
            z = probability.sum()
            probability /= z
            f += w * (scale + np.log(z) - potential[row[i]])
            residual = w * probability
            residual[row[i]] -= w
            for a in range(q):
                g[i + length * a] += residual[a]
                for j in range(length):
                    if j != i and row[j] >= 0:
                        g[ij(i, j, a, row[j])] += residual[a]
    return f, g


def test_real_success_double_acceptance_and_separate_float_export(tmp_path, builds):
    out, events, receipt = run_fixture(tmp_path, builds)
    assert receipt["scientific_success"] and receipt["history_resets"] == 0
    assert not list(out.glob("failure-*.double"))
    length, q, x = doubles(out / "final.double")
    assert np.any(x != x.astype(np.float32).astype(np.float64))
    f, g = independent_objective_gradient(x, length, q)
    final = receipt["true_final_event"]
    assert f == pytest.approx(float(final["objective"]), abs=1e-10)
    assert np.linalg.norm(g) == pytest.approx(float(final["gnorm"]), abs=1e-11)
    assert np.linalg.norm(g) / max(1., np.linalg.norm(x)) <= .001
    # Verify exported h/J are rounding of final.double, not the acceptance point.
    raw = (out / "final.params").read_bytes()
    nsite, q2, nseq, skipped, _ = struct.unpack_from("<5i", raw)
    assert (nsite, q2, nseq, skipped) == (length, q, 5, 0)
    offset = 40 + q + 4 * nseq + length + 4 * length + 4 * length * q
    h = np.frombuffer(raw, dtype="<f4", count=length * q, offset=offset)
    assert np.array_equal(h.reshape(length, q), x[:length*q].reshape(q, length).T.astype(np.float32))
    offset += 4 * length * q + 4 * (len(x) - length * q)
    couplings = np.frombuffer(raw, dtype="<f4", offset=offset).reshape(-1, q, q)
    for pair, (i, j) in enumerate((i, j) for i in range(length) for j in range(i + 1, length)):
        internal_pair = j * (j - 1) // 2 + i
        start = length * q + internal_pair * q * q
        assert np.array_equal(couplings[pair], x[start:start + q*q].reshape(q, q).T.astype(np.float32))
    protocol = next(e for e in events if e["event"] == "protocol")
    assert {k: protocol[k] for k in ("m", "past", "linesearch", "max_linesearch")} == {
        "m": "6", "past": "0", "linesearch": "0", "max_linesearch": "40"}
    for key, value in (("epsilon", .001), ("ftol", 1e-4), ("gtol", .9), ("xtol", 1e-16)):
        assert float(protocol[key]) == value


@pytest.mark.parametrize("errors,cap,restarts,success", [
    ("-1001", 5000, 1, True),
    ("-1001,-1001", 5000, 1, False),
    ("-1001", 2, 0, False),
    ("-1001", 3, 1, False),
    ("-998", 5000, 0, False),
])
def test_only_one_rounding_restart_with_shared_budget(tmp_path, builds, errors, cap, restarts, success):
    out, events, receipt = run_fixture(tmp_path, builds, cap=cap, testing=True, PLMC_TEST_ERRORS=errors)
    assert receipt["scientific_success"] is success
    assert receipt["history_resets"] == restarts
    ends = [e for e in events if e["event"] == "episode_end"]
    assert int(ends[-1]["cumulative"]) == sum(int(e["attempted"]) for e in ends) <= cap
    assert int(ends[0]["attempted"]) == 2  # includes the failed second search
    assert (out / "failure-episode-0.double").exists()
    if not success:
        assert receipt["process_returncode"] != 0
        assert not (out / "final.params").exists()
    if restarts:
        starts = [e for e in events if e["event"] == "episode_start"]
        assert int(starts[1]["remaining"]) == cap - 2


def test_failed_search_restores_exact_unquantized_accepted_point(tmp_path, builds):
    first = tmp_path / "first"
    first.mkdir()
    out, _, result = run_fixture(first, builds, cap=1)
    assert not result["scientific_success"]
    x = doubles(out / "final.double")[2]
    second = tmp_path / "second"
    second.mkdir()
    out, _, _ = run_fixture(second, builds, cap=2, testing=True, PLMC_TEST_ERRORS="-1001")
    np.testing.assert_array_equal(doubles(out / "failure-episode-0.double")[2], x)
    np.testing.assert_array_equal(doubles(out / "final.double")[2], x)


@pytest.mark.parametrize("verify", [False, True])
def test_nonfinite_never_restarts_or_exports(tmp_path, builds, verify):
    if verify:
        out, _, result = run_fixture(tmp_path, builds, testing=True, PLMC_TEST_VERIFY_NONFINITE="1")
    else:
        out, _, result = run_fixture(tmp_path, builds, testing=True, PLMC_TEST_NONFINITE="1")
    assert not result["scientific_success"] and result["process_returncode"] != 0
    assert result["history_resets"] == 0 and not (out / "final.params").exists()
    assert (out / "final.double").exists()


def test_production_ignores_injection_and_test_build_cannot_prepare(tmp_path, builds):
    _, _, result = run_fixture(tmp_path, builds, PLMC_TEST_ERRORS="-1001,-1001", PLMC_TEST_NONFINITE="1")
    assert result["scientific_success"] and result["history_resets"] == 0
    with pytest.raises(ValueError, match="test-injection"):
        R.verify_build(builds / "testing")


def test_zero_budget_refused_before_lbfgs(tmp_path, builds):
    out, events, result = run_fixture(tmp_path, builds, cap=0)
    assert not result["scientific_success"] and result["process_returncode"] != 0
    assert not (out / "final.params").exists()
    assert events == [{"event": "unsupported_configuration"}]


def test_exit_zero_without_verified_success_is_failure():
    assert not R.classify(0, [], 5000)["scientific_success"]
    final = dict(event="final", return_code="-1001", objective="1", xnorm="1",
                 gnorm="0", ratio="0", accepted="1", unquantized="1", cumulative="1")
    assert not R.classify(0, [final], 5000)["scientific_success"]
    final.update(return_code="0", ratio="nan")
    assert not R.classify(0, [final], 5000)["scientific_success"]


def test_prepare_refuses_original_tree_and_overwrite(tmp_path, builds):
    with pytest.raises(ValueError, match="independent"):
        R.prepare(tmp_path, tmp_path / "record", builds / "production", tmp_path / "out")
    with pytest.raises(FileExistsError):
        R.build(VENDOR, builds / "production")
    data = json.loads((builds / "production/build.json").read_text())
    assert not data["testing"] and data["historical_compile_argv"] is None


def test_real_subprocess_receipt_is_independent_and_append_only(tmp_path, builds):
    if R.resources()["available_memory_bytes"] < R.MEMORY_LIMIT + (2 << 30):
        pytest.skip("10 GiB production memory admission unavailable")
    source, _, _ = run_fixture(tmp_path, builds)
    out = tmp_path / "potts-numerical-recovery" / "synthetic"
    out.mkdir(parents=True)
    for name in ("full.fa", "full.weights"):
        (out / name).write_bytes((source / name).read_bytes())
    original = {str(p): R.digest(p) for p in source.iterdir() if p.is_file()}
    producer = json.loads((builds / "production/build.json").read_text())
    R.save_new(out / "preparation.json", {
        "build_dir": str(builds / "production"), "producer": producer,
        "original_files": original, "original_status": "synthetic_fixture_not_scientific",
        "inputs": {p: R.digest(out / p) for p in ("full.fa", "full.weights")}})
    receipt = R.execute(out)
    assert receipt["scientific_success"] and receipt["original_bytes_unchanged"]
    assert receipt["score_assembly"] == "not_performed"
    assert receipt["status"] == "converged_independent_producer"
    assert receipt["episodes"] and "fit.log" in receipt["artifacts"]
    assert (out / "receipt.json").is_file()
    R.check_files(original)
    before = R.digest(out / "receipt.json")
    with pytest.raises(FileExistsError):
        R.execute(out)
    assert R.digest(out / "receipt.json") == before
