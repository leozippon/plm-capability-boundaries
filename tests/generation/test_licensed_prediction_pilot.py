"""Contract tests for the licensed prediction pilot caller: protocol, calibration, ledger.

Every fixture here is a pure fake model/tokenizer/score callback. No real arm is
loaded, no GPU is touched and no scientific data is created. The calibration unit
tests run entirely on a synthetic complete ``licensed_native_calibration_v1``
fixture; exactly one optional integration test reads the frozen native
calibration asset when it is present and is skipped otherwise.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCRIPT = REPO / "scripts/capability/generation/collect_licensed_prediction_pilot.py"
CALIBRATION = REPO / "data/r6_licensed_prediction_pilot_20261013/native-calibration.json"
PREFIX = "MK"


def _module():
    spec = importlib.util.spec_from_file_location("licensed_pilot_collect", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cp = _module()


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _wt(length: int, *, first: str = "M") -> str:
    return first * 8 + "A" * (length - 8)


def _protocol(*, domains=None, seeds=None, models=None, length_shift=0):
    domain_keys = domains or {"1UFM.pdb": 72, "3L1X.pdb": 69}
    targets = {}
    for key, length in domain_keys.items():
        sequence = _wt(length + length_shift)
        targets[key] = {
            "wt_sequence": sequence,
            "wt_sha256": _sha(sequence),
            "L": len(sequence),
            "group": f"group-{key}",
        }
    return {
        "schema": cp.SCHEMA,
        "models": models or ["protgpt2", "progen2-medium"],
        "seed_streams": seeds or [610130, 710130, 810130],
        "sampling": dict(cp.SETTLED_SAMPLING),
        "native_contract": {
            "batch_size": 1,
            "budget": 1024,
            "dtype": "float32",
            "score": "raw_fullsum_residue_span_nats",
        },
        "targets": targets,
        "inputs": {
            "native_calibration": {"path": "cal.json", "sha256": "0" * 64},
        },
    }


def _synthetic_wt(length: int) -> str:
    return _wt(length, first="M")


def _single(sequence: str, position: int, mutant: str) -> str:
    return sequence[: position - 1] + mutant + sequence[position:]


def _synthetic_calibration():
    """A complete, tiny, unmeasured ``licensed_native_calibration_v1`` fixture.

    Both admitted arms carry both licensed domains (72 and 69 residues). Each
    target holds a poly-M/A wild type plus two distinct single substitutions,
    with full token ids, packed/scored/pooled counts and dummy provenance pins.
    """
    models = {}
    for arm in cp.ALLOWED_ARMS:
        targets = {}
        for domain, length in cp.LICENSED_TARGETS.items():
            wildtype = _synthetic_wt(length)
            sequences = [wildtype]
            positions = [0]
            mutants: list[str | None] = [None]
            for position, mutant in ((1, "A"), (9, "C")):
                sequences.append(_single(wildtype, position, mutant))
                positions.append(position)
                mutants.append(mutant)
            states = []
            for index, sequence in enumerate(sequences):
                ids = list(range(1, len(sequence) + 2))
                states.append(
                    {
                        "state": index,
                        "sequence": sequence,
                        "sequence_sha256": _sha(sequence),
                        "score_nats": float(100 + index),
                        "packed_token_count": len(ids),
                        "scored_token_count": len(sequence),
                        "pooled_token_count": len(sequence),
                        "token_ids": ids,
                        "position": positions[index],
                        "mutant": mutants[index],
                    }
                )
            targets[domain] = {
                "archive": {"path": f"archive/{arm}-{domain}.npz", "sha256": "a" * 64},
                "states": states,
            }
        models[arm] = {
            "identity": {
                "arm": arm,
                "batch_size": 1,
                "budget": 1024,
                "dtype": "float32",
                "likelihood_semantics": "raw_fullsum_residue_span_nats",
                "checkpoint_path": f"/pinned/{arm}",
                "checkpoint_metadata_sha256": {"config.json": "b" * 64},
                "checkpoint_tensor_files": [{"name": "pytorch_model.bin", "bytes": 1}],
            },
            "manifest": {"path": f"manifest/{arm}.json", "sha256": "c" * 64},
            "targets": targets,
        }
    return {
        "schema": cp.CALIBRATION_SCHEMA,
        "tolerance_nats": cp.SETTLED_TOLERANCE_NATS,
        "disclosure": "synthetic test fixture; no measured weights",
        "selection": "synthetic WT plus two single substitutions per licensed target",
        "sources": {"cohort": {"path": "synthetic/cohort.json", "sha256": "d" * 64}},
        "models": models,
    }


class FakeTokenizer:
    def __init__(self, table, eos_token_id=999):
        self.table = table
        self.eos_token_id = eos_token_id

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self.table[int(token)] for token in ids)


class Scripted:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.index = 0

    def __call__(self, ids):
        token = self.tokens[self.index]
        self.index += 1
        return token


def _attempt(module, tokenizer, tokens, *, arm_name="protgpt2", design_length=5,
             max_new_tokens=8, prefix=PREFIX):
    return module.sample_attempt(
        arm_name=arm_name,
        tokenizer=tokenizer,
        prompt_ids=[1],
        eos_id=tokenizer.eos_token_id,
        prefix_residues=prefix,
        design_length=design_length,
        max_new_tokens=max_new_tokens,
        seed=610130,
        attempt_index=0,
        next_id_fn=Scripted(tokens),
        wt_sequence=prefix + "A" * (design_length - len(prefix)),
        device="cpu",
    )


class ProtocolContract(unittest.TestCase):
    def test_valid_licensed_pair_through_real_validate_protocol(self):
        resolved = cp.validate_protocol(_protocol(), "protgpt2", "1UFM.pdb", 0)
        self.assertEqual(resolved["target"]["L"], 72)
        self.assertEqual(resolved["stream_seed"], 610130)
        self.assertEqual(resolved["sampling"], cp.SETTLED_SAMPLING)

    def test_generic_or_extra_domain_and_bad_models_are_refused(self):
        with self.assertRaisesRegex(ValueError, "licensed domain"):
            cp.validate_protocol(
                _protocol(domains={"1UFM.pdb": 72, "3L1X.pdb": 69, "5XYZ.pdb": 53}),
                "protgpt2", "1UFM.pdb", 0,
            )
        with self.assertRaisesRegex(ValueError, "licensed domain"):
            cp.validate_protocol(
                _protocol(domains={"1UFM72.pdb": 72, "3L1X.pdb": 69}),
                "protgpt2", "1UFM.pdb", 0,
            )
        with self.assertRaisesRegex(ValueError, "licensed length"):
            cp.validate_protocol(_protocol(length_shift=5), "protgpt2", "1UFM.pdb", 0)
        with self.assertRaisesRegex(ValueError, "list of the two"):
            cp.validate_protocol(_protocol(models={"protgpt2": {}, "progen2-medium": {}}),
                                 "protgpt2", "1UFM.pdb", 0)

    def test_seed_streams_valid_and_overlap_rejected(self):
        seeds = [610130, 710130, 810130]
        hard = cp.SETTLED_SAMPLING["hard_attempt_cap"]
        for left, right in zip(seeds[:-1], seeds[1:], strict=True):
            self.assertLess(left + hard, right)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            cp.validate_protocol(_protocol(seeds=[610130, 610200, 810130]),
                                 "protgpt2", "1UFM.pdb", 0)
        # Every attempt seed of one stream stays inside its own 10k range.
        for base in seeds:
            self.assertLess(base + hard - 1, base + 10000)


class NativeText(unittest.TestCase):
    def test_mixed_aa_lf_tokens_are_not_clipped_and_mapped(self):
        cleaned, count, bad, removed = cp.analyze_native_text("protgpt2", "AD\nC")
        self.assertEqual((cleaned, count, bad, removed), ("ADC", 3, "", [2]))
        cleaned, count, bad, removed = cp.analyze_native_text("progen2-medium", "AD\nC")
        self.assertEqual(cleaned, "AD\nC")
        self.assertEqual(removed, [])
        self.assertEqual(bad, "\n")
        cleaned, count, bad, _ = cp.analyze_native_text("protgpt2", "A?D")
        self.assertEqual(bad, "?")
        self.assertEqual(count, 3)


class AttemptLedger(unittest.TestCase):
    def test_every_status_is_retained_with_raw_token_path_and_exact_prefix(self):
        cases = [
            ("early_eos", FakeTokenizer({10: "A", 999: "<eos>"}), [999], 5, "early_eos"),
            ("noncanonical", FakeTokenizer({10: "A?"}), [10], 5, "noncanonical"),
            ("overshoot", FakeTokenizer({10: "A", 11: "DC"}), [10, 11], 4, "overshoot"),
            ("budget", FakeTokenizer({10: "A"}), [10] * 8, 12, "budget_exhausted"),
        ]
        for name, tokenizer, tokens, length, expected in cases:
            with self.subTest(name=name):
                row = _attempt(cp, tokenizer, tokens, design_length=length)
                self.assertEqual(row["status"], expected)
                self.assertFalse(row["is_eligible"])
                self.assertIsNone(row["candidate_sequence"])
                self.assertEqual(row["raw_token_ids"], tokens)
                self.assertTrue(row["raw_text"])

    def test_design_length_stop_is_not_native_completion_and_keeps_prefix(self):
        tokenizer = FakeTokenizer({10: "AD\n", 11: "C"})
        row = _attempt(cp, tokenizer, [10, 11], design_length=5)
        self.assertEqual(row["status"], "design_length_stop")
        self.assertTrue(row["is_eligible"])
        self.assertEqual(row["candidate_sequence"], "MKADC")
        self.assertEqual(row["raw_text"], "AD\nC")
        self.assertEqual(row["removed_lf_offsets"], [2])
        self.assertTrue(row["prefix_exact"])
        self.assertNotIn("EOS", row["raw_text"])

    def test_lf_only_text_is_rejected_for_the_non_lf_arm(self):
        tokenizer = FakeTokenizer({10: "AD\n"})
        row = _attempt(cp, tokenizer, [10], arm_name="progen2-medium", design_length=4)
        self.assertEqual(row["status"], "noncanonical")
        self.assertEqual(row["noncanonical_characters"], "\n")


class CandidateGrid(unittest.TestCase):
    @staticmethod
    def _tokens(sequence):
        return {
            "ids": list(range(1, len(sequence) + 1)),
            "scored_span": (1, len(sequence)),
            "pooled_positions": list(range(1, len(sequence))),
        }

    @staticmethod
    def _score(sequence):
        return float(sum((index + 1) * ord(residue) for index, residue in enumerate(sequence)))

    def _grid(self, sequence="MKAAAA"):
        return cp.candidate_grid(
            sequence,
            prefix_residues=PREFIX,
            design_length=6,
            token_record_fn=self._tokens,
            score_fn=self._score,
        )

    def test_suffix_positions_current_residue_zero_and_ragged_counts(self):
        sequence = "MKAAAA"
        grid = self._grid(sequence)
        self.assertEqual(grid["n_positions"], 4)
        self.assertEqual(grid["state_count"], 1 + 19 * 4)
        self.assertEqual(int(grid["state_position"][0]), 0)
        self.assertEqual(int(grid["state_mutant_index"][0]), -1)
        positions = [int(value) for value in grid["state_position"][1:]]
        self.assertEqual(sorted(set(positions)), [3, 4, 5, 6])
        self.assertEqual(positions, sorted(positions))
        for state in range(1, grid["state_count"]):
            position = int(grid["state_position"][state])
            self.assertNotEqual(str(grid["state_residue"][state]), sequence[position - 1])
        self.assertAlmostEqual(
            float(grid["delta"][0]), float(grid["mutant_scores"][0]) - float(grid["base_score"])
        )
        offsets = [int(value) for value in grid["token_offsets"]]
        self.assertEqual(offsets[0], 0)
        self.assertEqual(offsets[-1], len(grid["token_ids"]))
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual(len(grid["state_sequence_sha256"]), grid["state_count"])
        self.assertTrue(np.isfinite(grid["mutant_scores"]).all())

        # A multi-residue generated suffix (C/D/W currents) proves the canonical
        # AA20 index, the exact sequence SHA and the mutant delta of every state.
        multiresidue = "MKCDWA"
        grid = self._grid(multiresidue)
        self.assertEqual(str(grid["state_residue"][0]), "-")
        self.assertEqual(int(grid["state_mutant_index"][0]), -1)
        self.assertEqual(str(grid["state_sequence_sha256"][0]), _sha(multiresidue))
        seen: dict[int, list[str]] = {}
        for state in range(1, grid["state_count"]):
            position = int(grid["state_position"][state])
            residue = str(grid["state_residue"][state])
            self.assertEqual(int(grid["state_mutant_index"][state]), cp.AA20.index(residue))
            mutant = multiresidue[: position - 1] + residue + multiresidue[position:]
            self.assertEqual(str(grid["state_sequence_sha256"][state]), _sha(mutant))
            self.assertNotEqual(residue, multiresidue[position - 1])
            self.assertAlmostEqual(
                float(grid["delta"][state - 1]),
                float(grid["mutant_scores"][state - 1]) - float(grid["base_score"]),
            )
            seen.setdefault(position, []).append(residue)
        for position in range(len(PREFIX) + 1, len(multiresidue) + 1):
            self.assertEqual(
                sorted(seen[position]),
                sorted(set(cp.AA20) - {multiresidue[position - 1]}),
            )

    def test_nonfinite_and_prefix_defects_are_refused(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            cp.candidate_grid(
                "MKAAAA", prefix_residues=PREFIX, design_length=6,
                token_record_fn=self._tokens, score_fn=lambda sequence: float("inf"),
            )
        with self.assertRaisesRegex(ValueError, "conditioning prefix"):
            cp.candidate_grid(
                "MMAAAA", prefix_residues=PREFIX, design_length=6,
                token_record_fn=self._tokens, score_fn=self._score,
            )


    def test_duplicate_candidates_collapse_to_one_grid_with_all_attempts_kept(self):
        rows = [
            {"is_eligible": True, "candidate_sha256": "a" * 64, "candidate_sequence": "MKAAAA",
             "attempt_index": 0, "mutation_burden_vs_wt": 1, "novelty_vs_wt_hamming": 1},
            {"is_eligible": False, "candidate_sha256": None, "candidate_sequence": None,
             "attempt_index": 1},
            {"is_eligible": True, "candidate_sha256": "a" * 64, "candidate_sequence": "MKAAAA",
             "attempt_index": 2, "mutation_burden_vs_wt": 1, "novelty_vs_wt_hamming": 1},
            {"is_eligible": True, "candidate_sha256": "b" * 64, "candidate_sequence": "MKAAAC",
             "attempt_index": 3, "mutation_burden_vs_wt": 1, "novelty_vs_wt_hamming": 1},
        ]
        unique = cp.group_eligible_candidates(
            rows, arm="protgpt2", domain="1UFM.pdb", stream_seed=610130, stream_index=0
        )
        self.assertEqual([record["sha256"] for record in unique], ["a" * 64, "b" * 64])
        self.assertEqual(unique[0]["repeat_count"], 2)
        self.assertEqual(unique[0]["attempt_indexes"], [0, 2])
        self.assertEqual(unique[1]["repeat_count"], 1)


class NativeCalibration(unittest.TestCase):
    def setUp(self):
        self.payload = _synthetic_calibration()
        states = self.payload["models"]["protgpt2"]["targets"]["1UFM.pdb"]["states"]
        self.target = {
            "wt_sequence": states[0]["sequence"],
            "wt_sha256": states[0]["sequence_sha256"],
            "domain": "1UFM.pdb",
            "L": len(states[0]["sequence"]),
        }
        self.states = {state["sequence"]: state for state in states}
        self.contract = {"batch_size": 1, "budget": 1024, "dtype": "float32"}

    def _tokens(self, sequence):
        state = self.states[sequence]
        scored = state["scored_token_count"]
        return {
            "ids": state["token_ids"],
            "scored_span": (1, 1 + scored),
            "pooled_positions": list(range(state["pooled_token_count"])),
        }

    def _select(self):
        return cp.select_native_calibration(self.payload, "protgpt2", self.target, self.contract)

    def test_synthetic_fixture_is_complete_for_both_arms_and_targets(self):
        cp.validate_native_calibration_payload(self.payload)
        self.assertEqual(self.payload["schema"], cp.CALIBRATION_SCHEMA)
        self.assertEqual(float(self.payload["tolerance_nats"]), cp.SETTLED_TOLERANCE_NATS)
        for arm in cp.ALLOWED_ARMS:
            block = self.payload["models"][arm]
            self.assertEqual(block["identity"]["arm"], arm)
            self.assertEqual(block["identity"]["dtype"], "float32")
            self.assertEqual(sorted(block["targets"]), sorted(cp.LICENSED_TARGETS))
            for domain, length in cp.LICENSED_TARGETS.items():
                states = block["targets"][domain]["states"]
                self.assertEqual(len(states), 3)
                self.assertEqual(states[0]["sequence"], _synthetic_wt(length))
                self.assertIsNone(states[0]["mutant"])
                self.assertEqual(states[0]["position"], 0)
                for index, state in enumerate(states):
                    self.assertEqual(state["state"], index)
                    self.assertEqual(len(state["sequence"]), length)
                    self.assertEqual(len(state["token_ids"]), state["packed_token_count"])
                    self.assertEqual(state["packed_token_count"], length + 1)
                    self.assertEqual(state["scored_token_count"], length)
                    self.assertEqual(state["pooled_token_count"], length)
                    self.assertEqual(state["sequence_sha256"], _sha(state["sequence"]))
                    if index:
                        position, mutant = state["position"], state["mutant"]
                        self.assertEqual(_single(states[0]["sequence"], position, mutant), state["sequence"])

    def test_synthetic_blocks_select_wt_and_two_singles_by_wt_sha(self):
        selection = self._select()
        self.assertEqual(selection["calibration_target_key"], "1UFM.pdb")
        self.assertEqual([state["state"] for state in selection["states"]], [0, 1, 2])
        self.assertIsNone(selection["states"][0]["mutant"])
        self.assertEqual(selection["identity"]["arm"], "protgpt2")
        self.assertEqual(selection["manifest"]["path"], "manifest/protgpt2.json")
        self.assertEqual(selection["archive"]["path"], "archive/protgpt2-1UFM.pdb.npz")
        self.assertEqual(selection["cohort_sha256"], "d" * 64)

    def test_full_layout_and_tolerance_pass_and_single_sign_fail(self):
        selection = self._select()
        cached = {state["sequence"]: state["score_nats"] for state in selection["states"]}
        passed = cp.native_calibration_check(
            selection,
            score_fn=lambda sequence: cached[sequence] + 0.0005,
            token_record_fn=self._tokens,
        )
        self.assertEqual(len(passed["entries"]), 3)
        for direction in (0.002, -0.002):
            with self.assertRaisesRegex(ValueError, "tolerance"):
                cp.native_calibration_check(
                    selection,
                    score_fn=lambda sequence, d=direction: cached[sequence] + d,
                    token_record_fn=self._tokens,
                )

    def test_token_count_hash_and_nonfinite_refusals(self):
        selection = self._select()
        cached = {state["sequence"]: state["score_nats"] for state in selection["states"]}
        broken = dict(self.states)
        first = self.target["wt_sequence"]
        broken[first] = dict(broken[first])
        broken[first]["pooled_token_count"] = broken[first]["pooled_token_count"] - 1
        with self.assertRaisesRegex(ValueError, "pooled"):
            cp.native_calibration_check(
                selection,
                score_fn=lambda sequence: cached[sequence],
                token_record_fn=lambda sequence: {
                    "ids": broken[sequence]["token_ids"],
                    "scored_span": (1, 1 + broken[sequence]["scored_token_count"]),
                    "pooled_positions": list(range(broken[sequence]["pooled_token_count"])),
                },
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            cp.native_calibration_check(
                selection,
                score_fn=lambda sequence: float("nan"),
                token_record_fn=self._tokens,
            )
        with self.assertRaisesRegex(ValueError, "token ids"):
            cp.native_calibration_check(
                selection,
                score_fn=lambda sequence: cached[sequence],
                token_record_fn=lambda sequence: {
                    "ids": list(self.states[sequence]["token_ids"]) + [1],
                    "scored_span": (1, 1 + self.states[sequence]["scored_token_count"]),
                    "pooled_positions": list(range(self.states[sequence]["pooled_token_count"])),
                },
            )

    def test_schema_tolerance_and_wt_sha_bindings_are_enforced(self):
        tampered = copy.deepcopy(self.payload)
        tampered["tolerance_nats"] = 0.01
        with self.assertRaisesRegex(ValueError, "tolerance_nats"):
            cp.validate_native_calibration_payload(tampered)
        bad = copy.deepcopy(self.payload)
        bad["models"]["protgpt2"]["identity"]["dtype"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "dtype"):
            cp.select_native_calibration(bad, "protgpt2", self.target, self.contract)
        missing = copy.deepcopy(self.payload)
        missing["models"]["protgpt2"]["targets"]["1UFM.pdb"]["states"][0]["sequence_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "binds"):
            cp.select_native_calibration(missing, "protgpt2", self.target, self.contract)


@unittest.skipUnless(CALIBRATION.is_file(), "the frozen real calibration asset is absent")
class RealCalibrationAssetLayout(unittest.TestCase):
    """One optional read-only check of the frozen asset; no forward pass is run."""

    def test_two_arms_two_targets_all_twelve_state_layouts_bind(self):
        payload = cp.load_native_calibration(CALIBRATION)
        self.assertEqual(payload["schema"], cp.CALIBRATION_SCHEMA)
        self.assertEqual(float(payload["tolerance_nats"]), cp.SETTLED_TOLERANCE_NATS)
        contract = {"batch_size": 1, "budget": 1024, "dtype": "float32"}
        observed = 0
        for arm in cp.ALLOWED_ARMS:
            cohorts = payload["models"][arm]["targets"]
            self.assertEqual(sorted(cohorts), sorted(cp.LICENSED_TARGETS))
            for domain, length in cp.LICENSED_TARGETS.items():
                state0 = cohorts[domain]["states"][0]
                target = {
                    "wt_sequence": state0["sequence"],
                    "wt_sha256": state0["sequence_sha256"],
                    "domain": domain,
                    "L": length,
                }
                selection = cp.select_native_calibration(payload, arm, target, contract)
                self.assertEqual(selection["calibration_target_key"], domain)
                self.assertEqual(selection["identity"]["arm"], arm)
                self.assertEqual(len(selection["states"]), 3)
                for index, parsed in enumerate(selection["states"]):
                    self.assertEqual(parsed["state"], index)
                    self.assertEqual(len(parsed["sequence"]), length)
                    self.assertEqual(len(parsed["token_ids"]), parsed["packed_token_count"])
                    self.assertEqual(parsed["sequence_sha256"], _sha(parsed["sequence"]))
                    if index == 0:
                        self.assertIsNone(parsed["mutant"])
                        self.assertEqual(parsed["position"], 0)
                    else:
                        self.assertEqual(len(parsed["mutant"]), 1)
                        self.assertGreaterEqual(parsed["position"], 1)
                    observed += 1
        self.assertEqual(observed, 12)


class OutputsAndAvailability(unittest.TestCase):
    def test_prepare_output_dir_and_npz_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            cp.prepare_output_dir(out)
            (out / "existing.txt").write_text("x")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                cp.prepare_output_dir(out)
            grid = Path(temporary) / "grid.npz"
            cp.write_immutable_npz(grid, {"value": np.asarray([1.0])})
            with self.assertRaisesRegex(ValueError, "overwrite"):
                cp.write_immutable_npz(grid, {"value": np.asarray([2.0])})

    def test_resolve_inputs_hash_and_root_escape_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "cal.json"
            payload.write_text('{"schema": "licensed_native_calibration_v1"}')
            protocol = {"inputs": {"native_calibration": {"path": "cal.json", "sha256": _sha(payload.read_text())}}}
            resolved = cp.resolve_inputs(protocol, root)
            self.assertEqual(resolved["native_calibration"]["sha256"], _sha(payload.read_text()))
            bad = {"inputs": {"native_calibration": {"path": "cal.json", "sha256": "0" * 64}}}
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                cp.resolve_inputs(bad, root)
            escape = {"inputs": {"native_calibration": {"path": "../cal.json", "sha256": "0" * 64}}}
            with self.assertRaisesRegex(ValueError, "escapes"):
                cp.resolve_inputs(escape, root)

    def test_checkpoint_metadata_and_tensor_binding_refuses_mismatch(self):
        identity = {
            "checkpoint_metadata_sha256": {"config.json": "aa"},
            "checkpoint_tensor_files": [{"name": "model.safetensors", "bytes": 10}],
        }
        good = {
            "metadata_sha256": {"config.json": "aa"},
            "tensor_files": [{"name": "model.safetensors", "bytes": 10}],
        }
        cp.require_checkpoint_binding(good, identity)
        bad = {
            "metadata_sha256": {"config.json": "bb"},
            "tensor_files": [{"name": "model.safetensors", "bytes": 10}],
        }
        with self.assertRaisesRegex(ValueError, "config.json"):
            cp.require_checkpoint_binding(bad, identity)
        with self.assertRaisesRegex(ValueError, "model.safetensors"):
            cp.require_checkpoint_binding(
                {"metadata_sha256": {"config.json": "aa"}, "tensor_files": []}, identity
            )

    def test_zero_eligible_is_explicitly_unavailable_not_negative(self):
        available, reason = cp.analysis_availability(0)
        self.assertFalse(available)
        self.assertIn("unavailable", reason)
        self.assertIn("not a biological negative", reason)
        self.assertEqual(cp.analysis_availability(4), (True, None))


if __name__ == "__main__":
    unittest.main()