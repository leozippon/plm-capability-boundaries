"""Objective-tuned arms: the checkpoint-override defect, the pairing, every refusal.

The first class is the regression test for a defect that produced no error and no
wrong-looking record: ``load_generator`` resolved a ``--checkpoint`` override into
a path and then loaded five of its eight arms by *name* from the declared
``ArmSpec``, while ``write_interface_record`` reported the override. A fine-tuned
checkpoint passed that way generated from base weights and claimed otherwise,
which is exactly the contrast this experiment is built on. The test asserts the
path that reaches the loader, because that is the only place the two can be told
apart.

The tokenizer fakes below are small on purpose. The contracts they exercise are
about *token ids* -- a doubled beginning-of-sequence token, a residue that is two
pieces, a family token at a shifted index -- so a fake that encodes greedily over
a declared vocabulary reads the same contract a real tokenizer does, without
needing weights that a public checkout does not have. The two tests that must see
the real published tokenizers say so and skip when the checkpoints are absent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core import arms as A  # noqa: E402
from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.generation import objective_tuned_arms as ot  # noqa: E402
from src.capability.generation import unconditional_generation as ug  # noqa: E402

#: Every arm whose weights load through ``load_arm_spec`` and which must therefore
#: honour a checkpoint override. The five that silently ignored it are the panel
#: ProtGPT2 and ProGen2 rungs; ``protgpt3-1.3b`` loaded its declared path while the
#: record claimed the override.
ARM_SPEC_ARMS = (
    "protgpt2",
    "progen2-small",
    "progen2-base",
    "progen2-medium",
    "progen2-large",
    "progen2-xlarge",
    "protgpt3-1.3b",
    "protgpt3-112m",
    "protgpt3-112m-dpo",
    "protgpt3-1.3b-dpo",
    "progen2-small-hugohrban",
    "progen2-small-mix7",
)

NOT_OVERRIDABLE = ("galactica-125m", "galactica-30b", "instructprotein", "prollama", "prollama-stage-1")


def _module(relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(f"objective_tuned_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def stage():
    return _module("scripts/capability/stages/unconditional_generation.py")


def builder():
    return _module("scripts/capability/generation/build_objective_tuned_campaign.py")


class CharTokenizer:
    """A per-residue tokenizer: greedy longest-match over a declared vocabulary.

    Enough of the interface that ``rendering_marker_ids`` and the three contract
    checks read it exactly as they read a published one.
    """

    def __init__(self, vocab, *, bos=None, eos=None, unk=None, add_bos=False, specials=()):
        self.vocab = dict(vocab)
        self.inverse = {index: token for token, index in self.vocab.items()}
        self.bos_token_id = None if bos is None else self.vocab[bos]
        self.eos_token_id = None if eos is None else self.vocab[eos]
        self.unk_token_id = None if unk is None else self.vocab[unk]
        self.add_bos = add_bos
        self.all_special_tokens = list(specials)
        self.all_special_ids = [self.vocab[token] for token in specials if token in self.vocab]

    def __call__(self, text, add_special_tokens=True, **_kwargs):
        ids = [self.bos_token_id] if (self.add_bos and add_special_tokens) else []
        position = 0
        while position < len(text):
            for length in range(min(12, len(text) - position), 0, -1):
                piece = text[position : position + length]
                if piece in self.vocab:
                    ids.append(self.vocab[piece])
                    position += length
                    break
            else:
                raise AssertionError(f"{text[position]!r} is not in this fake vocabulary")
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.inverse[int(value)] for value in ids)

    def get_vocab(self):
        return dict(self.vocab)


def protgpt3_vocab():
    vocab = {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2, "[UNK]": 3, "1": 4, "2": 5}
    vocab.update({residue: 6 + index for index, residue in enumerate(AA20)})
    vocab["<unk>"] = 33
    return vocab


#: The ProtGPT3 family's published tokenizer, as the fake needs it spelled.
_FAKE = {
    "vocab": protgpt3_vocab(),
    "bos": "<|bos|>",
    "eos": "<|eos|>",
    "unk": "<unk>",
    "specials": ("<|pad|>", "<|bos|>", "<|eos|>", "<unk>"),
}


def protgpt3_tokenizer(**kwargs):
    return CharTokenizer(**{**_FAKE, **kwargs})


class TestLoaderOverride:
    """The checkpoint-override regression test: the path the loader actually saw."""

    def test_every_arm_spec_generator_loads_the_directory_it_was_given(self, monkeypatch, tmp_path):
        seen: dict[str, object] = {}

        def fake_load_arm_spec(spec, device="cuda:0", dtype="bfloat16", **_kwargs):
            seen["path"] = Path(spec.path)
            seen["name"] = spec.name
            seen["dtype"] = dtype
            return SimpleNamespace(model="model", tokenizer="tokenizer")

        def refuse(*_args, **_kwargs):
            raise AssertionError(
                "load_arm takes a name and reads the declared path; the override "
                "cannot reach it"
            )

        monkeypatch.setattr(A, "load_arm_spec", fake_load_arm_spec)
        monkeypatch.setattr(A, "load_arm", refuse)
        for arm in ARM_SPEC_ARMS:
            override = tmp_path / f"tuned-{arm}"
            seen.clear()
            assert ug.load_generator(arm, device="cpu", checkpoint=override) == (
                "model",
                "tokenizer",
            )
            assert seen["path"] == override, arm
            assert seen["name"] == arm, arm
            assert seen["dtype"] == ug.generation_point(arm).dtype, arm

    def test_no_override_keeps_the_declared_path(self, monkeypatch):
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            A,
            "load_arm_spec",
            lambda spec, device="cuda:0", dtype="bfloat16", **_kwargs: (
                seen.__setitem__("path", Path(spec.path)),
                SimpleNamespace(model="model", tokenizer="tokenizer"),
            )[1],
        )
        for arm in ("protgpt2", "protgpt3-1.3b-dpo"):
            ug.load_generator(arm, device="cpu")
            assert seen["path"] == ug.checkpoint_for_arm(arm)

    def test_the_generate_ledger_names_the_checkpoint_the_loader_is_given(self, tmp_path):
        """A pool must say which sibling produced it; the arm name does not say it.

        ``interface-only`` already recorded the checkpoint and ``generate`` did
        not, so a tuned pool and a base pool written under a ``--checkpoint``
        override were indistinguishable in the one record the campaign keeps.
        """

        for arm in ("protgpt3-112m", "protgpt3-112m-dpo"):
            out = tmp_path / arm
            ug.generate(
                arm,
                n=1,
                seed=20264011000,
                generate_fn=lambda **_kwargs: ["MKTA"],
                output_dir=out,
                checkpoint=ug.checkpoint_for_arm(arm),
            )
            summary = json.loads((out / ug.EXPECT_GENERATE).read_text(encoding="utf-8"))
            assert summary["checkpoint"] == str(ug.checkpoint_for_arm(arm))
        base = json.loads(
            (tmp_path / "protgpt3-112m" / ug.EXPECT_GENERATE).read_text(encoding="utf-8")
        )
        tuned = json.loads(
            (tmp_path / "protgpt3-112m-dpo" / ug.EXPECT_GENERATE).read_text(encoding="utf-8")
        )
        assert base["checkpoint"] != tuned["checkpoint"]

    def test_a_ledger_from_an_injected_generator_claims_no_checkpoint(self, tmp_path):
        """No weights were opened, so the record says so rather than naming a default."""

        ug.generate(
            "protgpt3-112m-dpo",
            n=1,
            generate_fn=lambda **_kwargs: ["MK"],
            output_dir=tmp_path,
        )
        summary = json.loads((tmp_path / ug.EXPECT_GENERATE).read_text(encoding="utf-8"))
        assert summary["checkpoint"] is None

    def test_the_interface_record_names_the_checkpoint_the_loader_is_given(self, tmp_path):
        override = tmp_path / "ProtGPT2-tuned"
        record = ug.write_interface_record(tmp_path / "out", "protgpt2", checkpoint=override)
        assert record["checkpoint"] == str(override)
        assert ug.interface_payload("protgpt2", checkpoint=override)["checkpoint"] == str(override)
        assert stage().resolve_checkpoint("protgpt2", override) == override

    @pytest.mark.parametrize("arm", NOT_OVERRIDABLE)
    def test_a_loader_that_cannot_be_redirected_refuses_rather_than_ignores(self, arm, tmp_path):
        with pytest.raises(ValueError, match="cannot be pointed at"):
            ug.load_generator(arm, device="cpu", checkpoint=tmp_path / "elsewhere")

    def test_the_declared_path_passed_explicitly_is_not_an_override(self, monkeypatch):
        import src.capability.models.galactica_fitness as galactica

        seen: dict[str, object] = {}
        monkeypatch.setattr(
            galactica,
            "load_galactica",
            lambda name, *, device, dtype: (
                seen.__setitem__("name", name),
                SimpleNamespace(model="model", tokenizer="tokenizer"),
            )[1],
        )
        declared = ug.checkpoint_for_arm("galactica-125m")
        assert ug.load_generator("galactica-125m", device="cpu", checkpoint=declared) == (
            "model",
            "tokenizer",
        )
        assert seen["name"] == "galactica-125m"


class TestPairDeclaration:
    """One base per tuned arm, one objective per pair, and nothing inside the panel."""

    def test_every_tuned_arm_resolves_one_base_and_one_objective(self):
        assert len(ot.PAIRS) == 5
        assert len({pair.tuned for pair in ot.PAIRS}) == 5
        for pair in ot.PAIRS:
            assert ot.base_of(pair.tuned) == pair.base
            assert ot.pair_for(pair.base) is pair
            assert ot.pair_for(pair.tuned) is pair
            assert pair.objective and pair.note
            assert pair.lane in ot.LANES
            with pytest.raises(KeyError, match="is the base of"):
                ot.base_of(pair.base)
            for name in pair.arms:
                assert ot.spec_for(name).name == name
        with pytest.raises(KeyError, match="not an objective-tuned arm"):
            ot.pair_for("gpt2-large")

    def test_the_declared_objectives_name_what_was_optimised(self):
        by_tuned = {pair.tuned: pair.objective for pair in ot.PAIRS}
        assert by_tuned["protgpt3-112m-dpo"] == by_tuned["protgpt3-1.3b-dpo"]
        assert "DPO" in by_tuned["protgpt3-112m-dpo"]
        assert "EC" in by_tuned["zymctrl"]
        assert "PF00042" in by_tuned["progen2-small-mix7"]
        assert [pair.secondary for pair in ot.PAIRS if pair.tuned == "progen2-small-mix7"] == [True]

    def test_no_objective_tuned_arm_leaks_into_the_panel_or_the_roster(self):
        from src.capability.interactions.pairwise_epistasis import ROSTER

        declared = set(ot.ARMS)
        assert len(ROSTER) == 33
        assert declared.isdisjoint(ROSTER)
        assert declared.isdisjoint(A.PANEL)
        assert declared.isdisjoint(A.STAGED_ARMS)
        assert declared.isdisjoint(A.PROGEN3_ARMS)
        for name in declared:
            with pytest.raises(KeyError):
                A.arm_spec(name)
            assert ot.spec_for(name).capabilities == frozenset()

    def test_the_counts_other_measurements_pin_are_unchanged(self):
        assert len(A.PANEL) == 15
        assert len(A.STAGED_ARMS) == 8
        assert len(A.PROGEN3_ARMS) == 2
        assert len(ug.ADMITTED_ARMS) == 17
        assert set(ug.OFF_PANEL).isdisjoint(ug.ADMITTED)
        assert set(ug.OFF_PANEL) == set(ot.OFF_PANEL_GENERATION_ARMS)
        # the pair members that are already declared stay exactly where they were
        assert {"protgpt2", "zymctrl"} <= set(A.PANEL)
        assert "protgpt3-1.3b" in A.STAGED_ARMS

    def test_the_conditioned_pair_is_declared_into_its_own_lane_and_refused_here(self):
        lanes = {pair.tuned: pair.lane for pair in ot.PAIRS}
        assert lanes["zymctrl"] == ot.LANE_EC_CONDITIONED
        assert {pair.tuned for pair in ot.pairs_in_lane(ot.LANE_EC_CONDITIONED)} == {"zymctrl"}
        assert "zymctrl" not in ug.OFF_PANEL
        with pytest.raises(ValueError, match="EC class tag"):
            ug.generation_point("zymctrl")

    def test_the_free_length_points_are_the_renderings_the_arms_declare(self):
        for name in ("protgpt3-112m", "protgpt3-112m-dpo", "protgpt3-1.3b-dpo"):
            assert ug.prompt_for(name) == f"<|bos|>{A.BOS_DIRECTION_N_TO_C}"
            assert ug.close_token_for(name) == "<|eos|>"
            assert ug.generation_point(name).use_cache is True
        for name in ("progen2-small-hugohrban", "progen2-small-mix7"):
            assert ug.prompt_for(name) == A.N_TO_C_MARKER
            assert ug.close_token_for(name) == A.BOS_DIRECTION_C_TO_N
            # the released remote code carries the legacy cache signature
            assert ug.generation_point(name).use_cache is False
        assert ug.prompt_for("protgpt3-1.3b") == ug.prompt_for("protgpt3-1.3b-dpo")

    def test_an_arm_with_no_free_length_rendering_cannot_be_given_a_point(self, monkeypatch):
        spec = replace(ot.ARMS["protgpt3-112m"], input_format=ot.INPUT_FORMAT_EC_CONDITIONED)
        monkeypatch.setitem(ot.ARMS, "protgpt3-112m", spec)
        with pytest.raises(AssertionError, match="no free-length generation point"):
            ug._off_panel_points()


class TestStaging:
    """A sharded checkpoint mid-download is not a staged checkpoint."""

    def _point(self, monkeypatch, tmp_path, name="protgpt3-10b"):
        monkeypatch.setitem(ot.ARMS, name, replace(ot.ARMS[name], path=tmp_path))
        return tmp_path

    def test_a_missing_directory_is_not_staged(self, monkeypatch, tmp_path):
        self._point(monkeypatch, tmp_path / "absent")
        assert ot.weights_complete("protgpt3-10b") is False

    def test_an_empty_directory_is_not_staged(self, monkeypatch, tmp_path):
        self._point(monkeypatch, tmp_path)
        assert ot.weights_complete("protgpt3-10b") is False

    def test_a_single_weight_file_is_staged(self, monkeypatch, tmp_path):
        self._point(monkeypatch, tmp_path)
        (tmp_path / "pytorch_model.bin").write_bytes(b"")
        assert ot.weights_complete("protgpt3-10b") is True

    def test_a_shard_index_needs_every_shard_it_names(self, monkeypatch, tmp_path):
        self._point(monkeypatch, tmp_path)
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "s-1.safetensors", "b": "s-2.safetensors"}})
        )
        (tmp_path / "s-1.safetensors").write_bytes(b"")
        assert ot.weights_complete("protgpt3-10b") is False
        (tmp_path / "s-2.safetensors").write_bytes(b"")
        assert ot.weights_complete("protgpt3-10b") is True

    def test_an_unstaged_pair_drops_out_of_the_schedule_with_its_reason(self, monkeypatch, tmp_path):
        self._point(monkeypatch, tmp_path / "absent")
        assert "protgpt3-10b" not in ot.scheduled_free_length_arms()
        assert {pair.tuned for pair in ot.pairs_in_lane(ot.LANE_FREE_LENGTH)} >= {
            "protgpt3-10b-dpo"
        }
        skipped = builder().skipped_pairs()
        assert "weights incomplete" in skipped["protgpt3-10b-dpo"]
        assert ot.LANE_EC_CONDITIONED in skipped["zymctrl"]


class TestPromptContract:
    """The ProtGPT3 beginning-of-sequence token, and the two ways to get it wrong."""

    def test_the_lane_prompt_and_the_published_recipe_reach_the_same_two_ids(self):
        lane = ot.prompt_contract("protgpt3-112m", protgpt3_tokenizer(), "<|bos|>1")
        assert lane["prompt_ids"] == [1, 4] == lane["declared_marker_ids"]
        published = ot.prompt_contract("protgpt3-112m", protgpt3_tokenizer(add_bos=True), "1")
        assert published["prompt_ids"] == [1, 4]

    def test_a_doubled_beginning_of_sequence_token_is_refused(self):
        with pytest.raises(ValueError, match=r"encodes to \(1, 1, 4\)"):
            ot.prompt_contract("protgpt3-112m", protgpt3_tokenizer(add_bos=True), "<|bos|>1")

    def test_a_missing_beginning_of_sequence_token_is_refused(self):
        with pytest.raises(ValueError, match=r"encodes to \(4,\)"):
            ot.prompt_contract("protgpt3-112m", protgpt3_tokenizer(), "1")

    def test_a_tokenizer_without_the_direction_token_is_refused(self):
        vocab = {token: index for token, index in protgpt3_vocab().items() if token != "1"}
        tokenizer = CharTokenizer(
            vocab, bos="<|bos|>", eos="<|eos|>", unk="<unk>", specials=("<|bos|>",)
        )
        with pytest.raises(ValueError, match="direction token"):
            ot.prompt_contract("protgpt3-112m", tokenizer, "<|bos|>")

    def test_the_progen2_prompt_is_the_direction_marker_alone(self):
        vocab = {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2, "1": 3, "2": 4}
        vocab.update({residue: 5 + index for index, residue in enumerate(AA20)})
        vocab["<|endoftext|>"] = 30
        tokenizer = CharTokenizer(vocab, unk="<|endoftext|>", specials=("<|endoftext|>",))
        assert ot.prompt_contract("progen2-small-mix7", tokenizer, "1")["prompt_ids"] == [3]


class TestResidueContract:
    """A per-residue tokenizer and a residue string, or a refusal."""

    def test_every_residue_is_one_round_tripping_id(self):
        contract = ot.residue_alphabet_contract("protgpt3-112m", protgpt3_tokenizer())
        assert contract["alphabet"] == AA20
        assert contract["residue_ids"]["A"] == 6

    def test_a_multi_piece_residue_is_refused(self):
        class Split(CharTokenizer):
            def __call__(self, text, add_special_tokens=True, **kwargs):
                ids = super().__call__(text, add_special_tokens, **kwargs)["input_ids"]
                return {"input_ids": ids * 2 if text == "W" else ids}

        with pytest.raises(ValueError, match="not one id"):
            ot.residue_alphabet_contract("protgpt3-112m", Split(**_FAKE))

    def test_an_id_that_does_not_round_trip_is_refused(self):
        class Lower(CharTokenizer):
            def decode(self, ids, skip_special_tokens=False):
                return super().decode(ids, skip_special_tokens).lower()

        with pytest.raises(ValueError, match="round-trips to"):
            ot.residue_alphabet_contract("protgpt3-112m", Lower(**_FAKE))

    def test_empty_and_censored_continuations_stay_in_the_denominator(self):
        report = ot.require_residue_sequences("protgpt3-112m", ["MKT", "", "", "AC"])
        assert report["attempts"] == 4
        assert report["non_empty"] == 2
        assert (report["residues_min"], report["residues_max"]) == (2, 3)

    def test_a_sample_that_extracted_nothing_is_an_interface_failure(self):
        with pytest.raises(ValueError, match="extracted to an empty residue run"):
            ot.require_residue_sequences("protgpt3-112m", ["", "", ""])

    def test_a_non_residue_character_is_refused(self):
        with pytest.raises(ValueError, match="non-residue characters"):
            ot.require_residue_sequences("protgpt3-112m", ["MKT<|eos|>"])

    def test_no_continuation_at_all_is_refused(self):
        with pytest.raises(ValueError, match="at least one continuation"):
            ot.require_residue_sequences("protgpt3-112m", [])


class TestMatchedPair:
    """The pair is the only control, so both of its matching properties are asserted."""

    def _side(self, *, depth=8, width=512, model_type="mixtral", tokenizer=None, **boundary):
        declared = {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}
        declared.update(boundary)
        return SimpleNamespace(
            config=SimpleNamespace(
                model_type=model_type, num_hidden_layers=depth, hidden_size=width, **declared
            ),
            tokenizer=tokenizer if tokenizer is not None else protgpt3_tokenizer(),
        )

    def _pair(self):
        return ot.pair_for("protgpt3-112m-dpo")

    def test_a_matched_pair_reports_its_shared_properties(self):
        pair = self._pair()
        matched = ot.require_matched_pair(
            pair, {pair.base: self._side(), pair.tuned: self._side()}
        )
        assert matched["depth_width"] == [8, 512]
        assert matched["architecture"] == "mixtral"
        assert matched["tokens_added_by_tuning"] == []
        assert matched["special_token_id_differences"] == {}
        assert matched["declared_boundary_ids"]["eos_token_id"] == 2

    def test_a_list_valued_eos_declaration_still_matches_a_scalar_one(self):
        pair = self._pair()
        matched = ot.require_matched_pair(
            pair,
            {
                pair.base: self._side(eos_token_id=(2,)),
                pair.tuned: self._side(eos_token_id=[2]),
            },
        )
        assert matched["declared_boundary_ids"]["eos_token_id"] == (2,)

    def test_differing_stored_precision_is_reported_and_not_refused(self):
        """The 10B pair is serialised bfloat16 against float32 and is still matched.

        The lane gives both sides one compute dtype, so the asymmetry is
        neutralised at load; refusing here would reject two matched checkpoints
        over how their files were written, and dropping it from the record would
        hide a deliberate deviation from one arm's stored precision.
        """

        pair = self._pair()
        base = self._side(torch_dtype="bfloat16")
        base.stored_dtype = "bfloat16"
        tuned = self._side(torch_dtype="bfloat16")
        tuned.stored_dtype = "float32"
        matched = ot.require_matched_pair(pair, {pair.base: base, pair.tuned: tuned})
        assert matched["checkpoint_dtypes"] == {
            pair.base: {"config": "bfloat16", "stored": "bfloat16"},
            pair.tuned: {"config": "bfloat16", "stored": "float32"},
        }
        assert matched["config_dtype_agrees"] is True
        assert matched["stored_dtype_agrees"] is False

    def test_a_pair_stored_at_one_precision_agrees(self):
        pair = self._pair()
        sides = {}
        for name in pair.arms:
            side = self._side(torch_dtype="torch.bfloat16")
            side.stored_dtype = "bfloat16"
            sides[name] = side
        matched = ot.require_matched_pair(pair, sides)
        assert matched["stored_dtype_agrees"] is True
        assert matched["config_dtype_agrees"] is True
        assert {entry["config"] for entry in matched["checkpoint_dtypes"].values()} == {"bfloat16"}

    def test_the_stored_dtype_is_read_from_the_weight_headers(self):
        """The config can be wrong about its own tensors; the headers cannot.

        ``ProtGPT3-10B-dpo`` ships ``dtype: bfloat16`` beside a stale
        ``torch_dtype: float32`` and float32 tensors, which is exactly the case a
        config-only comparison would call matched.
        """

        for arm, expected in (
            ("protgpt3-112m", "bfloat16"),
            ("protgpt3-10b", "bfloat16"),
            ("protgpt3-10b-dpo", "float32"),
        ):
            path = ot.spec_for(arm).path
            if not ot.weights_complete(arm):
                pytest.skip(f"{arm} is not staged on this host")
            assert ot.stored_dtype(path) == expected, arm

    def test_an_unstaged_checkpoint_has_no_stored_dtype(self, tmp_path):
        assert ot.stored_dtype(tmp_path) is None

    def test_a_different_shape_is_refused(self):
        pair = self._pair()
        with pytest.raises(ValueError, match=r"live \(depth, width\)"):
            ot.require_matched_pair(
                pair, {pair.base: self._side(), pair.tuned: self._side(width=1024)}
            )

    def test_a_different_architecture_is_refused(self):
        pair = self._pair()
        with pytest.raises(ValueError, match="model_type"):
            ot.require_matched_pair(
                pair, {pair.base: self._side(), pair.tuned: self._side(model_type="llama")}
            )

    def test_a_different_boundary_id_is_refused(self):
        pair = self._pair()
        with pytest.raises(ValueError, match="different eos_token_id"):
            ot.require_matched_pair(
                pair, {pair.base: self._side(), pair.tuned: self._side(eos_token_id=7)}
            )

    def test_a_shifted_content_token_is_refused(self):
        pair = self._pair()
        shifted = protgpt3_vocab()
        shifted["M"] = 99
        tokenizer = CharTokenizer(
            shifted, bos="<|bos|>", eos="<|eos|>", unk="<unk>", specials=("<|bos|>", "<|eos|>")
        )
        with pytest.raises(ValueError, match="content tokens"):
            ot.require_matched_pair(
                pair, {pair.base: self._side(), pair.tuned: self._side(tokenizer=tokenizer)}
            )

    def test_an_added_family_token_shifts_only_a_loader_injected_special(self):
        """The real ProGen2 pair's shape: seven added tokens, one displaced special."""

        pair = ot.pair_for("progen2-small-mix7")
        base_vocab = {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2, "1": 3, "2": 4}
        base_vocab.update({residue: 5 + index for index, residue in enumerate(AA20)})
        tuned_vocab = dict(base_vocab)
        base_vocab["<|endoftext|>"] = 30
        for index, family in enumerate(("pf00042", "pf03668", "pf00257")):
            tuned_vocab[f"<|{family}|>"] = 30 + index
        tuned_vocab["<|endoftext|>"] = 33
        specials = ("<|endoftext|>",)
        matched = ot.require_matched_pair(
            pair,
            {
                pair.base: self._side(
                    depth=12,
                    width=1024,
                    model_type="progen",
                    pad_token_id=None,
                    tokenizer=CharTokenizer(base_vocab, unk="<|endoftext|>", specials=specials),
                ),
                pair.tuned: self._side(
                    depth=12,
                    width=1024,
                    model_type="progen",
                    pad_token_id=None,
                    tokenizer=CharTokenizer(tuned_vocab, unk="<|endoftext|>", specials=specials),
                ),
            },
        )
        assert matched["tokens_added_by_tuning"] == [
            "<|pf00042|>",
            "<|pf00257|>",
            "<|pf03668|>",
        ]
        assert matched["special_token_id_differences"] == {"<|endoftext|>": [30, 33]}
        assert matched["content_alphabet_tokens"] == 25

    def test_an_unloaded_side_is_refused(self):
        pair = self._pair()
        with pytest.raises(ValueError, match="nothing loaded"):
            ot.require_matched_pair(pair, {pair.base: self._side()})


class TestPublishedTokenizers:
    """The two checks that must see the real files, skipped when they are absent."""

    def _tokenizer(self, arm, **kwargs):
        if not ot.spec_for(arm).path.is_dir():
            pytest.skip(f"{arm} is not staged on this host")
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(ot.spec_for(arm).path), **kwargs)

    @pytest.mark.parametrize("arm", ["protgpt3-112m", "protgpt3-112m-dpo", "protgpt3-1.3b-dpo"])
    def test_the_published_default_plus_the_lane_prompt_is_the_contract(self, arm):
        tokenizer = self._tokenizer(arm)
        assert tokenizer.init_kwargs.get("add_bos_token") is False
        contract = ot.prompt_contract(arm, tokenizer, ug.prompt_for(arm))
        assert contract["prompt_ids"] == [tokenizer.bos_token_id, 4]
        assert ot.residue_alphabet_contract(arm, tokenizer)["residue_ids"]["A"] == 6

    def test_the_published_keyword_over_the_lane_prompt_doubles_the_bos(self):
        arm = "protgpt3-112m"
        tokenizer = self._tokenizer(arm, add_bos_token=True, padding_side="left")
        bos = tokenizer.bos_token_id
        assert [int(value) for value in tokenizer(ug.prompt_for(arm))["input_ids"]] == [bos, bos, 4]
        with pytest.raises(ValueError, match="out of distribution"):
            ot.prompt_contract(arm, tokenizer, ug.prompt_for(arm))
        assert ot.prompt_contract(arm, tokenizer, A.BOS_DIRECTION_N_TO_C)["prompt_ids"] == [bos, 4]


class TestCampaignShape:
    """Three streams per arm, four base cells per tuned cell, non-overlapping seeds."""

    def test_the_cell_plan_is_the_declared_budget(self):
        build = builder()
        pairs = ot.pairs_in_lane(ot.LANE_FREE_LENGTH)
        cells = build.cell_plan(pairs)
        assert len(cells) == 15 * len(pairs)
        for pair in pairs:
            mine = [cell for cell in cells if cell["pair"] == pair.tuned]
            assert len(mine) == 15
            assert sum(cell["role"] == "base" for cell in mine) == 12
            assert sum(cell["role"] == "tuned" for cell in mine) == 3
            assert {cell["stream"] for cell in mine} == {0, 1, 2}
            for stream in (0, 1, 2):
                roles = [cell["role"] for cell in mine if cell["stream"] == stream]
                assert roles == ["base"] * 4 + ["tuned"]
            base = sum(cell["attempts"] for cell in mine if cell["role"] == "base")
            tuned = sum(cell["attempts"] for cell in mine if cell["role"] == "tuned")
            assert base == 4 * tuned
        assert all(cell["attempts"] == ug.POLICY["attempts"] for cell in cells)

    def test_the_seeds_cannot_overlap_a_neighbouring_cell(self):
        build = builder()
        cells = build.cell_plan(ot.pairs_in_lane(ot.LANE_FREE_LENGTH))
        seeds = [cell["seed"] for cell in cells]
        assert seeds == sorted(seeds)
        assert len(set(seeds)) == len(seeds)
        assert all(
            later - earlier == build.SEED_STEP for earlier, later in zip(seeds, seeds[1:])
        )
        build.require_seed_separation(cells, ug.arm_batch_size)
        # the worst batch size any arm could declare still fits inside one step
        build.require_seed_separation(cells, lambda _arm: 1)
        wider = [dict(cell, attempts=build.SEED_STEP + 1) for cell in cells]
        with pytest.raises(SystemExit, match="would overlap"):
            build.require_seed_separation(wider, lambda _arm: 1)

    def test_the_seeds_do_not_depend_on_which_pairs_are_staged(self):
        """Finishing a download must not move the draw of a cell that already ran."""

        build = builder()
        declared = ot.pairs_in_lane(ot.LANE_FREE_LENGTH)
        assert len(declared) == 4
        stable = {build.label(cell): cell["seed"] for cell in build.cell_plan(declared)}
        for subset in (declared[:1], declared[:2], declared[2:], declared):
            cells = build.scheduled_cells(build.cell_plan(declared), subset)
            assert cells
            assert all(cell["seed"] == stable[build.label(cell)] for cell in cells)
        # seeding the staged subset instead is exactly what would have moved them
        shrunk = build.cell_plan(declared[1:])
        assert any(cell["seed"] != stable[build.label(cell)] for cell in shrunk)

    def test_the_manifest_rows_name_the_stage_the_arm_and_the_seed(self, tmp_path):
        build = builder()
        assert build.main(["--run", str(tmp_path), "--gpus", "0", "1", "2"]) == 0
        manifest = tmp_path / "queue" / f"campaign_{ot.CAMPAIGN}_generate.tsv"
        lines = manifest.read_text().splitlines()
        header = [line for line in lines if line.startswith("#")]
        rows = [line.split("\t") for line in lines if not line.startswith("#")]
        assert any("not scheduled: zymctrl" in line for line in header)
        assert len(rows) == 15 * len(ot.staged_pairs(ot.LANE_FREE_LENGTH))
        seeds = set()
        for fields in rows:
            assert len(fields) == 8
            slot, key, gpu, stage_name, label, env, expect, args = fields
            assert key == "ot"
            assert gpu in {"0", "1", "2"}
            assert stage_name == "unconditional_generation.py"
            assert expect == ug.EXPECT_GENERATE
            assert env.startswith("TRANSFER_PYTHON=") and not any(ch.isspace() for ch in env.split()[0])
            assert "--stage generate --arm " in args
            arm = args.split("--arm ")[1].split()[0]
            assert arm in ot.scheduled_free_length_arms()
            assert label.endswith(tuple(f"_r{index}" for index in range(4)))
            assert arm in label
            seed = int(args.split("--seed ")[1])
            assert seed not in seeds
            seeds.add(seed)
        stable = {
            build.label(cell): cell["seed"]
            for cell in build.cell_plan(ot.pairs_in_lane(ot.LANE_FREE_LENGTH))
        }
        for fields in rows:
            label = fields[4]
            assert int(fields[7].split("--seed ")[1]) == stable[label], label
        plan = json.loads((tmp_path / "queue" / "cell_plan.json").read_text())
        assert plan["campaign"] == ot.CAMPAIGN
        assert plan["policy"] == ug.POLICY
        assert plan["streams"] == 3
        assert "zymctrl" in plan["skipped_pairs"]
        assert len(plan["cells"]) == len(rows)

    def test_the_manifest_stage_name_resolves_to_exactly_one_entry_point(self):
        from scripts.capability.entrypoints import stage_path

        build = builder()
        assert stage_path(ROOT, build.STAGE).name == "unconditional_generation.py"


class TestStageEntryPoint:
    """The lane's entry point serves an off-panel arm and carries the stream seed."""

    def test_the_seed_is_parsed_and_defaults_to_the_policy_seed(self):
        module = stage()
        assert module.parse_args(
            ["--stage", "generate", "--arm", "protgpt2", "--out", "x", "--seed", "7"]
        ).seed == 7
        assert (
            module.parse_args(["--stage", "generate", "--arm", "protgpt2", "--out", "x"]).seed
            is None
        )

    def test_interface_only_serves_an_objective_tuned_arm_without_a_ledger(self, tmp_path):
        module = stage()
        out = tmp_path / "protgpt3-112m-dpo"
        module.main(
            [
                "--stage",
                "interface-only",
                "--arm",
                "protgpt3-112m-dpo",
                "--out",
                str(out),
                "--device",
                "cpu",
            ]
        )
        record = json.loads((out / ug.EXPECT_INTERFACE).read_text(encoding="utf-8"))
        assert record["prompt"] == f"<|bos|>{A.BOS_DIRECTION_N_TO_C}"
        assert record["checkpoint"] == str(ug.checkpoint_for_arm("protgpt3-112m-dpo"))
        assert record["is_scientific_measurement"] is False
        assert not (out / ug.LEDGER_NAME).exists()

    def test_a_refused_arm_still_refuses_through_the_entry_point(self, tmp_path):
        module = stage()
        with pytest.raises(ValueError, match="EC class tag"):
            module.main(
                ["--stage", "interface-only", "--arm", "zymctrl", "--out", str(tmp_path / "z")]
            )

    def test_the_stream_seed_reaches_the_sampler_and_the_ledger(self, tmp_path):
        seen: dict[str, object] = {}

        def fake_generate(**kwargs):
            seen.update(kwargs)
            return ["MKTA", ""]

        rows = ug.generate(
            "protgpt3-112m-dpo",
            n=2,
            seed=20264011000,
            generate_fn=fake_generate,
            output_dir=tmp_path,
        )
        assert seen["seed"] == 20264011000
        assert seen["prompt"] == ug.prompt_for("protgpt3-112m-dpo")
        assert [row["sequence"] for row in rows] == ["MKTA", ""]
        summary = json.loads((tmp_path / ug.EXPECT_GENERATE).read_text(encoding="utf-8"))
        assert summary["seed"] == 20264011000
        assert summary["policy"]["seed"] == ug.POLICY["seed"]
        assert summary["is_scientific_measurement"] is False

    def test_the_ledger_and_the_structure_subset_are_written_for_an_off_panel_arm(self, tmp_path):
        """The whole per-cell output the queue expects, on a pair arm, without a GPU."""

        raws = ["A" * 40] * 40 + ["C" * 180] * 40 + ["D" * 300] * 40 + ["E" * 600] * 40 + ["", "!"]
        rows = ug.generate(
            "progen2-small-mix7",
            n=len(raws),
            seed=20264070000,
            generate_fn=lambda **_kwargs: list(raws),
            output_dir=tmp_path,
        )
        assert len(rows) == len(raws)
        # the two unusable attempts stay in the ledger's denominator
        assert sum(1 for row in rows if row["sequence"] == "") == 2
        assert sum(1 for row in rows if row["stop_status"] == "empty_or_noncanonical") == 2
        ledger = [json.loads(line) for line in (tmp_path / ug.LEDGER_NAME).read_text().splitlines()]
        assert len(ledger) == len(raws)

        structure = ug.write_structure_subset(tmp_path, rows, arm="progen2-small-mix7")
        subset = [
            json.loads(line)
            for line in (tmp_path / ug.STRUCTURE_SUBSET).read_text().splitlines()
        ]
        assert structure["structure_records_including_shuffles"] == len(subset)
        parents = [row for row in subset if row["role"] == "generation"]
        assert len(parents) == ug.POLICY["structure_samples"]
        assert len(subset) == 2 * len(parents)
        assert all(row["arm"] == "progen2-small-mix7" for row in subset)
        assert all(row["sequence"] for row in parents)

    def test_the_default_seed_is_the_policy_seed(self, tmp_path):
        seen: dict[str, object] = {}
        ug.generate(
            "progen2-small-mix7",
            n=1,
            generate_fn=lambda **kwargs: (seen.update(kwargs), ["MK"])[1],
            output_dir=tmp_path,
        )
        assert seen["seed"] == ug.POLICY["seed"]
        summary = json.loads((tmp_path / ug.EXPECT_GENERATE).read_text(encoding="utf-8"))
        assert summary["seed"] == ug.POLICY["seed"]
