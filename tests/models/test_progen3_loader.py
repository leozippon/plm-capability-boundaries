"""The ProGen3 loader, tested where it can actually go wrong.

``ProGen3ForCausalLM.from_pretrained(path, moe_implementation="eager")`` returns
a model whose every expert and router is random, without raising: the released
weights are in megablocks packing and carry no key the eager block recognises.
``src.capability.models.progen3`` exists to make that impossible, and the four things
that could let it back in are tested here.

**The mapping.** A wrong expert mapping still produces a state dict that loads
with ``strict=True``, so nothing structural catches it. Two tests cover it: the
conversion is exactly invertible, and the converted per-expert weights compute
the same function as the packed ones under the megablocks reference math -- and
stop doing so the moment the gate and up projections are exchanged.

**The load.** A missing or unexpected key must abort, because a partial load of
either checkpoint is silent and plausible. The 3B ships in two safetensors
shards named by an index, so the same refusal is tested on a release read from
several files as on one read from a single file, along with the refusal of a
release the index says is incomplete.

**The band.** The numerical tripwire is only worth its cost if it separates a
correct mapping from every corruption that survives ``strict=True`` *on the
checkpoint being scored*. The 112M and the 3B do not score anywhere near each
other, so this is a property of each declared reference rather than of one
module constant, and every declared checkpoint is held to it here. Checkable
without a GPU or the weights: the measurements themselves come from
``external/code/progen3_eager_probe`` and from the band
measurements recorded in ``src.capability.models.progen3``.

**The addressing.** ``fused_attention_norm`` moves the attention between two
places in the module tree, and the two released checkpoints disagree about it,
so which one an ablation reaches is tested on both layouts.

Nothing here loads a real checkpoint, imports the third-party ``progen3``
package, or needs a GPU; the releases it reads are ten-tensor ones it writes
itself.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.models.progen3 import (  # noqa: E402
    NON_RESIDUE_TOKENS,
    SELF_CHECK_HALF_WIDTH,
    SELF_CHECK_REFERENCES,
    Component,
    ProGen3,
    ablated,
    check_nll,
    components,
    convert_megablocks_state_dict,
    moe_intercept,
    n_to_c_target_rows,
    non_residue_token_ids,
    release_shards,
    released_state_dict,
    residue_target_mask,
    scored_logits,
    self_check_reference,
    strict_load,
)

EXPERTS = 3
FFN = 4
HIDDEN = 5
TOP_K = 2

#: The two checkpoints this module is declared for, by architecture fingerprint.
PROGEN3_112M = (10, 384, 1152)
PROGEN3_3B = (24, 1280, 3840)


def tiny_checkpoint(seed: int = 0, layers: int = 1) -> dict[str, torch.Tensor]:
    """A megablocks-packed checkpoint the size of a unit test."""

    generator = torch.Generator().manual_seed(seed)

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    state = {"model.embed_tokens.weight": normal(7, HIDDEN)}
    for layer in range(layers):
        prefix = f"model.layers.{layer}.block_sparse_moe"
        state[f"{prefix}.experts.mlp.w1"] = normal(EXPERTS * FFN, HIDDEN)
        state[f"{prefix}.experts.mlp.v1"] = normal(EXPERTS * FFN, HIDDEN)
        state[f"{prefix}.experts.mlp.w2"] = normal(EXPERTS * FFN, HIDDEN)
        state[f"{prefix}.router.layer.weight"] = normal(EXPERTS, HIDDEN)
    state["mlm_head.weight"] = normal(7, HIDDEN)
    return state


def convert(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return convert_megablocks_state_dict(
        raw, num_experts=EXPERTS, intermediate_size=FFN
    )


class TheMegablocksConversionIsExactlyInvertible(unittest.TestCase):
    """Repacking the converted tensors must give the released ones back."""

    def setUp(self) -> None:
        self.raw = tiny_checkpoint()
        self.converted = convert(self.raw)

    def test_every_packed_expert_tensor_is_recoverable(self):
        prefix = "model.layers.0.block_sparse_moe"
        for packed, target in (("w1", "w1"), ("v1", "w3"), ("w2", "w2")):
            slices = []
            for expert in range(EXPERTS):
                stored = self.converted[f"{prefix}.experts.{expert}.{target}.weight"]
                # w2 is the only transposed one, and it is the only one whose
                # stored shape is (hidden, ffn) rather than (ffn, hidden).
                expected_shape = (HIDDEN, FFN) if packed == "w2" else (FFN, HIDDEN)
                self.assertEqual(tuple(stored.shape), expected_shape, target)
                slices.append(stored.T if packed == "w2" else stored)
            repacked = torch.cat(slices, dim=0)
            self.assertTrue(
                torch.equal(repacked, self.raw[f"{prefix}.experts.mlp.{packed}"]),
                f"{packed} does not survive the round trip",
            )

    def test_the_router_is_renamed_and_the_mlm_head_is_dropped(self):
        prefix = "model.layers.0.block_sparse_moe"
        self.assertTrue(
            torch.equal(
                self.converted[f"{prefix}.gate.weight"],
                self.raw[f"{prefix}.router.layer.weight"],
            )
        )
        self.assertNotIn(f"{prefix}.router.layer.weight", self.converted)
        self.assertNotIn("mlm_head.weight", self.converted)

    def test_everything_outside_the_moe_is_passed_through_untouched(self):
        self.assertTrue(
            torch.equal(
                self.converted["model.embed_tokens.weight"],
                self.raw["model.embed_tokens.weight"],
            )
        )

    def test_a_packed_row_count_that_contradicts_the_config_is_refused(self):
        # The split is the one place a config/checkpoint disagreement turns into
        # silently wrong slices rather than an error.
        with self.assertRaises(ValueError) as caught:
            convert_megablocks_state_dict(
                self.raw, num_experts=EXPERTS + 1, intermediate_size=FFN
            )
        self.assertIn("packed row count", str(caught.exception))

    def test_an_undeclared_packed_expert_tensor_is_refused(self):
        raw = dict(self.raw)
        raw["model.layers.0.block_sparse_moe.experts.mlp.w9"] = torch.zeros(
            EXPERTS * FFN, HIDDEN
        )
        with self.assertRaises(ValueError) as caught:
            convert(raw)
        self.assertIn("no declared target", str(caught.exception))


class TheConvertedWeightsComputeTheMegablocksFunction(unittest.TestCase):
    """Invertibility is not enough: the split must land on the right projections.

    ``w1`` is the gate and ``v1`` is the up projection. Exchanging them
    round-trips perfectly, loads with ``strict=True``, and scores 3.18
    nats/token against 2.29 -- so only the arithmetic separates them.
    """

    def setUp(self) -> None:
        self.raw = tiny_checkpoint(seed=3)
        self.x = torch.randn(6, HIDDEN, generator=torch.Generator().manual_seed(9)).double()

    def packed_reference(self, raw: dict[str, torch.Tensor]) -> torch.Tensor:
        """megablocks 0.7.0: LearnedRouter + MemoryOptimizedGroupedGLU, run densely."""

        prefix = "model.layers.0.block_sparse_moe"
        w1 = raw[f"{prefix}.experts.mlp.w1"].view(EXPERTS, FFN, HIDDEN)
        v1 = raw[f"{prefix}.experts.mlp.v1"].view(EXPERTS, FFN, HIDDEN)
        w2 = raw[f"{prefix}.experts.mlp.w2"].view(EXPERTS, FFN, HIDDEN)
        scores = F.linear(self.x, raw[f"{prefix}.router.layer.weight"]).softmax(-1)
        weights, chosen = torch.topk(scores, TOP_K, dim=-1)
        weights = weights / weights.norm(p=1, dim=-1, keepdim=True)
        out = torch.zeros_like(self.x)
        for expert in range(EXPERTS):
            for slot in range(TOP_K):
                selected = chosen[:, slot] == expert
                if not selected.any():
                    continue
                rows = self.x[selected]
                activation = F.silu(rows @ w1[expert].T) * (rows @ v1[expert].T)
                out[selected] += (activation @ w2[expert]) * weights[selected, slot, None]
        return out

    def eager_from_converted(self, converted: dict[str, torch.Tensor]) -> torch.Tensor:
        """The same math through the eager ``GLUMLP``: w2(silu(w1(x)) * w3(x))."""

        prefix = "model.layers.0.block_sparse_moe"
        scores = F.linear(self.x, converted[f"{prefix}.gate.weight"]).softmax(-1)
        weights, chosen = torch.topk(scores, TOP_K, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        out = torch.zeros_like(self.x)
        for expert in range(EXPERTS):
            gate = converted[f"{prefix}.experts.{expert}.w1.weight"]
            up = converted[f"{prefix}.experts.{expert}.w3.weight"]
            down = converted[f"{prefix}.experts.{expert}.w2.weight"]
            for slot in range(TOP_K):
                selected = chosen[:, slot] == expert
                if not selected.any():
                    continue
                rows = self.x[selected]
                activation = F.silu(F.linear(rows, gate)) * F.linear(rows, up)
                out[selected] += F.linear(activation, down) * weights[selected, slot, None]
        return out

    def test_the_split_experts_reproduce_the_packed_grouped_glu(self):
        self.assertTrue(
            torch.allclose(
                self.eager_from_converted(convert(self.raw)),
                self.packed_reference(self.raw),
                atol=1e-12,
            )
        )

    def test_exchanging_the_gate_and_up_projections_changes_the_function(self):
        swapped = {}
        for key, value in self.raw.items():
            if key.endswith(".experts.mlp.w1"):
                swapped[key.replace(".mlp.w1", ".mlp.v1")] = value
            elif key.endswith(".experts.mlp.v1"):
                swapped[key.replace(".mlp.v1", ".mlp.w1")] = value
            else:
                swapped[key] = value
        corrupted = convert(swapped)
        # It round-trips and keeps every key, which is why strict loading cannot
        # see it...
        self.assertEqual(set(corrupted), set(convert(self.raw)))
        # ...and computes a different function, which is why the NLL can.
        self.assertFalse(
            torch.allclose(
                self.eager_from_converted(corrupted),
                self.packed_reference(self.raw),
                atol=1e-6,
            )
        )


def write_single(root: Path, state: dict[str, torch.Tensor]) -> Path:
    """A release the shape ProGen3-112M ships in: one ``model.safetensors``."""

    root.mkdir(parents=True, exist_ok=True)
    save_file(state, str(root / "model.safetensors"))
    return root


def write_sharded(root: Path, shards: list[dict[str, torch.Tensor]]) -> Path:
    """A release the shape ProGen3-3B ships in: several files and an index."""

    root.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for position, shard in enumerate(shards, start=1):
        name = f"model-{position:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(root / name))
        weight_map.update(dict.fromkeys(shard, name))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return root


class _ReleaseOnDisk(unittest.TestCase):
    """A scratch directory to write toy releases into."""

    def setUp(self) -> None:
        self.raw = tiny_checkpoint(seed=11, layers=2)
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)

    def halves(self) -> list[dict[str, torch.Tensor]]:
        """The tensors split by layer, the way the 3B's own index splits them."""

        first = {
            key: value
            for key, value in self.raw.items()
            if not key.startswith("model.layers.1.")
        }
        second = {key: value for key, value in self.raw.items() if key not in first}
        return [first, second]


class TheReleaseIsReadWhicheverWayItIsShipped(_ReleaseOnDisk):
    """One file for the 112M, two and an index for the 3B, one state dict either way.

    Where the index exists it is the authority on what the release consists of,
    which is what makes a directory listing the wrong thing to read and half a
    download something to name rather than to open.
    """

    def test_a_single_file_release_is_read(self):
        state = released_state_dict(write_single(self.root / "one", self.raw))
        self.assertEqual(set(state), set(self.raw))

    def test_a_sharded_release_reads_to_the_same_tensors_as_a_single_file_one(self):
        single = released_state_dict(write_single(self.root / "one", self.raw))
        sharded = released_state_dict(write_sharded(self.root / "many", self.halves()))
        self.assertEqual(set(sharded), set(single))
        for key, value in single.items():
            self.assertTrue(torch.equal(sharded[key], value), key)

    def test_the_index_rather_than_the_directory_listing_names_the_shards(self):
        checkpoint = write_sharded(self.root / "many", self.halves())
        (checkpoint / "extra.safetensors").write_bytes(b"not a shard")
        self.assertEqual(
            [path.name for path in release_shards(checkpoint)],
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        )

    def test_a_shard_the_index_names_and_the_directory_lacks_is_refused(self):
        # The state an interrupted download leaves behind: the index is small and
        # arrives first, so it describes more than the directory holds.
        checkpoint = write_sharded(self.root / "many", self.halves())
        (checkpoint / "model-00002-of-00002.safetensors").unlink()
        with self.assertRaises(FileNotFoundError) as caught:
            released_state_dict(checkpoint)
        message = str(caught.exception)
        self.assertIn("model-00002-of-00002.safetensors", message)
        self.assertIn("incomplete", message)

    def test_a_directory_holding_neither_layout_is_refused(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError) as caught:
            released_state_dict(empty)
        self.assertIn("PROGEN3_DIR", str(caught.exception))


def eager_target(layers: int = 1) -> nn.Module:
    """A module whose state-dict keys are exactly the ones the conversion produces."""

    backbone = nn.Module()
    backbone.embed_tokens = nn.Embedding(7, HIDDEN)
    decoder = []
    for _ in range(layers):
        block = nn.Module()
        block.experts = nn.ModuleList(
            nn.ModuleDict(
                {
                    "w1": nn.Linear(HIDDEN, FFN, bias=False),
                    "w2": nn.Linear(FFN, HIDDEN, bias=False),
                    "w3": nn.Linear(HIDDEN, FFN, bias=False),
                }
            )
            for _ in range(EXPERTS)
        )
        block.gate = nn.Linear(HIDDEN, EXPERTS, bias=False)
        layer = nn.Module()
        layer.block_sparse_moe = block
        decoder.append(layer)
    backbone.layers = nn.ModuleList(decoder)
    model = nn.Module()
    model.model = backbone
    return model.double()


class AShardedReleaseIsRefusedUnlessItIsConverted(_ReleaseOnDisk):
    """The refusal this module exists for, on the path the 3B takes to disk.

    Reading a release from two files instead of one changes nothing about what
    the eager model expects, and that is the claim: the released expert and
    router names are still names the model does not have, so a release loaded
    straight off disk must still be refused however many files it arrived in.
    """

    def state(self) -> dict[str, torch.Tensor]:
        return released_state_dict(write_sharded(self.root / "many", self.halves()))

    def test_the_converted_shards_load_into_the_eager_key_set(self):
        model = eager_target(layers=2)
        strict_load(model, convert(self.state()))
        packed = self.raw["model.layers.1.block_sparse_moe.experts.mlp.w1"]
        self.assertTrue(
            torch.equal(
                model.model.layers[1].block_sparse_moe.experts[2]["w1"].weight,
                packed[2 * FFN : 3 * FFN, :],
            )
        )

    def test_the_unconverted_shards_are_refused(self):
        # Exactly what from_pretrained does not do: it keeps the model it built
        # and reports the leftovers in a warning.
        model = eager_target(layers=2)
        before = model.model.layers[0].block_sparse_moe.gate.weight.clone()
        with self.assertRaises(RuntimeError) as caught:
            strict_load(model, self.state())
        message = str(caught.exception)
        self.assertIn("random initialisation", message)
        self.assertIn("block_sparse_moe.experts.mlp.w1", message)
        self.assertTrue(
            torch.equal(model.model.layers[0].block_sparse_moe.gate.weight, before)
        )

    def test_a_shard_that_is_present_but_short_is_refused_by_name(self):
        # The file is there and loads; it just does not carry what the index said.
        # strict_load is the backstop, and it has to name the layer that is absent
        # rather than report a model that quietly kept its initialisation.
        first, second = self.halves()
        short = {
            key: value
            for key, value in second.items()
            if "block_sparse_moe.experts" not in key
        }
        state = released_state_dict(write_sharded(self.root / "short", [first, short]))
        with self.assertRaises(RuntimeError) as caught:
            strict_load(eager_target(layers=2), convert(state))
        self.assertIn("model.layers.1.block_sparse_moe.experts.0.w1.weight", str(caught.exception))


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 2, bias=False)
        self.second = nn.Linear(2, 3, bias=False)


class APartialLoadIsRefusedRatherThanRun(unittest.TestCase):
    """The failure this module exists to prevent is a load that half-works."""

    def test_a_complete_state_dict_loads(self):
        model = _TinyModel()
        state = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
        strict_load(model, state)
        self.assertTrue(torch.equal(model.first.weight, torch.ones(2, 3)))

    def test_a_missing_key_raises_and_names_it(self):
        model = _TinyModel()
        state = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
        del state["second.weight"]
        with self.assertRaises(RuntimeError) as caught:
            strict_load(model, state)
        message = str(caught.exception)
        self.assertIn("second.weight", message)
        self.assertIn("random initialisation", message)

    def test_an_unexpected_key_raises_and_names_it(self):
        model = _TinyModel()
        state = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
        state["third.weight"] = torch.zeros(2, 2)
        with self.assertRaises(RuntimeError) as caught:
            strict_load(model, state)
        self.assertIn("third.weight", str(caught.exception))

    def test_a_refused_load_leaves_the_model_untouched(self):
        # A refusal that had already written half the tensors would leave the
        # caller holding exactly the object this module refuses to produce --
        # which is what happens if the key check runs after the assignment.
        model = _TinyModel()
        before = model.first.weight.clone()
        with self.assertRaises(RuntimeError):
            strict_load(model, {"first.weight": torch.ones(2, 3)})
        self.assertTrue(torch.equal(model.first.weight, before))


class EveryDeclaredBandSeparatesTheMappingsThatStrictLoadingCannot(unittest.TestCase):
    """The band is the only thing standing between a wrong mapping and a result.

    Checked against each checkpoint's recorded measurements rather than by
    running the model: the corrupted variants need the real weights, and the
    numbers they produced are the evidence each band was chosen from. Every
    declared checkpoint is held to the same requirements, so a third one cannot
    be added on a band nobody showed to discriminate.
    """

    def test_every_measured_correct_mapping_is_accepted(self):
        for reference in SELF_CHECK_REFERENCES.values():
            with self.subTest(checkpoint=reference.name):
                record = check_nll(reference, reference.correct_mapping)
                self.assertEqual(record["verdict"], "PASS")
                self.assertEqual(record["checkpoint"], reference.name)

    def test_every_measured_corruption_is_rejected(self):
        for reference in SELF_CHECK_REFERENCES.values():
            for name, value in reference.corruptions.items():
                with self.subTest(checkpoint=reference.name, corruption=name):
                    with self.assertRaises(RuntimeError) as caught:
                        check_nll(reference, value)
                    self.assertIn("outside the band", str(caught.exception))

    def test_every_checkpoint_declares_a_corruption_its_band_must_clear(self):
        # A reference with no corruption beside it is a band nobody has shown to
        # discriminate: it would return PASS on the checkpoint's own number and
        # say nothing about what else it accepts.
        for reference in SELF_CHECK_REFERENCES.values():
            with self.subTest(checkpoint=reference.name):
                self.assertTrue(reference.corruptions, reference.name)

    def test_every_band_clears_its_own_nearest_corruption(self):
        # A band that touched the nearest corruption would pass a broken mapping
        # on any environment that shifted the number slightly.
        for reference in SELF_CHECK_REFERENCES.values():
            with self.subTest(checkpoint=reference.name):
                nearest = min(reference.corruptions.values())
                self.assertLess(reference.band[1], nearest)
                self.assertGreater(nearest - reference.band[1], 0.25)

    def test_every_band_brackets_its_own_measurement_widely_enough_to_survive(self):
        # The observed spread of a correct load across dtype, batch size, scoring
        # direction and CPU-versus-GPU is 0.005 nats; anything under 0.05 would be
        # a tripwire for the hardware rather than for the mapping.
        self.assertGreater(SELF_CHECK_HALF_WIDTH, 0.05)
        for reference in SELF_CHECK_REFERENCES.values():
            with self.subTest(checkpoint=reference.name):
                low, high = reference.band
                self.assertLess(low, reference.correct_mapping)
                self.assertLess(reference.correct_mapping, high)

    def test_a_value_inside_the_band_is_reported_with_that_checkpoints_evidence(self):
        # And with no other checkpoint's: a reader handed both sets can draw a
        # comparison between two models' numbers that means nothing.
        for reference in SELF_CHECK_REFERENCES.values():
            with self.subTest(checkpoint=reference.name):
                record = check_nll(reference, reference.correct_mapping)
                self.assertEqual(record["reference"], reference.measured)
                self.assertEqual(record["band"], list(reference.band))
                others = {
                    value
                    for other in SELF_CHECK_REFERENCES.values()
                    if other is not reference
                    for value in other.measured.values()
                }
                self.assertFalse(others & set(record["reference"].values()))

    def test_the_declared_bands_reject_each_others_measurements(self):
        """Why the band is per-checkpoint rather than one module constant.

        A correct 3B scores 0.78 nats below a correct 112M -- more than two
        half-widths -- so the band this module used to declare globally would have
        refused the load it exists to certify, reporting it as a change to the
        scored-target convention. Widening one band to hold both is the other way
        to make that go away, and it is worse: its floor would sit 0.78 nats under
        the 112M's own value, which is 150 times the spread a correct load shows
        across every environment, so on the smaller checkpoint the lower edge
        would stop being a tripwire at all.
        """

        small = SELF_CHECK_REFERENCES[PROGEN3_112M]
        large = SELF_CHECK_REFERENCES[PROGEN3_3B]
        with self.assertRaises(RuntimeError):
            check_nll(small, large.correct_mapping)
        with self.assertRaises(RuntimeError):
            check_nll(large, small.correct_mapping)
        gap = small.correct_mapping - large.correct_mapping
        self.assertGreater(gap, 2 * SELF_CHECK_HALF_WIDTH)


class TheBandIsResolvedFromTheCheckpointAboutToBeScored(unittest.TestCase):
    """Which band applies is read off the loaded config, not off a path or a flag.

    ``PROGEN3_DIR`` relocates the weights and a mirror is free to rename
    the directory, so a name-keyed lookup could hand a 3B the 112M's band. The
    shape the state dict has to match cannot move the same way.
    """

    @staticmethod
    def config(fingerprint: tuple[int, int, int]) -> SimpleNamespace:
        layers, hidden, intermediate = fingerprint
        return SimpleNamespace(
            num_hidden_layers=layers,
            hidden_size=hidden,
            intermediate_size=intermediate,
        )

    def test_each_declared_fingerprint_resolves_to_its_own_reference(self):
        for fingerprint, reference in SELF_CHECK_REFERENCES.items():
            with self.subTest(checkpoint=reference.name):
                self.assertIs(
                    self_check_reference(self.config(fingerprint)), reference
                )

    def test_an_undeclared_architecture_is_refused_rather_than_gated_on_another(self):
        # ProGen3-1B, which nobody has measured. Scoring it against the 112M's
        # band would report a verdict, and a verdict is worse than no gate.
        with self.assertRaises(KeyError) as caught:
            self_check_reference(self.config((16, 1536, 4608)))
        message = str(caught.exception)
        self.assertIn("no self-check reference has been measured", message)
        self.assertIn("progen3-112m", message)


class _Block(nn.Module):
    """A stand-in for ``SparseMoeBlock``: returns (hidden, router probabilities)."""

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden * 2.0, torch.zeros(hidden.shape[0], 3)


class _Layer(nn.Module):
    """A miniature of ProGen3's serial decoder layer.

    ``forward`` is the identity the estimand check exists to verify: the residual
    the normalisation read, plus the MoE block's output, IS the layer's output.
    ``leak`` breaks exactly that and nothing else, which is what lets the check be
    tested on its failing path rather than only on its passing one.
    """

    def __init__(
        self, heads: int, head_dim: int, leak: float = 0.0, fused: bool = False
    ) -> None:
        super().__init__()
        self.block_sparse_moe = _Block()
        self.post_attention_layernorm = nn.Identity()
        attention = nn.Module()
        attention.o_proj = nn.Linear(heads * head_dim, heads * head_dim, bias=False)
        # The 3B's config fuses the attention with the norms either side of it,
        # which moves the module the ablation has to reach.
        if fused:
            self.norm_attn_norm = nn.Module()
            self.norm_attn_norm.self_attn = attention
        else:
            self.self_attn = attention
        self.leak = leak

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        contribution, _router = self.block_sparse_moe(self.post_attention_layernorm(hidden))
        return hidden + contribution + self.leak


class _Backbone(nn.Module):
    def __init__(
        self,
        layers: int,
        heads: int,
        head_dim: int,
        leak: float = 0.0,
        fused: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_Layer(heads, head_dim, leak, fused) for _ in range(layers)]
        )


class _Model(nn.Module):
    def __init__(
        self,
        layers: int,
        heads: int,
        head_dim: int,
        leak: float = 0.0,
        fused: bool = False,
    ) -> None:
        super().__init__()
        self.model = _Backbone(layers, heads, head_dim, leak, fused)
        self.width = heads * head_dim

    def forward(self, input_ids=None, **kwargs):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, self.width)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=hidden)


class _Preparer:
    """Just enough of the released batch preparer to make a forward pass happen."""

    def get_batch_kwargs(self, sequences, *, device, reverse=False):
        width = max(len(s) for s in sequences)
        ids = torch.zeros(len(sequences), width, dtype=torch.long)
        for row, sequence in enumerate(sequences):
            ids[row, : len(sequence)] = torch.tensor(
                [1 + (ord(c) % 20) for c in sequence], dtype=torch.long
            )
        return {
            "input_ids": ids.to(device),
            "position_ids": torch.arange(width, device=device).expand(len(sequences), width),
            "sequence_ids": torch.zeros(len(sequences), width, dtype=torch.long, device=device),
        }


def stub(
    layers: int = 2,
    heads: int = 3,
    head_dim: int = 4,
    leak: float = 0.0,
    fused: bool = False,
) -> ProGen3:
    return ProGen3(
        model=_Model(layers, heads, head_dim, leak, fused),
        config=SimpleNamespace(
            num_hidden_layers=layers,
            num_attention_heads=heads,
            hidden_size=heads * head_dim,
            fused_attention_norm=fused,
            pad_token_id=0,
        ),
        preparer=_Preparer(),
        device=torch.device("cpu"),
        checkpoint=Path("/nonexistent"),
    )


class TheInterventionSurfaceHonoursTheBlockContract(unittest.TestCase):
    """The MoE block returns a pair, and a hook that forgets that is silent."""

    def test_returning_none_leaves_the_output_untouched(self):
        pg = stub()
        seen: list[int] = []
        x = torch.ones(2, 12)
        with moe_intercept(pg, lambda layer, i, o: seen.append(layer) or None):
            hidden, router = pg.moe_blocks[0](x)
        self.assertEqual(seen, [0])
        self.assertTrue(torch.equal(hidden, x * 2.0))
        self.assertEqual(tuple(router.shape), (2, 3))

    def test_a_replacement_keeps_the_router_term(self):
        pg = stub()
        x = torch.ones(2, 12)
        with moe_intercept(pg, lambda layer, i, o: torch.zeros_like(o)):
            hidden, router = pg.moe_blocks[1](x)
        self.assertTrue(torch.equal(hidden, torch.zeros(2, 12)))
        self.assertEqual(tuple(router.shape), (2, 3))

    def test_the_interceptor_sees_the_blocks_input_and_output(self):
        pg = stub()
        recorded: dict[str, torch.Tensor] = {}

        def tap(layer, block_input, block_output):
            recorded["in"] = block_input
            recorded["out"] = block_output
            return None

        x = torch.arange(12, dtype=torch.float32).reshape(1, 12)
        with moe_intercept(pg, tap):
            pg.moe_blocks[0](x)
        self.assertTrue(torch.equal(recorded["in"], x))
        self.assertTrue(torch.equal(recorded["out"], x * 2.0))

    def test_an_ablation_applies_on_top_of_a_replacement(self):
        # The composition the causal stage depends on: the replacement runs, and
        # the ablation removes the replacement's own output rather than the
        # original block's.
        pg = stub()
        x = torch.ones(1, 12)
        with moe_intercept(pg, lambda layer, i, o: o + 100.0):
            with ablated(pg, Component("moe_block", 0)):
                inner, _ = pg.moe_blocks[0](x)
            outer, _ = pg.moe_blocks[0](x)
        self.assertTrue(torch.equal(inner, torch.zeros(1, 12)))
        self.assertTrue(torch.equal(outer, x * 2.0 + 100.0))

    def test_every_hook_is_removed_when_the_context_exits(self):
        pg = stub()
        with moe_intercept(pg, lambda layer, i, o: torch.zeros_like(o)):
            pass
        hidden, _ = pg.moe_blocks[0](torch.ones(1, 12))
        self.assertTrue(torch.equal(hidden, torch.full((1, 12), 2.0)))


class HeadAblationRemovesExactlyOneHeadsColumns(unittest.TestCase):
    """Head ``h`` owns ``o_proj`` input columns ``h*head_dim .. (h+1)*head_dim``."""

    def test_only_the_named_heads_slice_is_zeroed(self):
        pg = stub(layers=1, heads=3, head_dim=4)
        projection = pg.model.model.layers[0].self_attn.o_proj
        x = torch.arange(12, dtype=torch.float32).reshape(1, 12)
        with ablated(pg, Component("attention_head", 0, 1)):
            observed = projection(x)
        expected_input = x.clone()
        expected_input[:, 4:8] = 0.0
        self.assertTrue(torch.allclose(observed, projection(expected_input)))
        # The caller's tensor must not have been modified in place.
        self.assertTrue(torch.equal(x, torch.arange(12, dtype=torch.float32).reshape(1, 12)))

    def test_the_hook_is_removed_afterwards(self):
        pg = stub(layers=1, heads=3, head_dim=4)
        projection = pg.model.model.layers[0].self_attn.o_proj
        x = torch.ones(1, 12)
        with ablated(pg, Component("attention_head", 0, 0)):
            pass
        self.assertTrue(torch.allclose(projection(x), projection(x.clone())))
        with ablated(pg, Component("attention_head", 0, 0)):
            ablated_output = projection(x)
        self.assertFalse(torch.allclose(ablated_output, projection(x)))

    def test_a_component_kind_with_no_ablation_path_is_refused(self):
        pg = stub()
        with self.assertRaises(ValueError) as caught:
            with ablated(pg, Component("expert", 0, 1)):
                pass
        self.assertIn("no ablation is implemented", str(caught.exception))

    def test_the_ablation_reaches_the_attention_the_fused_layout_hides(self):
        # The 3B sets fused_attention_norm, which moves the attention under
        # norm_attn_norm. Addressing layer.self_attn there raises, but a property
        # that fell back to walking the tree would find nothing and ablate
        # nothing -- an intervention that reports an effect of zero.
        pg = stub(layers=1, heads=3, head_dim=4, fused=True)
        projection = pg.model.model.layers[0].norm_attn_norm.self_attn.o_proj
        self.assertIs(pg.attention_blocks[0].o_proj, projection)
        x = torch.arange(12, dtype=torch.float32).reshape(1, 12)
        with ablated(pg, Component("attention_head", 0, 1)):
            observed = projection(x)
        expected_input = x.clone()
        expected_input[:, 4:8] = 0.0
        self.assertTrue(torch.allclose(observed, projection(expected_input)))

    def test_the_layout_is_read_from_the_config_and_not_from_the_module_tree(self):
        # A config that says fused over a tree that is not must raise rather than
        # quietly address whatever else is there.
        pg = stub(layers=1, fused=False)
        pg.config.fused_attention_norm = True
        with self.assertRaises(AttributeError):
            _ = pg.attention_blocks

    def test_the_component_grid_is_every_head_then_every_block(self):
        pg = stub(layers=2, heads=3, head_dim=4)
        grid = components(pg)
        self.assertEqual(len(grid), 2 * 3 + 2)
        self.assertEqual(grid[0].label, "attention_head.L0H0")
        self.assertEqual(grid[-1].label, "moe_block.L1")
        self.assertEqual(
            [component.kind for component in grid],
            ["attention_head"] * 6 + ["moe_block"] * 2,
        )




#: A tokenizer and a rendering small enough to state the scored span on.
#:
#: The ids follow the released vocabulary's arrangement: padding and the special
#: tokens first, then the direction markers as **ordinary vocabulary entries** --
#: which is why the span cannot be built from ``all_special_ids`` -- then the
#: residues. ``_ClmPreparer`` renders exactly what the released
#: ``ProGen3BatchPreparer.prepare_clm`` renders, ``<bos>`` + ``"1"`` + sequence +
#: ``"2"`` + ``<eos>``, with ``labels`` identical to ``input_ids``.
MARKER_VOCAB = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<mask>": 3,
    "1": 4,
    "2": 5,
    "A": 6,
    "C": 7,
    "D": 8,
    "E": 9,
    "F": 10,
}

#: ``ACDEF`` under that rendering: the residue ids, and the labels one target
#: column is shifted from.
RESIDUE_IDS = [6, 7, 8, 9, 10]
N_TO_C_LABELS = [1, 4, 6, 7, 8, 9, 10, 5, 2]


class _MarkerTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return dict(MARKER_VOCAB)


class _ClmPreparer:
    """The released CLM rendering, for one short record."""

    def __init__(self) -> None:
        self.tokenizer = _MarkerTokenizer()

    def get_batch_kwargs(self, sequences, *, device, reverse=False):
        rows = []
        for sequence in sequences:
            text = "1" + sequence + "2"
            if reverse:
                text = text[::-1]
            rows.append(
                [MARKER_VOCAB["<bos>"]]
                + [MARKER_VOCAB[character] for character in text]
                + [MARKER_VOCAB["<eos>"]]
            )
        width = max(len(row) for row in rows)
        ids = torch.tensor(
            [row + [MARKER_VOCAB["<pad>"]] * (width - len(row)) for row in rows],
            dtype=torch.long,
        ).to(device)
        return {"input_ids": ids, "labels": ids, "position_ids": ids.clone(), "sequence_ids": ids.clone()}


def marker_stub() -> ProGen3:
    """A handle whose preparer renders the released CLM format and whose model is a stub."""

    return ProGen3(
        model=_Model(1, 3, 4),
        config=SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=3,
            hidden_size=12,
            fused_attention_norm=False,
            pad_token_id=0,
        ),
        preparer=_ClmPreparer(),
        device=torch.device("cpu"),
        checkpoint=Path("/nonexistent"),
    )


class TheScoredSpanIsTheResidues(unittest.TestCase):
    """No marker the released rendering writes may be a scored target.

    ``prepare_clm`` puts ``"1"`` inside the labels after the BOS, and ``"2"``
    and ``<eos>`` at the end, and the author's own scorer takes every non-pad
    label after the first position. Read that way, this lineage scores three
    near-deterministic marker positions per record that no other arm of the
    protein side scores: ProGen2's direction marker is position 0 and is never a
    target, and ProtGPT2's, ZymCTRL's, ProteinGLM's and the residue-level arms'
    renderings carry no terminal marker at all. Under a held-out unigram of the
    scored cohort that arm-specific term is priced as rare and the model assigns
    it near-zero likelihood, so it enters the context-information figure with no
    analogue on any arm it is compared against.

    The exclusion is not restated here: the expected positions are derived from
    :func:`src.capability.models.progen3.non_residue_token_ids`, so a token that joins or
    leaves :data:`src.capability.models.progen3.NON_RESIDUE_TOKENS` moves these assertions
    with it.
    """

    def test_the_scored_columns_are_the_residue_positions(self):
        pg = marker_stub()
        batch = pg.batch(["ACDEF"])
        mask = residue_target_mask(pg, batch["labels"])
        # One target column per skipped label: "1" is a target because only the
        # BOS precedes it, and "2" and <eos> are the last two.
        self.assertEqual(
            mask[0].tolist(), [False, True, True, True, True, True, False, False]
        )
        self.assertEqual(int(mask[0].sum()), len(RESIDUE_IDS))

    def test_the_targets_are_the_residues_and_no_declared_marker(self):
        pg = marker_stub()
        rows = n_to_c_target_rows(pg, ["ACDEF"], max_len=32)
        self.assertEqual(rows[0].tolist(), RESIDUE_IDS)
        for value in non_residue_token_ids(pg):
            self.assertNotIn(int(value), rows[0].tolist())

    def test_the_reversed_rendering_scores_its_residues_too(self):
        pg = marker_stub()
        batch = pg.batch(["ACDEF"], reverse=True)
        self.assertEqual(batch["input_ids"][0].tolist(), [1, 5, 10, 9, 8, 7, 6, 4, 2])
        mask = residue_target_mask(pg, batch["labels"])
        self.assertEqual(mask[0].tolist(), [False, True, True, True, True, True, False, False])

    def test_the_scoring_path_reads_the_labels_it_was_given(self):
        # The rendering is not the fix: the ids the model is fed, and the targets
        # the shift produces from them, still carry the markers. Only the mask
        # that selects the scored columns moved.
        pg = marker_stub()
        batch = pg.batch(["ACDEF"])
        self.assertEqual(batch["input_ids"][0].tolist(), N_TO_C_LABELS)
        self.assertEqual(batch["labels"][0].tolist(), N_TO_C_LABELS)
        _, targets, mask = scored_logits(pg, batch)
        self.assertEqual(targets[0].tolist(), N_TO_C_LABELS[1:])
        self.assertEqual(
            mask[0].tolist(), [False, True, True, True, True, True, False, False]
        )

    def test_the_content_span_and_the_scored_span_are_one_declaration(self):
        # name -> id, resolved once. A tokenizer the released rendering uses but
        # that does not carry a declared marker is refused rather than masked
        # around, which is what keeps the two spans from drifting apart.
        pg = marker_stub()
        self.assertEqual(
            non_residue_token_ids(pg),
            tuple(MARKER_VOCAB[name] for name in NON_RESIDUE_TOKENS),
        )
        pg.preparer.tokenizer = SimpleNamespace(get_vocab=lambda: {"<pad>": 0})
        with self.assertRaises(RuntimeError) as caught:
            residue_target_mask(pg, pg.batch(["ACDEF"])["labels"])
        self.assertIn("non-residue positions", str(caught.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
