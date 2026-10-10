"""The token grid the intervention needs, the decoding policy, and the sampler.

Three things are tested without a checkpoint, on a stub whose tokenizer and model
are small enough to reason about exactly:

*the admissibility gate*, which must refuse a tokenisation on which "the residue
at position p" is not one logit column, or on which a prefix retokenises -- the
two properties a multi-residue BPE arm fails, and the reason such an arm is
outside this design rather than inside it with a caveat;

*the decoding policy*, which must agree with the library's nucleus rule wherever
that rule is well defined, and must not depend on sort order where it is not;

*the sampler*, which must put the forced residue at the anchor and nowhere else,
stop at the span end, and record a non-residue token as a censored outcome rather
than as a residue.

One test uses a real staged checkpoint's tokenizer, because the gate's whole
purpose is to measure a property of a real tokenizer rather than trust a
declaration; it skips when the checkpoint is not on this host.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.forcing import forcing_design as D  # noqa: E402
from src.capability.forcing.forced_completion import (  # noqa: E402
    GRID_PROBE,
    ResidueGrid,
    completion_flags,
    render_prefix,
    residue_distribution,
    residue_grid,
    residue_ids_of,
    sample_cell,
    teacher_forced_cell,
    warp,
)

#: The stub vocabulary: twenty residues, one marker, one end token.
MARKER_ID = 0
END_ID = 21
RESIDUE_IDS = tuple(range(1, 21))
VOCAB = 22


class StubTokenizer:
    """A one-token-per-residue tokenizer with a single leading marker."""

    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token_id = END_ID

    def __init__(self, *, merge: bool = False, split_single: bool = False) -> None:
        self.merge = merge
        self.split_single = split_single
        self.ids = {residue: RESIDUE_IDS[index] for index, residue in enumerate(AA20)}

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        if self.split_single and len(text) == 1:
            return {"input_ids": [self.ids[text], self.ids[text]]}
        ids = []
        index = 0
        while index < len(text):
            if text.startswith("1", index) and index == 0:
                ids.append(MARKER_ID)
                index += 1
                continue
            if self.merge and index + 1 < len(text) and text[index] in self.ids:
                # A two-residue merge, which is what makes a prefix retokenise.
                ids.append(VOCAB + self.ids[text[index]])
                index += 2
                continue
            ids.append(self.ids[text[index]])
            index += 1
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids):
        reverse = {value: key for key, value in self.ids.items()}
        return "".join(reverse.get(int(value), "<end>") for value in ids)


class StubModel(torch.nn.Module):
    """Returns a fixed logit row, optionally one that ends the sequence."""

    def __init__(self, *, favour=None, end_at=None) -> None:
        super().__init__()
        self.favour = favour
        self.end_at = end_at
        self.config = type("Config", (), {"vocab_size": VOCAB})()

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, **kwargs):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, VOCAB), -20.0)
        logits[..., list(RESIDUE_IDS)] = 0.0
        if self.favour is not None:
            logits[..., self.favour] = 6.0
        if self.end_at is not None and length >= self.end_at:
            logits[..., END_ID] = 30.0
        return type("Output", (), {"logits": logits})()


class StubSpec:
    name = "stub"
    input_format = "n_to_c_control"
    tokenisation = "residue"
    modality = "protein"


class StubArm:
    def __init__(self, **kwargs) -> None:
        self.spec = StubSpec()
        self.tokenizer = StubTokenizer(
            merge=kwargs.pop("merge", False),
            split_single=kwargs.pop("split_single", False),
        )
        self.model = StubModel(**kwargs)

    @property
    def name(self):
        return "stub"


def stub_grid() -> ResidueGrid:
    return ResidueGrid(arm="stub", markers=1, residue_ids=RESIDUE_IDS, evidence={})


class TestPrefixRendering:
    def test_each_declared_rendering_places_its_own_markers(self):
        assert render_prefix(StubArm(), "ACDE") == "1ACDE"

    def test_an_undeclared_rendering_is_refused_by_name(self):
        arm = StubArm()
        arm.spec = type("Spec", (StubSpec,), {"input_format": "fasta_wrapped"})()
        with pytest.raises(ValueError, match="has no prefix rendering"):
            render_prefix(arm, "ACDE")

    def test_the_refusal_explains_why_a_wrapped_rendering_has_no_prefix(self):
        arm = StubArm()
        arm.spec = type("Spec", (StubSpec,), {"input_format": "ec_conditioned"})()
        with pytest.raises(ValueError, match="trained input"):
            render_prefix(arm, "ACDE")


class TestTokenGrid:
    def test_a_one_token_per_residue_grid_is_admitted(self):
        grid = residue_grid(StubArm())
        assert grid.markers == 1
        assert grid.residue_ids == RESIDUE_IDS
        assert grid.evidence["probe_ids"] == len(GRID_PROBE) + 1

    def test_the_grid_maps_ids_back_to_residues(self):
        grid = residue_grid(StubArm())
        assert grid.residue_of_id[RESIDUE_IDS[0]] == AA20[0]

    def test_a_merging_tokenisation_is_refused(self):
        """ProtGPT2's failure, in miniature: a prefix retokenises, so forcing a
        position would also change the positions before it."""

        with pytest.raises(ValueError, match="one token per residue|is not the prefix"):
            residue_grid(StubArm(merge=True))

    def test_a_multi_token_residue_is_refused_before_anything_is_generated(self):
        with pytest.raises(ValueError, match="not a single token"):
            residue_grid(StubArm(split_single=True))

    def test_two_residues_sharing_one_id_is_refused(self):
        arm = StubArm()
        arm.tokenizer.ids["C"] = arm.tokenizer.ids["A"]
        with pytest.raises(ValueError, match="share one token id"):
            residue_grid(arm)

    def test_residue_ids_of_refuses_a_non_canonical_residue(self):
        with pytest.raises(ValueError, match="no token id"):
            residue_ids_of(RESIDUE_IDS, "ACX")


class TestDecodingPolicy:
    def test_the_policy_agrees_with_the_library_wherever_it_is_well_defined(self):
        """Untied continuous logits have one nucleus, and the two rules must give
        it. Where a reduced-precision checkpoint ties logits at the boundary the
        library's answer depends on sort order, which is why this design keeps
        every tied token and runs in float32."""

        from transformers.generation.logits_process import TopPLogitsWarper

        generator = np.random.default_rng(0)
        worst = 0.0
        for _ in range(200):
            logits = (generator.normal(size=64) * 3.0).astype(np.float64)
            processed = TopPLogitsWarper(D.TOP_P)(
                None, torch.tensor(logits[None, :])
            ).numpy()[0]
            keep = np.isfinite(processed)
            mass = np.exp(processed[keep] - processed[keep].max())
            library = np.zeros_like(logits)
            library[keep] = mass / mass.sum()
            worst = max(worst, float(np.abs(library - warp(logits)).max()))
        assert worst < 1e-12

    def test_the_policy_keeps_every_token_tied_with_the_nucleus_boundary(self):
        logits = np.log(np.array([0.5, 0.2, 0.1, 0.1, 0.1]))
        probability = warp(logits, top_p=0.8)
        assert probability[2] > 0 and probability[3] > 0 and probability[4] > 0

    def test_the_policy_is_permutation_invariant(self):
        generator = np.random.default_rng(3)
        logits = generator.normal(size=32)
        order = generator.permutation(32)
        assert np.allclose(warp(logits)[order], warp(logits[order]))

    def test_the_retained_mass_is_never_below_the_threshold(self):
        generator = np.random.default_rng(1)
        for _ in range(50):
            logits = generator.normal(size=40) * 2.0
            probability = np.exp(logits - logits.max())
            probability /= probability.sum()
            kept = warp(logits) > 0
            assert probability[kept].sum() >= D.TOP_P - 1e-12

    def test_a_nonfinite_logit_row_is_refused(self):
        with pytest.raises(ValueError, match="finite"):
            warp(np.array([1.0, np.nan, 0.0]))

    def test_the_residue_restriction_renormalises_inside_the_alphabet(self):
        probability = np.zeros(VOCAB)
        probability[RESIDUE_IDS[0]] = 0.25
        probability[RESIDUE_IDS[1]] = 0.25
        probability[END_ID] = 0.5
        restricted = residue_distribution(probability, stub_grid())
        assert restricted.sum() == pytest.approx(1.0)
        assert restricted[0] == pytest.approx(0.5)

    def test_a_policy_leaving_no_residue_mass_is_refused(self):
        probability = np.zeros(VOCAB)
        probability[END_ID] = 1.0
        with pytest.raises(ValueError, match="no mass on any canonical residue"):
            residue_distribution(probability, stub_grid())


class TestSampler:
    wildtype = ("ACDEFGHIKLMNPQRSTVWY" * 6)[:120]

    def test_the_forced_residue_sits_at_the_anchor_and_the_prompt_stops_there(self):
        arm = StubArm()
        grid = stub_grid()
        anchor = 40
        prompt = render_prefix(arm, self.wildtype[:anchor] + "W")
        ids = arm.tokenizer(prompt)["input_ids"]
        assert len(ids) == grid.markers + anchor + 1
        assert ids[-1] == grid.residue_ids[AA20.index("W")]
        assert ids[-2] == grid.residue_ids[AA20.index(self.wildtype[anchor - 1])]

    def test_a_cell_emits_the_requested_span_and_records_the_read_positions(self):
        arm = StubArm()
        cell = sample_cell(
            arm, stub_grid(), wildtype=self.wildtype, anchor=40, forced_residue="W",
            span_end=46, read_positions=[44, 45, 46], draws=4, seed=5, batch_size=4,
        )
        assert cell["new_tokens"] == 6
        assert len(cell["draws"]) == 4
        assert cell["distributions"].shape == (4, 3, len(AA20))
        for row in cell["draws"]:
            assert row["emitted_residues"] == 6
            assert set(row["realised"]) == {"44", "45", "46"}
            assert all(value in AA20 for value in row["realised"].values())

    def test_the_recorded_distribution_is_the_one_the_draw_came_from(self):
        arm = StubArm(favour=RESIDUE_IDS[AA20.index("K")])
        cell = sample_cell(
            arm, stub_grid(), wildtype=self.wildtype, anchor=40, forced_residue="A",
            span_end=44, read_positions=[44], draws=8, seed=1, batch_size=8,
        )
        rows = cell["distributions"][:, 0, :]
        assert np.allclose(rows.sum(axis=1), 1.0, atol=1e-5)
        assert rows[:, AA20.index("K")].min() > 0.5
        realised = [row["realised"]["44"] for row in cell["draws"]]
        assert realised.count("K") >= 6

    def test_sampling_is_reproducible_from_the_seed(self):
        arm = StubArm()
        kwargs = dict(
            wildtype=self.wildtype, anchor=40, forced_residue="W", span_end=46,
            read_positions=[46], draws=6, batch_size=3,
        )
        first = sample_cell(arm, stub_grid(), seed=9, **kwargs)
        second = sample_cell(arm, stub_grid(), seed=9, **kwargs)
        third = sample_cell(arm, stub_grid(), seed=10, **kwargs)
        assert [row["completion"] for row in first["draws"]] == [
            row["completion"] for row in second["draws"]
        ]
        assert [row["completion"] for row in first["draws"]] != [
            row["completion"] for row in third["draws"]
        ]

    def test_a_non_residue_token_censors_the_draw_rather_than_becoming_a_residue(self):
        arm = StubArm(end_at=44)
        cell = sample_cell(
            arm, stub_grid(), wildtype=self.wildtype, anchor=40, forced_residue="A",
            span_end=48, read_positions=[46, 47, 48], draws=4, seed=2, batch_size=4,
        )
        assert cell["censored_draws"] == 4
        for row in cell["draws"]:
            assert row["censored"]
            assert row["terminator"]["token_id"] == END_ID
            assert all(value is None for value in row["realised"].values())
            assert all(residue in AA20 for residue in row["completion"])

    def test_the_sampler_refuses_a_span_outside_the_wild_type(self):
        arm, grid = StubArm(), stub_grid()
        common = dict(wildtype=self.wildtype, anchor=40, forced_residue="A", draws=2, seed=1)
        with pytest.raises(ValueError, match="span ends after the anchor"):
            sample_cell(arm, grid, span_end=40, read_positions=[40], **common)
        with pytest.raises(ValueError, match="span ends after the anchor"):
            sample_cell(arm, grid, span_end=120, read_positions=[119], **common)
        with pytest.raises(ValueError, match="every read position"):
            sample_cell(arm, grid, span_end=46, read_positions=[40], **common)

    def test_the_sampler_refuses_a_non_canonical_forced_residue(self):
        with pytest.raises(ValueError, match="canonical"):
            sample_cell(
                StubArm(), stub_grid(), wildtype=self.wildtype, anchor=40,
                forced_residue="X", span_end=46, read_positions=[46], draws=2, seed=1,
            )

    def test_a_prompt_the_grid_does_not_predict_is_refused(self):
        """A tokenizer whose grid has drifted must stop the cell, not shift every
        position by one."""

        arm = StubArm()
        grid = ResidueGrid(arm="stub", markers=3, residue_ids=RESIDUE_IDS, evidence={})
        with pytest.raises(ValueError, match="token grid and the prompt disagree"):
            sample_cell(
                arm, grid, wildtype=self.wildtype, anchor=40, forced_residue="A",
                span_end=46, read_positions=[46], draws=2, seed=1,
            )


class TestTeacherForced:
    wildtype = ("ACDEFGHIKLMNPQRSTVWY" * 6)[:120]

    def test_one_forward_pass_yields_every_read_position(self):
        cell = teacher_forced_cell(
            StubArm(), stub_grid(), wildtype=self.wildtype, anchor=40,
            forced_residue="W", span_end=48, read_positions=[46, 47, 48],
        )
        assert cell["distributions"].shape == (3, len(AA20))
        assert np.allclose(cell["distributions"].sum(axis=1), 1.0, atol=1e-5)
        assert cell["wild_type_residues"] == {
            str(position): self.wildtype[position] for position in (46, 47, 48)
        }

    def test_the_prompt_holds_the_forced_residue_and_wild_type_elsewhere(self):
        arm, grid = StubArm(), stub_grid()
        cell = teacher_forced_cell(
            arm, grid, wildtype=self.wildtype, anchor=40, forced_residue="W",
            span_end=48, read_positions=[48],
        )
        assert cell["prompt_tokens"] == grid.markers + 49

    def test_both_modes_read_one_decoding_policy(self):
        """The teacher-forced conditional and the sampled draws must come from the
        same distribution when the model is position-independent; otherwise the
        sampled-minus-teacher-forced decomposition measures the two
        implementations rather than the model."""

        arm = StubArm(favour=RESIDUE_IDS[AA20.index("K")])
        exact = teacher_forced_cell(
            arm, stub_grid(), wildtype=self.wildtype, anchor=40, forced_residue="A",
            span_end=44, read_positions=[44],
        )["distributions"][0]
        sampled = sample_cell(
            arm, stub_grid(), wildtype=self.wildtype, anchor=40, forced_residue="A",
            span_end=44, read_positions=[44], draws=4, seed=1, batch_size=4,
        )["distributions"][0, 0]
        assert np.allclose(np.asarray(exact), np.asarray(sampled), atol=1e-6)


class TestCompletionFlags:
    def test_a_repeat_is_flagged_and_a_varied_completion_is_not(self):
        assert completion_flags("A" * 40)["repeat_flagged"]
        generator = np.random.default_rng(0)
        varied = "".join(AA20[index] for index in generator.integers(20, size=40))
        assert not completion_flags(varied)["repeat_flagged"]

    def test_the_hydrophobic_fraction_uses_the_declared_set(self):
        assert completion_flags("AAAA")["hydrophobic_fraction"] == 1.0
        assert completion_flags("KKKK")["hydrophobic_fraction"] == 0.0

    def test_an_empty_completion_carries_no_fraction_and_is_not_flagged(self):
        flags = completion_flags("")
        assert flags["hydrophobic_fraction"] is None and not flags["repeat_flagged"]


class TestRealTokenizer:
    """The gate exists to measure a real tokenizer, so one is measured."""

    def _tokenizer(self, name):
        from transformers import AutoTokenizer

        from src.capability.core.arms import PANEL, arm_spec

        spec = PANEL[name] if name in PANEL else arm_spec(name)
        if not spec.path.is_dir():
            pytest.skip(f"{name} is not staged on this host")
        return spec, AutoTokenizer.from_pretrained(str(spec.path), trust_remote_code=True)

    def test_a_declared_arm_passes_the_grid_gate(self):
        spec, tokenizer = self._tokenizer("progen2-medium")
        arm = type("Arm", (), {"spec": spec, "tokenizer": tokenizer, "name": spec.name})()
        grid = residue_grid(arm)
        assert grid.markers == 1 and len(grid.residue_ids) == 20

    def test_protgpt2_is_refused_by_the_grid_gate(self):
        """Measured rather than assumed: a multi-residue BPE arm cannot carry a
        position-level forcing intervention, which is why it is not in the panel."""

        spec, tokenizer = self._tokenizer("protgpt2")
        arm = type("Arm", (), {"spec": spec, "tokenizer": tokenizer, "name": spec.name})()
        with pytest.raises(ValueError):
            residue_grid(arm)
        assert "protgpt2" in D.UNSELECTED_ARMS
