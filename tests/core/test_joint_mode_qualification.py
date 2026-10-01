"""The declarations and guards behind the joint-checkpoint qualification stage.

Nothing here needs a GPU, a network or a 7B checkpoint. What is covered is the
part of ``src.capability.models.joint_modes`` and
``scripts/capability/stages/joint_mode_qualification.py`` that can be wrong *silently*:
a rendering that merged residues, a residue alphabet the tokenizer does not
carry, a scored span that quietly included a delimiter, a held-out reference that
overlapped the sample it normalises, and a below-threshold mode reported as a
failure rather than as unmeasurable.

The tokenizers are stubs, and they are stubs of exactly the behaviour that caused
the defect: a vocabulary that merges adjacent residues into two-residue pieces
unless something stops it. ``StubTokenizer(split_marker=...)`` carries Galactica's
``Split``-and-remove pretokenizer rule, and the same stub without it is a
tokenizer that ignores the escape -- which is what an unaided ``AutoTokenizer``
call amounts to and what costs about 2.9 nats/token on the real checkpoint.

The same merging stub is also what a *token-unit* family reads as its declared
format, which is the point of the symbol-unit declaration: the two cases produce
the same record and only one of them is a defect. What separates them here is
``symbol_unit``, never a call site.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import io
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.capability.models import joint_modes as JM  # noqa: E402
from src.capability.core.arms import AA20, Cohort, sampling_record, selected_positions  # noqa: E402
from src.capability.core.budget import (  # noqa: E402
    MEASURABLE,
    MIN_CONTEXT_INFORMATION_NATS,
    SCREENING_CONTEXT_INFORMATION_NATS,
    UNMEASURABLE,
)


from scripts.capability.entrypoints import stage_path

def _load_stage(filename: str):
    """Import a stage whose module name starts with a digit."""

    path = stage_path(REPO, filename)
    spec = importlib.util.spec_from_file_location(f"_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE = _load_stage("joint_mode_qualification.py")


# ------------------------------------------------------------------ stub tokenizer


class StubTokenizer:
    """A tokenizer with the one behaviour that matters: it merges residues.

    Longest match wins over a vocabulary that contains every two-residue pair, so
    a bare protein string tokenises to roughly half as many pieces as it has
    residues -- the real defect, in twenty lines. ``split_marker`` reproduces
    Galactica's released ``Split``/``Removed`` pretokenizer rule: the literal is
    consumed and each fragment is tokenised on its own, which is what makes the
    escape work.
    """

    def __init__(
        self,
        *,
        specials: tuple[str, ...],
        split_marker: str | None = None,
        bos_id: int | None = None,
        bos_token: str = "<s>",
        alias: dict[str, str] | None = None,
    ) -> None:
        self.split_marker = split_marker
        self.bos_id = bos_id
        self._vocab: dict[str, int] = {}
        self._inverse: dict[int, str] = {}
        if bos_id is not None:
            # A real tokenizer spells its beginning-of-sequence token, and both the
            # spelled-run rule and a declared rendering prefix read the spelling:
            # InstructProtein's ``bos_token`` is "</s>", not "<s>", so the stub
            # takes the spelling as an argument rather than assuming one.
            self._inverse[bos_id] = bos_token
        self._add("<unk>")
        for token in specials:
            self._add(token)
        for residue in AA20:
            self._add(residue)
        for left in AA20:
            for right in AA20:
                self._add(left + right)
        for character in "Ƥ abcdefghijklmnopqrstuvwxyz.,\n#":
            self._add(character)
        for token, target in (alias or {}).items():
            self._vocab[token] = self._vocab[target]
        if bos_id is not None:
            # Added after the vocabulary is built, so that declaring the prefix
            # token does not shift every other id: its id is the tokenizer's own
            # rather than the next free one.
            self._vocab[bos_token] = bos_id
        self._longest = max(len(token) for token in self._vocab)
        self.unk_id = self._vocab["<unk>"]

    def _add(self, token: str) -> None:
        if token in self._vocab:
            return
        index = len(self._vocab) + 10
        self._vocab[token] = index
        self._inverse[index] = token

    def __len__(self) -> int:
        return len(self._vocab) + 10

    def convert_tokens_to_ids(self, token: str):
        return self._vocab.get(token)

    def convert_ids_to_tokens(self, index: int):
        return self._inverse.get(int(index))

    def _fragment(self, text: str) -> list[int]:
        ids: list[int] = []
        cursor = 0
        while cursor < len(text):
            for length in range(min(self._longest, len(text) - cursor), 0, -1):
                candidate = text[cursor : cursor + length]
                if candidate in self._vocab:
                    ids.append(self._vocab[candidate])
                    cursor += length
                    break
            else:
                ids.append(self.unk_id)
                cursor += 1
        return ids

    def __call__(self, text: str, return_tensors=None) -> dict[str, list[int]]:
        fragments = (
            text.split(self.split_marker) if self.split_marker else [text]
        )
        ids: list[int] = [] if self.bos_id is None else [self.bos_id]
        for fragment in fragments:
            ids.extend(self._fragment(fragment))
        return {"input_ids": ids}


def galactica_stub(*, honours_the_split_rule: bool = True) -> StubTokenizer:
    return StubTokenizer(
        specials=("[START_AMINO]", "[END_AMINO]"),
        split_marker=JM.GALACTICA_SPLIT_MARKER if honours_the_split_rule else None,
    )


class MergingBeyondOneResidue(StubTokenizer):
    """Per-residue for one residue, merged for two.

    The state a resolution that only probed single residues would accept and a
    residue-unit claim must never be made in. Nothing about it is visible until a
    real sequence is rendered, which is why resolution renders one.
    """

    def __init__(self) -> None:
        super().__init__(
            specials=("[START_AMINO]", "[END_AMINO]"),
            split_marker=JM.GALACTICA_SPLIT_MARKER,
        )

    def __call__(self, text: str, return_tensors=None) -> dict[str, list[int]]:
        if text.count(self.split_marker) <= 2:
            return super().__call__(text, return_tensors=return_tensors)
        return {"input_ids": self._fragment(text.replace(self.split_marker, ""))}


def instructprotein_stub(*, drop: tuple[str, ...] = (), alias=None) -> StubTokenizer:
    residues = tuple(f"Ƥ{residue}" for residue in AA20 if residue not in drop)
    return StubTokenizer(
        specials=("<protein>", "</protein>") + residues,
        bos_id=2,
        # The staged tokenizer's own spelling: ``bos_token`` is "</s>" at id 2,
        # which is also its end-of-sequence token.
        bos_token="</s>",
        alias=alias,
    )


def prollama_stub() -> StubTokenizer:
    """A tokenizer that merges residues and whose delimiters are NOT its tokens.

    The two properties that matter about the real staged ProLLaMA_Stage_1
    tokenizer, in stub form: ``Seq=<`` is spelled out of ordinary pieces rather
    than carried as one, and a residue string comes back as multi-residue pieces.
    ``Seq``/``=``/``<``/``>`` are added as pieces so the prefix is spellable at
    all, but none of them is the declared ``Seq=<``.
    """

    return StubTokenizer(
        specials=("Seq", "=", "<", ">", "[", "]", "Generate", "by", "superfamily"),
        bos_id=2,
    )


SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"


# ------------------------------------------------------------------ declarations


def _declaration(**overrides) -> JM.JointRendering:
    """A minimal declaration, so a refusal test names only the field it is about."""

    fields = {
        "name": "broken",
        "symbol_unit": JM.RESIDUE_UNIT,
        "protein_start": "<a>",
        "protein_end": "</a>",
        "residue_escape": "^",
        "escape_before_end_delimiter": False,
        "protein_context_template": None,
        "prefix_marker": None,
        "scored_target_rule": JM.BETWEEN_DELIMITERS,
        "residue_subspace_disjoint_from_text": False,
        "note": "",
    }
    fields.update(overrides)
    return JM.JointRendering(**fields)


class RenderingDeclaration(unittest.TestCase):
    def test_every_qualified_family_is_declared_in_one_place(self):
        self.assertEqual(
            sorted(JM.RENDERING_NAMES), ["galactica", "instructprotein", "prollama"]
        )
        self.assertIs(JM.rendering("galactica"), JM.JOINT_RENDERINGS["galactica"])

    def test_an_unknown_rendering_name_is_refused(self):
        with self.assertRaises(KeyError):
            JM.rendering("esm2")

    def test_the_stage_offers_exactly_the_declared_families_and_refuses_the_rest(self):
        parser = STAGE.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--checkpoint", "/nowhere", "--rendering", "esm2"])
        for name in JM.RENDERING_NAMES:
            parsed = parser.parse_args(["--checkpoint", "/nowhere", "--rendering", name])
            self.assertEqual(parsed.rendering, name)

    def test_the_checkpoint_is_required_because_it_is_not_a_panel_arm(self):
        parser = STAGE.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--rendering", "galactica"])

    def test_the_declared_strings_are_the_ones_the_checkpoints_were_trained_on(self):
        galactica = JM.rendering("galactica")
        marker = JM.GALACTICA_SPLIT_MARKER
        self.assertEqual(
            galactica.render_protein("MK"),
            f"[START_AMINO]{marker}M{marker}K{marker}[END_AMINO]",
        )
        self.assertEqual(
            galactica.render_protein("MK", variant=JM.NAIVE), "[START_AMINO]MK[END_AMINO]"
        )
        self.assertEqual(
            galactica.render_protein("MK", context="Serine protease"),
            f"# Serine protease\n\n[START_AMINO]{marker}M{marker}K{marker}[END_AMINO]",
        )
        instruct = JM.rendering("instructprotein")
        self.assertEqual(instruct.render_protein("MK"), "<protein>ƤMƤK</protein>")
        self.assertEqual(
            instruct.render_protein("MK", variant=JM.NAIVE), "<protein>MK</protein>"
        )
        self.assertEqual(
            instruct.render_protein("MK", context="I would like a protein."),
            "Instruction: I would like a protein.\nOutput: <protein>ƤMƤK</protein>",
        )

    def test_a_residue_unit_declaration_with_no_escape_is_refused(self):
        # The escape IS the naive control, so a residue-unit family cannot
        # declare an empty one and call it the trained format.
        with self.assertRaises(ValueError):
            _declaration(symbol_unit=JM.RESIDUE_UNIT, residue_escape="")

    def test_a_token_unit_declaration_that_also_declares_an_escape_is_refused(self):
        # An escape that reached the per-residue alphabet would make the family a
        # residue-unit one; one that did not would not be an escape.
        with self.assertRaises(ValueError) as raised:
            _declaration(symbol_unit=JM.TOKEN_UNIT, residue_escape="^")
        self.assertIn("symbol unit", str(raised.exception))

    def test_a_declaration_with_an_undeclared_symbol_unit_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            _declaration(symbol_unit="per_residue_ish", residue_escape="^")
        self.assertIn("symbol unit", str(raised.exception))

    def test_a_declaration_with_an_undeclared_scoring_rule_is_refused(self):
        with self.assertRaises(ValueError):
            _declaration(
                symbol_unit=JM.RESIDUE_UNIT,
                residue_escape="^",
                scored_target_rule="whatever_looks_right",
            )

    def test_a_non_canonical_sequence_is_refused_rather_than_rendered(self):
        with self.assertRaises(ValueError):
            JM.rendering("galactica").render_protein("MKTX")

    def test_an_unknown_variant_is_refused(self):
        with self.assertRaises(ValueError):
            JM.rendering("galactica").render_protein("MK", variant="escaped-ish")

    def test_only_the_family_with_dedicated_residue_tokens_declares_a_text_subspace(self):
        # This is what the text-mode control is gated on: Galactica's residues are
        # ordinary capital letters, so a "residue probability mass" on its text
        # mode would identify nothing. ProLLaMA's are ordinary letters too.
        self.assertFalse(JM.rendering("galactica").residue_subspace_disjoint_from_text)
        self.assertTrue(JM.rendering("instructprotein").residue_subspace_disjoint_from_text)
        self.assertFalse(JM.rendering("prollama").residue_subspace_disjoint_from_text)

    def test_prollama_declares_the_token_as_its_symbol_unit_and_the_others_the_residue(self):
        self.assertEqual(JM.rendering("prollama").symbol_unit, JM.TOKEN_UNIT)
        for name in ("galactica", "instructprotein"):
            self.assertEqual(JM.rendering(name).symbol_unit, JM.RESIDUE_UNIT, name)

    def test_prollama_renders_the_format_the_lineage_was_trained_on(self):
        prollama = JM.rendering("prollama")
        self.assertEqual(prollama.render_protein("MK"), "Seq=<MK>")
        self.assertEqual(
            prollama.render_protein("MK", context="Ferritin-like superfamily"),
            "[Generate by superfamily] Superfamily=<Ferritin-like superfamily> Seq=<MK>",
        )

    def test_only_a_family_with_an_escape_has_a_naive_control_at_all(self):
        # Requirement 5: the control is defined by the escape, and a family with
        # no escape refuses to render it rather than returning the declared form
        # under the control's name.
        self.assertFalse(JM.rendering("prollama").naive_control_available)
        for name in ("galactica", "instructprotein"):
            self.assertTrue(JM.rendering(name).naive_control_available, name)
        with self.assertRaises(ValueError) as raised:
            JM.rendering("prollama").render_protein("MK", variant=JM.NAIVE)
        self.assertIn("no per-residue escape", str(raised.exception))


# ----------------------------------------------------- resolution against a tokenizer


class ResolutionAgainstATokenizer(unittest.TestCase):
    def test_the_residue_alphabet_is_derived_from_the_declaration_itself(self):
        for name, tokenizer in (
            ("galactica", galactica_stub()),
            ("instructprotein", instructprotein_stub()),
        ):
            resolved = JM.resolve(tokenizer, name)
            self.assertEqual(sorted(resolved.residue_ids), sorted(AA20))
            self.assertEqual(len(set(resolved.residue_ids.values())), len(AA20))
            self.assertNotIn(resolved.start_id, set(resolved.residue_ids.values()))
            self.assertNotIn(resolved.end_id, set(resolved.residue_ids.values()))

    def test_a_tokenizer_that_does_not_carry_a_declared_residue_is_refused(self):
        # The declared alphabet is twenty tokens; a tokenizer missing one spells
        # it as two pieces, which is exactly the state a residue-level claim must
        # never be made in.
        with self.assertRaises(ValueError) as raised:
            JM.resolve(instructprotein_stub(drop=("W",)), "instructprotein")
        self.assertIn("per-residue alphabet", str(raised.exception))

    def test_two_residues_sharing_one_token_id_are_refused(self):
        with self.assertRaises(ValueError) as raised:
            JM.resolve(instructprotein_stub(alias={"ƤC": "ƤA"}), "instructprotein")
        self.assertIn("share a token id", str(raised.exception))

    def test_a_tokenizer_without_the_declared_delimiters_is_refused(self):
        with self.assertRaises(ValueError):
            JM.resolve(galactica_stub(), "instructprotein")

    def test_a_per_residue_family_on_a_merging_tokenizer_is_refused_at_resolution(self):
        # The load-bearing refusal, moved ahead of the weights: this tokenizer
        # honours the escape for one residue and merges two, so a probe that only
        # ever saw one residue would have accepted it.
        with self.assertRaises(ValueError) as raised:
            JM.resolve(MergingBeyondOneResidue(), "galactica")
        self.assertIn("merged residues", str(raised.exception))

    def test_a_per_residue_family_whose_escape_is_ignored_is_refused_at_resolution(self):
        with self.assertRaises(ValueError):
            JM.resolve(galactica_stub(honours_the_split_rule=False), "galactica")

    def test_a_token_unit_family_resolves_without_its_delimiters_being_tokens(self):
        # 'Seq=<' is not a token of this vocabulary and never will be; the family
        # locates its span by spelling, so resolution must not demand one.
        tokenizer = prollama_stub()
        self.assertIsNone(tokenizer.convert_tokens_to_ids("Seq=<"))
        resolved = JM.resolve(tokenizer, "prollama")
        self.assertIsNone(resolved.start_id)
        self.assertIsNone(resolved.end_id)
        self.assertEqual(sorted(resolved.residue_ids), sorted(AA20))

    def test_the_token_unit_support_is_every_residue_spelling_token(self):
        resolved = JM.resolve(prollama_stub(), "prollama")
        support = set(resolved.scored_target_ids)
        # Twenty singles plus every two-residue piece the stub carries, and
        # nothing that spells anything else.
        self.assertEqual(len(support), len(AA20) + len(AA20) ** 2)
        self.assertTrue(set(resolved.residue_ids.values()) <= support)
        for token in ("Seq", "=", "<", ">", "<unk>"):
            self.assertNotIn(resolved.tokenizer.convert_tokens_to_ids(token), support, token)

    def test_a_residue_unit_family_supports_exactly_its_twenty_residue_ids(self):
        for name, tokenizer in (
            ("galactica", galactica_stub()),
            ("instructprotein", instructprotein_stub()),
        ):
            resolved = JM.resolve(tokenizer, name)
            self.assertEqual(
                resolved.scored_target_ids,
                tuple(sorted(resolved.residue_ids.values())),
                name,
            )


# -------------------------------------------------------- the per-residue guard


class PerResidueVerification(unittest.TestCase):
    """The defect this stage exists to prevent, from both directions."""

    def test_the_escaped_rendering_passes_and_is_one_token_per_residue(self):
        resolved = JM.resolve(galactica_stub(), "galactica")
        record = resolved.render(SEQUENCE)
        self.assertEqual(record.n_scored_tokens, len(SEQUENCE))
        self.assertEqual(record.residues_per_scored_token, 1.0)
        resolved.verify_one_token_per_residue(record)

    def test_the_naive_rendering_fails_verification_and_really_did_merge(self):
        resolved = JM.resolve(galactica_stub(), "galactica")
        naive = resolved.render(SEQUENCE, variant=JM.NAIVE)
        self.assertGreater(naive.residues_per_scored_token, 1.0)
        with self.assertRaises(ValueError) as raised:
            resolved.verify_one_token_per_residue(naive)
        self.assertIn("merged residues", str(raised.exception))

    def test_a_tokenizer_that_ignores_the_escape_raises_on_the_DECLARED_rendering(self):
        # The load-bearing case: the escape is present in the rendered string and
        # the tokenizer does not carry the Split rule that consumes it. The run
        # must stop, because the merged measurement looks exactly like a valid one.
        resolved = JM.resolve(
            galactica_stub(honours_the_split_rule=True), "galactica"
        )
        deaf = JM.JointTokenisation(
            declaration=resolved.declaration,
            tokenizer=galactica_stub(honours_the_split_rule=False),
            start_id=resolved.start_id,
            end_id=resolved.end_id,
            prefix_marker_ids=resolved.prefix_marker_ids,
            residue_ids=resolved.residue_ids,
            scored_target_ids=resolved.scored_target_ids,
        )
        with self.assertRaises(ValueError):
            deaf.render(SEQUENCE)

    def test_the_same_guard_holds_for_the_identity_rule_family(self):
        resolved = JM.resolve(instructprotein_stub(), "instructprotein")
        resolved.verify_one_token_per_residue(resolved.render(SEQUENCE))
        naive = resolved.render(SEQUENCE, variant=JM.NAIVE)
        with self.assertRaises(ValueError):
            resolved.verify_one_token_per_residue(naive)

    def test_the_declared_variant_is_verified_automatically_with_no_opt_out(self):
        # `render` takes no flag that skips the check, so a caller cannot obtain
        # an unverified declared record; only the naive control, whose name says
        # what it is, reaches a scored span without it.
        resolved = JM.resolve(galactica_stub(), "galactica")
        signature = inspect.signature(resolved.render)
        self.assertEqual(
            sorted(signature.parameters), ["context", "sequence", "variant"]
        )
        self.assertEqual(signature.parameters["variant"].default, JM.DECLARED)

    def test_the_guard_takes_no_bypass_parameter_and_the_unit_decides_instead(self):
        # Requirement 1 in one assertion: the per-residue guard's signature is the
        # record and nothing else, so the residue/token distinction cannot be made
        # at a call site. `verify_symbol_unit` is the only thing that chooses.
        resolved = JM.resolve(galactica_stub(), "galactica")
        for method in (resolved.verify_one_token_per_residue, resolved.verify_symbol_unit):
            self.assertEqual(sorted(inspect.signature(method).parameters), ["record"])

    def test_a_token_unit_family_does_not_raise_on_merged_tokens(self):
        # The same merging tokenizer that is a stopped run for Galactica is the
        # declared, trained behaviour here, and the record says how far it merged
        # rather than refusing to exist.
        resolved = JM.resolve(prollama_stub(), "prollama")
        record = resolved.render(SEQUENCE)
        self.assertLess(record.n_scored_tokens, len(SEQUENCE))
        self.assertGreater(record.residues_per_scored_token, 1.0)
        resolved.verify_symbol_unit(record)

    def test_the_unit_and_not_the_call_site_decides_which_verification_runs(self):
        # The two families are handed structurally identical merged records. Only
        # the declared symbol unit separates a defect from a trained format.
        merging = JM.resolve(prollama_stub(), "prollama")
        per_residue = JM.resolve(galactica_stub(), "galactica")
        merged = per_residue.render(SEQUENCE, variant=JM.NAIVE)
        self.assertGreater(merged.residues_per_scored_token, 1.0)
        with self.assertRaises(ValueError):
            per_residue.verify_symbol_unit(merged)
        merging.verify_symbol_unit(merging.render(SEQUENCE))

    def test_a_token_unit_target_outside_the_enumerated_support_is_refused(self):
        # The token-unit counterpart of the stray-id half of the per-residue
        # guard: a span and a support that disagree would leave the held-out
        # unigram unable to represent the sample it normalises.
        resolved = JM.resolve(prollama_stub(), "prollama")
        record = resolved.render(SEQUENCE)
        narrowed = JM.JointTokenisation(
            declaration=resolved.declaration,
            tokenizer=resolved.tokenizer,
            start_id=None,
            end_id=None,
            prefix_marker_ids=resolved.prefix_marker_ids,
            residue_ids=resolved.residue_ids,
            scored_target_ids=tuple(sorted(resolved.residue_ids.values())),
        )
        with self.assertRaises(ValueError) as raised:
            narrowed.verify_symbol_unit(record)
        self.assertIn("outside", str(raised.exception))


# ------------------------------------------------------------- scored positions


class ScoredPositions(unittest.TestCase):
    def test_the_selector_returns_the_residues_and_excludes_both_delimiters(self):
        for name, tokenizer in (
            ("galactica", galactica_stub()),
            ("instructprotein", instructprotein_stub()),
        ):
            resolved = JM.resolve(tokenizer, name)
            record = resolved.render(SEQUENCE)
            ids = record.token_ids
            opening = ids.index(resolved.start_id)
            closing = ids.index(resolved.end_id)
            self.assertEqual(
                list(record.scored_positions),
                list(range(opening + 1, closing)),
                name,
            )
            self.assertNotIn(opening, record.scored_positions)
            self.assertNotIn(closing, record.scored_positions)
            self.assertEqual(
                [ids[position] for position in record.scored_positions],
                [resolved.residue_ids[residue] for residue in SEQUENCE],
                name,
            )

    def test_the_identity_rule_is_what_instructprotein_actually_uses(self):
        resolved = JM.resolve(instructprotein_stub(), "instructprotein")
        rendered = resolved.declaration.render_protein(SEQUENCE)
        by_identity = JM.scored_target_positions(
            resolved.tokenizer,
            resolved.declaration,
            rendered,
            residue_ids=resolved.residue_ids,
        )
        by_position = JM.scored_target_positions(
            resolved.tokenizer,
            resolved.declaration,
            rendered,
            rule=JM.BETWEEN_DELIMITERS,
        )
        self.assertEqual(by_identity, by_position)
        self.assertEqual(len(by_identity), len(SEQUENCE))

    def test_the_identity_rule_refuses_to_run_without_the_declared_alphabet(self):
        resolved = JM.resolve(instructprotein_stub(), "instructprotein")
        with self.assertRaises(ValueError):
            JM.scored_target_positions(
                resolved.tokenizer,
                resolved.declaration,
                resolved.declaration.render_protein(SEQUENCE),
            )

    def test_a_context_prefix_moves_the_span_rather_than_being_scored(self):
        resolved = JM.resolve(galactica_stub(), "galactica")
        bare = resolved.render(SEQUENCE)
        titled = resolved.render(SEQUENCE, context="a serine protease")
        self.assertEqual(titled.n_scored_tokens, bare.n_scored_tokens)
        self.assertGreater(min(titled.scored_positions), min(bare.scored_positions))

    def test_a_rendering_without_exactly_one_delimiter_pair_is_refused(self):
        resolved = JM.resolve(galactica_stub(), "galactica")
        with self.assertRaises(ValueError):
            JM.scored_target_positions(
                resolved.tokenizer, resolved.declaration, "[START_AMINO]"
            )

    def test_the_prollama_span_is_strictly_inside_Seq_and_the_closing_bracket(self):
        resolved = JM.resolve(prollama_stub(), "prollama")
        record = resolved.render(SEQUENCE)
        pieces = [resolved.tokenizer.convert_ids_to_tokens(i) for i in record.token_ids]
        positions = record.scored_positions
        # Contiguous, and its spellings are exactly the sequence: no delimiter
        # fragment inside, no residue left outside.
        self.assertEqual(list(positions), list(range(positions[0], positions[-1] + 1)))
        self.assertEqual("".join(pieces[i] for i in positions), SEQUENCE)
        # Both delimiters are outside it, and they are spelled by several tokens
        # each -- which is exactly why a delimiter-id rule cannot run here.
        self.assertEqual("".join(pieces[: positions[0]])[-len("Seq=<") :], "Seq=<")
        self.assertEqual("".join(pieces[positions[-1] + 1 :]), ">")
        for piece in (pieces[i] for i in positions):
            self.assertTrue(set(piece) <= set(AA20), piece)

    def test_the_prollama_span_excludes_an_instruction_prefix_rather_than_scoring_it(self):
        resolved = JM.resolve(prollama_stub(), "prollama")
        bare = resolved.render(SEQUENCE)
        prompted = resolved.render(SEQUENCE, context="Ferritin like superfamily")
        pieces = [resolved.tokenizer.convert_ids_to_tokens(i) for i in prompted.token_ids]
        self.assertEqual(prompted.n_scored_tokens, bare.n_scored_tokens)
        self.assertGreater(min(prompted.scored_positions), min(bare.scored_positions))
        self.assertEqual(
            "".join(pieces[i] for i in prompted.scored_positions), SEQUENCE
        )

    def test_the_spelled_run_rule_refuses_to_run_without_the_sequence(self):
        resolved = JM.resolve(prollama_stub(), "prollama")
        with self.assertRaises(ValueError):
            JM.scored_target_positions(
                resolved.tokenizer,
                resolved.declaration,
                resolved.declaration.render_protein(SEQUENCE),
            )

    def test_a_sequence_that_does_not_start_on_a_token_boundary_is_refused(self):
        # The failure a delimiter-id rule would have caught by construction: a
        # residue merged into the opening delimiter, so token position is not
        # residue position. It raises rather than returning a shifted span.
        resolved = JM.resolve(prollama_stub(), "prollama")
        merging = prollama_stub()
        merging._add("<M")
        merging._longest = max(merging._longest, 2)
        with self.assertRaises(ValueError) as raised:
            JM.scored_target_positions(
                merging,
                resolved.declaration,
                resolved.declaration.render_protein(SEQUENCE),
                sequence=SEQUENCE,
            )
        self.assertIn("token boundaries", str(raised.exception))

    def test_a_context_that_repeats_the_sequence_is_refused_as_ambiguous(self):
        resolved = JM.resolve(prollama_stub(), "prollama")
        with self.assertRaises(ValueError) as raised:
            JM.scored_target_positions(
                resolved.tokenizer,
                resolved.declaration,
                resolved.declaration.render_protein(SEQUENCE, context=SEQUENCE),
                sequence=SEQUENCE,
            )
        self.assertIn("ambiguous", str(raised.exception))


# ----------------------------------------------------------- the held-out draw


SCORED_RECORDS = ["".join(AA20) * 4, "".join(reversed(AA20)) * 4, ("MK" * 40)]
REFERENCE_RECORDS = [("AC" * 40), ("DE" * 40), ("FG" * 40)]


def stub_protein_draw(calls: list[dict]):
    """``arms.protein_cohort`` replaced by two fixed, overlapping windows."""

    def draw(n, min_len, max_len, *, skip=0, name="", with_ec=False, seed=None):
        calls.append({"n": n, "skip": skip, "seed": seed})
        # The reference block deliberately repeats one scored record, the way
        # Swiss-Prot repeats a sequence under several accessions.
        records = (
            list(SCORED_RECORDS) if skip == 0 else [SCORED_RECORDS[0], *REFERENCE_RECORDS]
        )
        return Cohort(
            name,
            "protein",
            records[:n] if skip == 0 else records,
            min_len,
            max_len,
            {
                "sampling": sampling_record(
                    seed=seed,
                    skip=skip,
                    requested=n,
                    eligible=64,
                    corpus="stub_swissprot",
                )
            },
        )

    return draw


def cohort_args(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        sequences=len(SCORED_RECORDS),
        unigram_sequences=4,
        protein_min_len=8,
        protein_max_len=400,
        text_min_chars=8,
        cohort_draw_seed=20260728,
    )
    vars(args).update(overrides)
    return args


class HeldOutUnigramDraw(unittest.TestCase):
    """The reference must be a different sample, and the stage must prove it."""

    def _args(self) -> argparse.Namespace:
        return cohort_args()

    def _stub_draw(self, calls: list[dict]):
        return stub_protein_draw(calls)

    def test_the_reference_is_drawn_past_the_scored_cohort_and_deduplicated(self):
        calls: list[dict] = []
        original = STAGE.protein_cohort
        STAGE.protein_cohort = self._stub_draw(calls)
        try:
            scored, reference, overlap = STAGE.mode_cohorts(self._args(), "protein")
        finally:
            STAGE.protein_cohort = original
        self.assertEqual([call["skip"] for call in calls], [0, len(SCORED_RECORDS)])
        self.assertEqual({call["seed"] for call in calls}, {20260728})
        self.assertEqual(set(scored.records) & set(reference.records), set())
        self.assertEqual(overlap["dropped_sequences_shared_with_cohort"], 1)
        self.assertEqual(reference.records, REFERENCE_RECORDS)
        # The reference is the denominator of the estimand, so how IT was drawn
        # has to reach the artefact too (Appendix B rule 1).
        self.assertEqual(reference.sampling["mode"], "seeded_permutation")
        self.assertEqual(reference.sampling["seed"], 20260728)
        self.assertEqual(reference.sampling["skip"], len(SCORED_RECORDS))

    def test_an_overlapping_reference_stops_the_run_rather_than_leaking(self):
        original_draw = STAGE.protein_cohort
        original_holdout = STAGE.held_out_cohort
        STAGE.protein_cohort = self._stub_draw([])
        # A deduplication step that silently kept the shared record: the final
        # assertion is what has to catch it.
        STAGE.held_out_cohort = lambda candidate, scored: (candidate, {})
        try:
            with self.assertRaises(ValueError):
                STAGE.mode_cohorts(self._args(), "protein")
        finally:
            STAGE.protein_cohort = original_draw
            STAGE.held_out_cohort = original_holdout

    def test_the_seeded_permutation_is_the_panels_own_and_two_windows_are_disjoint(self):
        # The draw itself is arms.selected_positions, imported rather than
        # reimplemented; this pins the property the stage relies on.
        first = selected_positions(500, n=64, skip=0, seed=20260728, label="scored")
        second = selected_positions(500, n=400, skip=64, seed=20260728, label="reference")
        self.assertEqual(set(first) & set(second), set())


# --------------------------------------------------------------- the verdicts


class Measurability(unittest.TestCase):
    """The verdict is identification, and the retired floor rides beside it inert.

    ``tests/models/test_joint_mode_identification_contract.py`` holds the static half --
    that the 0.30-nat magnitude is nowhere an operative default and that a ``_v1``
    artefact cannot be read as a ``_v2`` one. These are the value tests.
    """

    def test_a_below_floor_mode_is_unmeasurable_rather_than_failing(self):
        # galactica-1.3b's protein mode on the second draw, +0.0481 nats: the one
        # published refusal that the calibrated floor keeps, by 0.0019 nats.
        record = STAGE.verdict_record(0.0481, SCREENING_CONTEXT_INFORMATION_NATS)
        self.assertEqual(record["identification_verdict"], UNMEASURABLE)
        self.assertNotIn("FAIL", json.dumps(record))
        self.assertIn("not failing", record["identification_verdict_note"])

    def test_a_mode_above_the_floor_is_measurable(self):
        record = STAGE.verdict_record(4.6147, SCREENING_CONTEXT_INFORMATION_NATS)
        self.assertEqual(record["identification_verdict"], MEASURABLE)

    def test_a_collapsed_mode_is_still_reported_as_unmeasurable(self):
        # InstructProtein's text mode reads -12.03: worse than context-free, and
        # still a statement about this interface rather than a failed gate.
        record = STAGE.verdict_record(-12.0259, SCREENING_CONTEXT_INFORMATION_NATS)
        self.assertEqual(record["identification_verdict"], UNMEASURABLE)
        self.assertNotIn("FAIL", json.dumps(record))

    def test_the_retired_floor_is_a_column_that_admits_and_refuses_nothing(self):
        """Llama-2-7b-hf's protein mode: admitted on identification, 0.30 says no."""

        record = STAGE.verdict_record(0.084287, SCREENING_CONTEXT_INFORMATION_NATS)
        self.assertEqual(record["identification_verdict"], MEASURABLE)
        self.assertEqual(
            record["legacy_qualification_floor_nats"], MIN_CONTEXT_INFORMATION_NATS
        )
        self.assertFalse(record["clears_legacy_qualification_floor"])
        # The two readings disagree and the artefact carries both, under names a
        # reader cannot swap. Neither the verdict nor the floor is spelled the
        # way a v1 artefact spelled it.
        self.assertNotIn("verdict", record)
        self.assertNotIn("minimum_context_information_nats", record)
        self.assertIn("decides nothing", record["legacy_qualification_floor_note"])


# ------------------------------------------------------------- unigram support


class UnigramSupport(unittest.TestCase):
    def test_the_protein_reference_is_fitted_over_the_residue_alphabet_only(self):
        resolved = JM.resolve(galactica_stub(), "galactica")
        counts = STAGE.scored_target_counts(resolved, [SEQUENCE], context=None)
        self.assertEqual(counts.size, len(AA20))
        self.assertEqual(int(counts.sum()), len(SEQUENCE))
        # Support size decides the smoothing bias, which is why it is recorded.
        record = STAGE.unigram_record(
            counts,
            counts,
            support="test",
            reference=Cohort("r", "protein", [SEQUENCE], 8, 400, {}),
            overlap={},
        )
        self.assertEqual(record["support_size"], len(AA20))
        self.assertLess(record["smoothing_mass_fraction"], 0.5)
        self.assertGreater(record["cross_entropy_nats"], 0.0)

    def test_a_token_unit_reference_is_fitted_over_the_token_support_it_is_scored_on(self):
        # Rule 26's other half: the unit moved, so the support has to move with
        # it. Counting a token-unit family over twenty residue ids would leave
        # every merged target unrepresentable.
        resolved = JM.resolve(prollama_stub(), "prollama")
        counts = STAGE.scored_target_counts(resolved, [SEQUENCE], context=None)
        self.assertEqual(counts.size, len(resolved.scored_target_ids))
        self.assertGreater(counts.size, len(AA20))
        rendered = resolved.render(SEQUENCE)
        self.assertEqual(int(counts.sum()), rendered.n_scored_tokens)
        self.assertLess(int(counts.sum()), len(SEQUENCE))
        record = STAGE.unigram_record(
            counts,
            counts,
            support="test",
            reference=Cohort("r", "protein", [SEQUENCE], 8, 400, {}),
            overlap={},
        )
        self.assertEqual(record["support_size"], len(resolved.scored_target_ids))


class ScoringPrimitive(unittest.TestCase):
    """The two ways ``score_positions`` could be wrong without anything raising."""

    def _model(self, logits: "torch.Tensor"):
        class Fixed:
            def __call__(self, ids):
                return SimpleNamespace(logits=logits)

        return Fixed()

    def test_a_target_is_read_from_the_logits_of_the_position_before_it(self):
        torch.manual_seed(0)
        token_ids = [3, 1, 4, 1, 5, 9]
        logits = torch.randn(1, len(token_ids), 12)
        positions = [1, 3, 5]
        nll, mass = STAGE.score_positions(
            self._model(logits), token_ids, positions, device="cpu"
        )
        reference = torch.log_softmax(logits[0].float(), dim=-1)
        expected = [
            float(-reference[position - 1, token_ids[position]]) for position in positions
        ]
        self.assertEqual([round(value, 6) for value in nll], [round(v, 6) for v in expected])
        self.assertIsNone(mass)
        # The off-by-one this pins down is silent: it produces finite, plausible
        # cross-entropies for the wrong pairing.
        off_by_one = [
            float(-reference[position, token_ids[position]]) for position in positions
        ]
        self.assertNotEqual(
            [round(value, 6) for value in nll], [round(v, 6) for v in off_by_one]
        )

    def test_the_subspace_mass_is_the_probability_on_the_declared_residue_ids(self):
        torch.manual_seed(1)
        token_ids = [2, 7, 7, 3]
        logits = torch.randn(1, len(token_ids), 12)
        subspace = [0, 5, 11]
        _, mass = STAGE.score_positions(
            self._model(logits), token_ids, [1, 2, 3], device="cpu", subspace=subspace
        )
        probabilities = torch.softmax(logits[0].float(), dim=-1)
        expected = [float(probabilities[position - 1, subspace].sum()) for position in (1, 2, 3)]
        self.assertEqual([round(v, 6) for v in mass], [round(v, 6) for v in expected])

    def test_a_record_with_no_scored_position_is_refused(self):
        with self.assertRaises(ValueError):
            STAGE.score_positions(self._model(torch.zeros(1, 3, 5)), [1, 2, 3], [], device="cpu")


# -------------------------------------------------- the protein record, end to end


class ZeroLogitModel:
    """A decoder with no opinion, so what is under test is the record, not the number."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, ids):
        return SimpleNamespace(logits=torch.zeros(1, ids.shape[1], self.vocab_size))


class TheProteinRecordAndItsControls(unittest.TestCase):
    """What reaches the artefact, for both symbol units, through the real path."""

    def _protein_mode(self, name: str, tokenizer) -> dict:
        return self._protein_mode_with_records(name, tokenizer)[0]

    def _protein_mode_with_records(self, name: str, tokenizer):
        tokenisation = JM.resolve(tokenizer, name)
        original = STAGE.protein_cohort
        STAGE.protein_cohort = stub_protein_draw([])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return STAGE.protein_mode(
                    cohort_args(
                        device="cpu",
                        protein_context=None,
                        max_tokens=4096,
                        identification_floor_nats=SCREENING_CONTEXT_INFORMATION_NATS,
                    ),
                    ZeroLogitModel(len(tokenizer)),
                    tokenisation,
                )
        finally:
            STAGE.protein_cohort = original

    def test_a_token_unit_record_carries_its_measured_residues_per_token_and_its_unit(self):
        record = self._protein_mode("prollama", prollama_stub())
        self.assertEqual(record["symbol_unit"], JM.TOKEN_UNIT)
        self.assertEqual(record["context_information_unit"], "nats per scored token")
        measured = record["measured_residues_per_scored_token"]
        self.assertGreater(measured, 1.0)
        self.assertEqual(measured, record["declared_rendering"]["residues_per_scored_token"])
        # The estimand really was formed in the unit the record names, and over a
        # reference fitted on the same token support.
        self.assertAlmostEqual(
            record["context_information_nats"],
            record["unigram_reference"]["cross_entropy_nats"]
            - record["declared_rendering"]["clean_nll_nats_per_scored_token"],
            places=12,
        )
        self.assertGreater(record["unigram_reference"]["support_size"], len(AA20))

    def test_a_token_unit_magnitude_is_marked_non_comparable_in_its_own_field(self):
        record = self._protein_mode("prollama", prollama_stub())
        comparability = record["cross_arm_comparability"]
        self.assertEqual(comparability["verdict"], "NOT_COMPARABLE_ACROSS_ARMS")
        self.assertEqual(comparability["symbol_unit"], JM.TOKEN_UNIT)
        self.assertEqual(
            comparability["measured_residues_per_scored_token"],
            record["measured_residues_per_scored_token"],
        )
        self.assertIn("L23", comparability["note"])

    def test_a_per_residue_record_stays_in_residues_and_stays_comparable(self):
        record = self._protein_mode("galactica", galactica_stub())
        self.assertEqual(record["symbol_unit"], JM.RESIDUE_UNIT)
        self.assertEqual(record["context_information_unit"], "nats per scored residue")
        self.assertEqual(record["measured_residues_per_scored_token"], 1.0)
        self.assertEqual(
            record["cross_arm_comparability"]["verdict"], "COMPARABLE_ACROSS_ARMS"
        )
        self.assertEqual(record["unigram_reference"]["support_size"], len(AA20))
        self.assertAlmostEqual(
            record["context_information_nats"],
            record["unigram_reference"]["cross_entropy_nats"]
            - record["declared_rendering"]["clean_nll_nats_per_residue"],
            places=12,
        )

    def test_the_naive_control_is_withheld_with_a_reason_rather_than_fabricated(self):
        record = self._protein_mode("prollama", prollama_stub())
        naive = record["controls"]["naive_rendering"]
        self.assertEqual(naive["verdict"], "WITHHELD")
        self.assertIn("no per-residue escape", naive["reason"])
        # No number is emitted under the control's name, in either unit.
        for key in (
            "price_nats_per_residue",
            "price_nats_per_scored_token",
            "clean_nll_nats_per_scored_token",
            "clean_nll_nats_per_residue",
            "residues_per_scored_token",
        ):
            self.assertNotIn(key, naive)

    def test_the_naive_control_is_still_priced_where_an_escape_exists(self):
        record = self._protein_mode("galactica", galactica_stub())
        naive = record["controls"]["naive_rendering"]
        self.assertNotIn("verdict", naive)
        self.assertIn("price_nats_per_residue", naive)
        self.assertGreater(naive["residues_per_scored_token"], 1.0)
        self.assertFalse(naive["verified_against_declared_symbol_unit"])

    def test_the_reversal_control_survives_the_unit_change_and_is_read_per_residue(self):
        # Requirement 4: a within-arm difference over an identical residue
        # multiset. The token counts differ between the two conditions, which is
        # exactly why the cost is not read per token.
        record = self._protein_mode("prollama", prollama_stub())
        declared = record["declared_rendering"]
        reversed_score = record["controls"]["reversed"]
        self.assertEqual(reversed_score["cost_unit"], "nats per residue")
        self.assertIn("cost_nats_per_residue", reversed_score)
        self.assertNotIn("cost_nats_per_scored_token", reversed_score)
        self.assertNotIn("context_information_nats_per_residue", reversed_score)
        self.assertIn("different token population", reversed_score["note"])
        self.assertEqual(
            reversed_score["n_scored_residues"], declared["n_scored_residues"]
        )
        self.assertAlmostEqual(
            reversed_score["cost_nats_per_residue"],
            reversed_score["clean_nll_nats_per_residue"]
            - declared["clean_nll_nats_per_residue"],
            places=12,
        )

    def test_the_reversal_control_keeps_its_shared_reference_for_a_per_residue_family(self):
        record = self._protein_mode("galactica", galactica_stub())
        reversed_score = record["controls"]["reversed"]
        self.assertEqual(reversed_score["cost_unit"], "nats per residue")
        self.assertIn("preserves the residue multiset", reversed_score["note"])
        self.assertEqual(
            reversed_score["n_scored_tokens"],
            record["declared_rendering"]["n_scored_tokens"],
        )

    def test_the_persisted_symbol_label_names_what_the_array_actually_counts(self):
        """Both symbol units persist residues, so both must say residue.

        The defect this closes: a token-unit family carried
        ``n_symbols_is = "a scored token"`` beside an ``n_symbols`` array holding
        residues -- 20866 residues against 13609 tokens on ``Llama-2-7b-hf`` --
        so bits per symbol read off that sidecar was bits per residue under a
        label that said token.
        """

        for name, tokenizer, unit in (
            ("prollama", prollama_stub(), JM.TOKEN_UNIT),
            ("galactica", galactica_stub(), JM.RESIDUE_UNIT),
        ):
            record, statistics = self._protein_mode_with_records(name, tokenizer)
            self.assertEqual(record["symbol_unit"], unit)
            self.assertEqual(statistics.symbol_definition, STAGE.RESIDUE_SYMBOL)
            for condition, held in statistics.conditions.items():
                self.assertEqual(
                    int(held.n_symbols.sum()),
                    record["declared_rendering"]["n_scored_residues"]
                    if condition == "protein_declared"
                    else record["controls"]["reversed"]["n_scored_residues"],
                    condition,
                )
            declared = statistics.conditions["protein_declared"]
            # On the token-unit family the two arrays genuinely differ, which is
            # what makes the label a claim rather than a restatement.
            if unit == JM.TOKEN_UNIT:
                self.assertNotEqual(
                    int(declared.n_symbols.sum()), int(declared.token_count.sum())
                )
            else:
                self.assertEqual(
                    int(declared.n_symbols.sum()), int(declared.token_count.sum())
                )

    def test_a_token_label_over_a_residue_array_is_refused_rather_than_persisted(self):
        """The label and the array cannot drift apart without the stage stopping."""

        _, statistics = self._protein_mode_with_records("prollama", prollama_stub())
        with self.assertRaises(ValueError) as raised:
            replace(statistics, symbol_definition=STAGE.TOKEN_SYMBOL)
        self.assertIn("n_symbols is declared to count", str(raised.exception))

        # The text mode's array IS the token count, so the same label is
        # accepted there; the guard is about agreement, not about the string.
        text = replace(
            statistics,
            mode="text",
            symbol_definition=STAGE.TOKEN_SYMBOL,
            conditions={
                name: replace(held, n_symbols=held.token_count)
                for name, held in statistics.conditions.items()
            },
        )
        self.assertEqual(text.symbol_definition, STAGE.TOKEN_SYMBOL)

        with self.assertRaises(ValueError) as raised:
            replace(statistics, symbol_definition="a scored thing")
        self.assertIn("not one of the two declared symbol definitions", str(raised.exception))

    def test_the_rendering_facts_name_the_unit_and_the_support_they_were_read_with(self):
        facts = JM.resolve(prollama_stub(), "prollama").facts()
        self.assertEqual(facts["symbol_unit"], JM.TOKEN_UNIT)
        self.assertFalse(facts["naive_control_available"])
        self.assertFalse(facts["delimiters_are_tokens"])
        self.assertIsNone(facts["start_token_id"])
        self.assertIsNone(facts["prefix_marker"])
        self.assertEqual(facts["prefix_marker_ids"], [])
        self.assertEqual(
            facts["n_scored_target_token_ids"], len(facts["scored_target_token_ids"])
        )
        self.assertGreater(facts["n_scored_target_token_ids"], len(AA20))
        json.dumps(facts)
        galactica = JM.resolve(galactica_stub(), "galactica").facts()
        self.assertEqual(galactica["symbol_unit"], JM.RESIDUE_UNIT)
        self.assertTrue(galactica["delimiters_are_tokens"])
        self.assertEqual(galactica["n_scored_target_token_ids"], len(AA20))
        self.assertIsNone(galactica["prefix_marker"])
        self.assertEqual(galactica["prefix_marker_ids"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
