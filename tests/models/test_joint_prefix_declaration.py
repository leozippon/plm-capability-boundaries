"""InstructProtein's declared prefix, and the ids of every checkpoint sharing it.

InstructProtein's protein rendering is ``</s>`` followed by the declared block,
and the ``</s>`` entered through ``joint_modes.encode``'s default
``add_special_tokens`` -- a fact about one tokenizer configuration that no
declaration named, in a family whose other rendering facts are all declared.
``JointRendering.prefix_marker`` now names it beside the rendering,
``joint_modes.prefix_marker_ids`` resolves it to ids,
``JointTokenisation.render`` refuses an encoding that does not open with it, and
the rendering facts every artefact carries report it.

The one ``encode`` call also governs ProLLaMA, ProLLaMA Stage 1 and Llama-2-7b,
whose tokenizers prepend ``<s>`` (id 1) and whose prefix no measurement prices.
Neither of those families declares a marker, so the resolver refuses the question
rather than answering "this rendering prefixes nothing", and the last test here
pins all four staged tokenizers' id lists for the declared block and for a
document. Those lists are what the same code produced before the declaration
existed -- the declaration added no parameter to the rendering, changed no
call and moved no default -- so they are the regression guard on the shared line
rather than a record of it.

No weights and no GPU: the pinned test reads tokenizer files, and skips a
checkpoint directory that is not staged on this host.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.capability.models import joint_modes as JM  # noqa: E402
from src.capability.core.arms import MODEL_ROOT  # noqa: E402
from tests.core.test_joint_mode_qualification import (  # noqa: E402
    instructprotein_stub,
    prollama_stub,
)

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
DOCUMENT = "The protein binds ATP and hydrolyses it."

#: The declared block of ``SEQUENCE`` on the staged InstructProtein tokenizer:
#: ``</s>`` (2), ``<protein>`` (50265), the 33 ``Ƥ`` residue ids and ``</protein>``
#: (50266).
INSTRUCTPROTEIN_BLOCK_IDS = [
    2,
    50265,
    50277,
    50275,
    50283,
    50267,
    50286,
    50274,
    50267,
    50275,
    50280,
    50281,
    50280,
    50274,
    50282,
    50271,
    50284,
    50275,
    50282,
    50273,
    50271,
    50282,
    50281,
    50280,
    50276,
    50270,
    50270,
    50281,
    50276,
    50272,
    50276,
    50274,
    50270,
    50284,
    50280,
    50266,
]

INSTRUCTPROTEIN_DOCUMENT_IDS = [2, 133, 8276, 40361, 22020, 8, 13575, 352, 24151, 24, 4]

#: The three staged LLaMA-lineage tokenizers carry the same 32000-piece
#: SentencePiece vocabulary, so one pinned list is what all three must produce and
#: a failure names which directory drifted.
LLAMA_LINEAGE_BLOCK_IDS = [
    1,
    25981,
    29922,
    29966,
    29924,
    29968,
    6040,
    29979,
    10764,
    29968,
    29984,
    29934,
    29984,
    3235,
    29943,
    29963,
    29968,
    7068,
    9998,
    29934,
    29984,
    1307,
    1001,
    29931,
    29954,
    5265,
    22240,
    29984,
    29958,
]

LLAMA_LINEAGE_DOCUMENT_IDS = [
    1,
    450,
    26823,
    7868,
    29879,
    27884,
    322,
    17546,
    368,
    29879,
    267,
    372,
    29889,
]

#: Checkpoint directory -> (rendering, declared-block ids, document ids).
CHECKPOINTS = {
    "InstructProtein": (
        "instructprotein",
        INSTRUCTPROTEIN_BLOCK_IDS,
        INSTRUCTPROTEIN_DOCUMENT_IDS,
    ),
    "ProLLaMA": ("prollama", LLAMA_LINEAGE_BLOCK_IDS, LLAMA_LINEAGE_DOCUMENT_IDS),
    "ProLLaMA_Stage_1": ("prollama", LLAMA_LINEAGE_BLOCK_IDS, LLAMA_LINEAGE_DOCUMENT_IDS),
    "Llama-2-7b-hf": ("prollama", LLAMA_LINEAGE_BLOCK_IDS, LLAMA_LINEAGE_DOCUMENT_IDS),
}


class TheDeclaredPrefixIsResolvedAndNotScored(unittest.TestCase):
    def test_the_declaration_names_the_token_the_ids_open_with(self):
        declaration = JM.rendering("instructprotein")
        self.assertEqual(declaration.prefix_marker, "</s>")
        tokenizer = instructprotein_stub()
        self.assertEqual(JM.prefix_marker_ids(tokenizer, declaration), (2,))

    def test_the_declared_span_is_the_residues_and_the_prefix_is_not_a_target(self):
        resolved = JM.resolve(instructprotein_stub(), "instructprotein")
        self.assertEqual(resolved.prefix_marker_ids, (2,))
        record = resolved.render(SEQUENCE)
        self.assertEqual(record.token_ids[0], 2)
        # Position 1 is <protein>, so the scored run starts at 2 and has one
        # position per residue: the prefix occupies position 0 and no rule scores
        # it, before or after this declaration.
        self.assertEqual(list(record.scored_positions), list(range(2, 2 + len(SEQUENCE))))
        self.assertNotIn(2, {record.token_ids[p] for p in record.scored_positions})
        facts = resolved.facts()
        self.assertEqual(facts["prefix_marker"], "</s>")
        self.assertEqual(facts["prefix_marker_ids"], [2])

    def test_a_family_whose_prefix_is_unpriced_is_refused_rather_than_answered_empty(self):
        declaration = JM.rendering("prollama")
        self.assertIsNone(declaration.prefix_marker)
        with self.assertRaises(ValueError) as raised:
            JM.prefix_marker_ids(prollama_stub(), declaration)
        self.assertIn("declares no prefix marker", str(raised.exception))
        # The refusal is not a claim that nothing is there: the same call does put
        # a token in front, which is why an empty answer would be wrong.
        self.assertEqual(JM.encode(prollama_stub(), "Seq=<MK>")[0], 2)
        resolved = JM.resolve(prollama_stub(), declaration)
        self.assertEqual(resolved.prefix_marker_ids, ())
        self.assertEqual(resolved.facts()["prefix_marker"], None)
        self.assertEqual(resolved.facts()["prefix_marker_ids"], [])

    def test_an_encoding_that_stops_opening_with_the_declared_prefix_is_refused(self):
        tokenizer = instructprotein_stub()
        tokenizer.bos_id = None
        with self.assertRaises(ValueError) as raised:
            JM.resolve(tokenizer, "instructprotein")
        self.assertIn("declared to open with", str(raised.exception))


class TheStagedTokenizersEncodeTheIdsTheDeclarationWasReadAt(unittest.TestCase):
    def _tokenizer(self, directory: str):
        path = MODEL_ROOT / directory
        if not (path / "tokenizer_config.json").is_file():
            self.skipTest(f"{directory} is not staged on this host")
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(path))

    def test_every_governed_checkpoint_encodes_the_pinned_ids(self):
        for directory, (rendering, block_ids, document_ids) in CHECKPOINTS.items():
            with self.subTest(directory):
                tokenizer = self._tokenizer(directory)
                declaration = JM.rendering(rendering)
                self.assertEqual(
                    JM.encode(tokenizer, declaration.render_protein(SEQUENCE)),
                    block_ids,
                )
                self.assertEqual(JM.encode(tokenizer, DOCUMENT), document_ids)
                # The prefix is present whatever the text is, including an empty one.
                self.assertEqual(JM.encode(tokenizer, ""), [block_ids[0]])
                if declaration.prefix_marker is None:
                    with self.assertRaises(ValueError):
                        JM.prefix_marker_ids(tokenizer, declaration)
                else:
                    self.assertEqual(
                        JM.prefix_marker_ids(tokenizer, declaration), (block_ids[0],)
                    )
                    resolved = JM.resolve(tokenizer, declaration)
                    self.assertNotIn(
                        block_ids[0],
                        {
                            resolved.render(SEQUENCE).token_ids[position]
                            for position in resolved.render(SEQUENCE).scored_positions
                        },
                    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
