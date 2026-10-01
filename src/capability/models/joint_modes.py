"""How one joint language-protein checkpoint is rendered in each of its modes.

A joint decoder has two input formats and one set of weights, so Appendix B rule
4 -- feed every arm the format it was trained on, and verify the rendering
against the model's own likelihood -- applies twice to the same checkpoint. This
module is the single place either format is decided (Appendix B rule 12);
``scripts/capability/stages/joint_mode_qualification.py`` names a family on the command
line and reads everything else from here.

**Why the per-residue escape is load-bearing rather than cosmetic.** Galactica's
protein format is ``[START_AMINO]...[END_AMINO]``, and its released
``tokenizer.json`` carries a ``Split`` pretokenizer on the literal
``SPL1T-TH1S-Pl3A5E`` with ``behavior: "Removed"`` -- so the marker is consumed by
the tokenizer itself, and inserting it before every residue is the *supported*
way to reach the model's own one-token-per-residue protein alphabet rather than a
workaround. Without it ``AutoTokenizer`` silently merges residues into
multi-residue BPE pieces at about 1.79 residues per token, and the measured cost
of that merge is 2.886 nats/token on ``galactica-1.3b`` (EXP-R2-151). Nothing
about a merged rendering looks wrong in an artefact -- the run completes, the
numbers are finite, and they are a measurement of a different object. That is why
:meth:`JointTokenisation.render` **raises** on a rendering that declared the
per-residue alphabet and did not reach it, instead of recording a warning field
nobody reads.

**The symbol unit is declared, because not every family has one.** Galactica and
InstructProtein both reach a per-residue alphabet, so a scored position *is* a
residue and the estimand is in nats per residue. ProLLaMA does not: its trained
protein format is ``Seq=<...>`` over the unmodified 32000-piece LLaMA-2
SentencePiece vocabulary, which merges ``TA``, ``IA``, ``IS``, ``SH``, ``FS`` and
several hundred other residue runs into single pieces at about 1.54 residues per
token on a Swiss-Prot draw. That is not a rendering defect that an escape could
repair -- there is no escape, and the merged rendering *is* what the checkpoint
was trained on. The difference between the two cases cannot live at a call site,
because a caller cannot tell them apart; it lives in
:attr:`JointRendering.symbol_unit`, and it decides three things at once: which
verification a declared rendering must pass, what one scored symbol is, and
therefore what the estimand's unit is.

The guard is unchanged for the families that declare the residue as their unit.
:meth:`JointTokenisation.verify_one_token_per_residue` still raises, still takes
no flag, and is still what a residue-unit rendering is checked against.
:meth:`JointTokenisation.verify_symbol_unit` is the dispatcher that chooses it;
a token-unit family is checked against the property that holds for *it* -- every
scored target inside the enumerated residue-spelling support -- rather than
against a property it was never going to have.

**One derivation, three rules.** The families locate their scored positions
differently, and the difference follows from the format rather than from taste.
Galactica's residues are ordinary single-letter pieces of a text vocabulary and
are identified by *position* (strictly between the delimiter ids); InstructProtein
declares twenty dedicated ``ƤA``..``ƤY`` tokens and its residues are identified by
*identity*; ProLLaMA's delimiters ``Seq=<`` and ``>`` are not tokens of the
vocabulary at all -- ``Seq=<`` is three pieces -- so neither positional rule can
run, and its scored span is the one contiguous token run whose *spellings*
concatenate to exactly the sequence. That third rule fails loudly when a residue
merges into a delimiter or into the document context, which is the failure a
delimiter-id rule would have caught by construction.

All three rules are read off the same declaration, and the residue token ids the
identity rule needs are derived by rendering a one-residue sequence through the
very same declaration and the very same span locator -- so a family cannot end up
with one alphabet for selection and another for verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.capability.core.arms import AA20

#: The literal Galactica's released pretokenizer splits on and removes. Declared
#: here because it is part of the rendering, not a property of any one model
#: directory; the ``Split`` rule that consumes it lives in the checkpoint's own
#: ``tokenizer.json``, which is what makes this the supported path.
GALACTICA_SPLIT_MARKER = "SPL1T-TH1S-Pl3A5E"

#: What one scored symbol of a family's protein mode IS. ``RESIDUE_UNIT`` means
#: the rendering reaches the checkpoint's per-residue alphabet, so a scored
#: position carries exactly one residue and the estimand is in nats per residue.
#: ``TOKEN_UNIT`` means it does not and cannot: the tokenizer decides how many
#: residues a scored target carries, so the estimand is in nats per token and a
#: magnitude from such a family is not commensurable with a residue-unit family's
#: (Appendix B rule 26, limitation L23).
RESIDUE_UNIT = "residue"
TOKEN_UNIT = "token"
SYMBOL_UNITS = (RESIDUE_UNIT, TOKEN_UNIT)

SYMBOL_UNIT_DEFINITIONS = {
    RESIDUE_UNIT: (
        "one scored symbol is one residue; the declared rendering is verified at "
        "exactly one token per residue and refused otherwise"
    ),
    TOKEN_UNIT: (
        "one scored symbol is one token carrying one or more residues, because "
        "the family's trained protein format has no per-residue alphabet to "
        "reach. The measured residues per scored token is a property of the "
        "corpus and the tokenizer together and must be reported beside every "
        "magnitude this family produces"
    ),
}

#: How a rendering locates the positions whose target token carries residues.
BETWEEN_DELIMITERS = "between_delimiters"
DECLARED_RESIDUE_IDS = "declared_residue_ids"
SPELLED_SEQUENCE_RUN = "spelled_sequence_run"
SCORED_TARGET_RULES = (BETWEEN_DELIMITERS, DECLARED_RESIDUE_IDS, SPELLED_SEQUENCE_RUN)

#: ``declared`` is the format the checkpoint was trained on. ``naive`` is the
#: same block with the per-residue escape removed -- the rendering an unaided
#: ``AutoTokenizer`` call produces, kept as a *control* so that a stage can price
#: the rendering in the estimand's own units rather than assert its value. It
#: exists only for a family that has an escape to remove; see
#: :attr:`JointRendering.naive_control_available`.
DECLARED = "declared"
NAIVE = "naive"
RENDERING_VARIANTS = (DECLARED, NAIVE)


@dataclass(frozen=True)
class JointRendering:
    """One family's declaration of how a protein is written for a joint decoder.

    ``symbol_unit`` is what one scored symbol of this family's protein mode is,
    and it is the field the rest of the module dispatches on. It is declared
    rather than inferred because the two states it separates are indistinguishable
    from a measurement: a residue-unit family whose tokenizer merged residues and
    a token-unit family reading its own trained format produce the same shape of
    record, and only the first is a defect.

    ``residue_escape`` is the string inserted into the *rendered text* before
    every residue. It is not necessarily part of the resulting token's spelling:
    Galactica's marker is removed by the pretokenizer, so the token is the bare
    letter, while InstructProtein's ``Ƥ`` is the first character of a dedicated
    added token. Nothing here needs to know which, because the residue token ids
    are derived from the tokenizer rather than spelled a second time. It is
    ``None`` exactly for a token-unit family: an escape that reached the model's
    per-residue alphabet would make the family a residue-unit one, and one that
    did not would not be an escape. That is also what decides whether the naive
    rendering control is definable at all -- see :attr:`naive_control_available`.

    ``protein_context_template`` is the optional document context the block is
    embedded in, with a single ``{context}`` field. It is optional in the strong
    sense: a rendering with no context is the bare block, and the context string
    that was used reaches the artefact rather than being implied by a flag.

    ``prefix_marker`` is the token this family's tokenizer puts in front of the
    *encoded* rendering, and it is not part of the string ``render_protein``
    returns. It is declared here because :func:`encode` calls the tokenizer with
    ``add_special_tokens`` at its default, so the ids a rendering produces are its
    text plus whatever the post-processor adds, and that addition is otherwise
    invisible from every artefact: InstructProtein's protein rendering is
    ``</s>`` followed by the declared block, which is also its end-of-sequence id
    and is what its ``tokenizer_config.json``'s ``add_bos_token`` puts there.
    ``None`` means this family declares no prefix marker, which is not the
    statement that its rendering prefixes nothing: ProLLaMA's and LLaMA-2's
    tokenizers prepend ``<s>`` under the same call and neither family's prefix has
    been priced, so neither declares one. :func:`prefix_marker_ids` is the door a
    measurement resolves it through, and the counterpart of
    :func:`src.capability.core.arms.rendering_marker_ids` for the panel's own prefixes.
    """

    name: str
    symbol_unit: str
    protein_start: str
    protein_end: str
    residue_escape: str | None
    escape_before_end_delimiter: bool
    protein_context_template: str | None
    prefix_marker: str | None
    scored_target_rule: str
    residue_subspace_disjoint_from_text: bool
    note: str

    def __post_init__(self) -> None:
        if self.symbol_unit not in SYMBOL_UNITS:
            raise ValueError(
                f"{self.name}: symbol unit {self.symbol_unit!r} is not declared; known "
                f"units are {SYMBOL_UNITS}. The unit decides which verification the "
                "declared rendering must pass and what the estimand counts, so it "
                "cannot be left to a default"
            )
        if self.scored_target_rule not in SCORED_TARGET_RULES:
            raise ValueError(
                f"{self.name}: scored target rule {self.scored_target_rule!r} is not "
                f"declared; known rules are {SCORED_TARGET_RULES}"
            )
        if self.symbol_unit == RESIDUE_UNIT and not self.residue_escape:
            raise ValueError(
                f"{self.name}: a joint rendering that declares the residue as its "
                "symbol unit must declare the per-residue escape that produces one "
                "token per residue; an empty one IS the naive control and cannot also "
                "be the declared format"
            )
        if self.symbol_unit == TOKEN_UNIT and self.residue_escape is not None:
            raise ValueError(
                f"{self.name}: declares the token as its symbol unit and a per-residue "
                "escape as well. An escape that reached the model's per-residue "
                "alphabet would make this a residue-unit family and one that did not "
                "would not be an escape, so the two cannot both hold"
            )
        if self.escape_before_end_delimiter and not self.residue_escape:
            raise ValueError(
                f"{self.name}: declares the escape is repeated before the closing "
                "delimiter but declares no escape to repeat"
            )
        if not self.protein_start or not self.protein_end:
            raise ValueError(f"{self.name}: both protein delimiters must be declared")
        if self.protein_start == self.protein_end:
            raise ValueError(
                f"{self.name}: the two delimiters are the same string, so the scored "
                "span cannot be located"
            )
        if self.protein_context_template is not None and (
            "{context}" not in self.protein_context_template
        ):
            raise ValueError(
                f"{self.name}: the context template carries no {{context}} field, so "
                "the context string a run used could not reach the artefact"
            )

    @property
    def naive_control_available(self) -> bool:
        """Whether "the same block with the per-residue escape removed" exists.

        A family with no escape has nothing to remove, so the naive rendering is
        the declared rendering, its price is zero by construction, and reporting
        it would be a control that measured nothing. The stage withholds it with
        that reason instead.
        """

        return self.residue_escape is not None

    @property
    def positional_scored_target_rule(self) -> str:
        """The rule that locates the span *without* knowing the residue ids yet.

        Two callers need it: resolution, which derives the residue ids from a
        one-residue probe and therefore cannot use the identity rule, and the
        naive control, which has no residue tokens by construction so the
        identity rule would select nothing in it.
        """

        if self.scored_target_rule == DECLARED_RESIDUE_IDS:
            return BETWEEN_DELIMITERS
        return self.scored_target_rule

    def render_protein(
        self, sequence: str, *, context: str | None = None, variant: str = DECLARED
    ) -> str:
        """The string this family feeds a joint decoder for one sequence."""

        if variant not in RENDERING_VARIANTS:
            raise ValueError(
                f"unknown rendering variant {variant!r}; declared: {RENDERING_VARIANTS}"
            )
        if variant == NAIVE and not self.naive_control_available:
            raise ValueError(
                f"{self.name} declares no per-residue escape, so the naive rendering "
                "IS the declared rendering and pricing it would report a control that "
                "measured nothing. The stage withholds that control with its reason "
                "rather than emitting a zero"
            )
        if not sequence:
            raise ValueError("cannot render an empty sequence")
        illegal = sorted(set(sequence) - set(AA20))
        if illegal:
            raise ValueError(
                f"{self.name}: sequence carries non-canonical symbols {illegal}; the "
                "residue alphabet this rendering is declared over is src.capability.core.arms.AA20"
            )
        escape = (self.residue_escape or "") if variant == DECLARED else ""
        body = "".join(escape + residue for residue in sequence)
        if escape and self.escape_before_end_delimiter:
            body += escape
        block = f"{self.protein_start}{body}{self.protein_end}"
        if context is None:
            return block
        if self.protein_context_template is None:
            raise ValueError(
                f"{self.name} declares no document-context template, so it cannot be "
                f"rendered with the context {context!r}"
            )
        return self.protein_context_template.format(context=context) + block


JOINT_RENDERINGS: dict[str, JointRendering] = {
    "galactica": JointRendering(
        name="galactica",
        symbol_unit=RESIDUE_UNIT,
        protein_start="[START_AMINO]",
        protein_end="[END_AMINO]",
        residue_escape=GALACTICA_SPLIT_MARKER,
        # galai's own form, and the one verified at EXP-R2-151: the marker is
        # repeated once before the closing delimiter.
        escape_before_end_delimiter=True,
        protein_context_template="# {context}\n\n",
        # The released tokenizer declares no beginning-of-sequence token and its
        # post-processor adds nothing, so this rendering's encoded ids are its own
        # text and there is no prefix to declare.
        prefix_marker=None,
        # The residues are ordinary single-letter pieces of a 50000-piece text
        # vocabulary, so identity cannot separate a residue from the same letter
        # occurring in prose. Position can, and the delimiters are added tokens.
        scored_target_rule=BETWEEN_DELIMITERS,
        residue_subspace_disjoint_from_text=False,
        note=(
            "protein is [START_AMINO]...[END_AMINO] with the SPL1T-TH1S-Pl3A5E "
            "marker before every residue and once before the closing delimiter. "
            "The released tokenizer.json carries a Split pretokenizer on that "
            "literal with behavior 'Removed', so the marker is consumed by the "
            "tokenizer and this is the supported path rather than a workaround. "
            "The residue tokens are ordinary letters of the text vocabulary, so "
            "this family declares NO disjoint residue subspace and a "
            "residue-probability-mass statistic on its text mode would be a "
            "statistic about twenty capital letters"
        ),
    ),
    "instructprotein": JointRendering(
        name="instructprotein",
        symbol_unit=RESIDUE_UNIT,
        protein_start="<protein>",
        protein_end="</protein>",
        residue_escape="Ƥ",
        escape_before_end_delimiter=False,
        protein_context_template="Instruction: {context}\nOutput: ",
        # The prefix this checkpoint's own ``tokenizer_config.json`` puts there: it
        # declares ``add_bos_token`` true with ``bos_token`` "</s>", so ``encode``
        # opens the ids with id 2 whatever the text is. Measured on the staged
        # checkpoint at stage 01's cohort draw seed 20260728 in float32, over 8
        # Swiss-Prot records of 64-246 residues: the protein-mode span reads
        # 0.878428 nats/residue with the prefix against 3.088913 without, so
        # removing it costs +2.210484 nats/residue, and residue 1's mean NLL is
        # 0.035087 nats (stderr 5.7742e-08) against 16.340752 (stderr 8.0095e-07).
        # At position 0 the token that follows is <protein> with probability
        # 0.981308, mean over the same 8 records. The prefix is therefore part of
        # this rendering and is declared here rather than left to the call's
        # default.
        prefix_marker="</s>",
        # added_tokens.json declares <protein> 50265, </protein> 50266 and
        # ƤA..ƤY 50267-50286, so the residue output space is enumerable and
        # disjoint from the text vocabulary and a scored position is decidable by
        # inspection rather than by inference from position.
        scored_target_rule=DECLARED_RESIDUE_IDS,
        residue_subspace_disjoint_from_text=True,
        note=(
            "protein is <protein>...</protein> with every residue written as its "
            "own dedicated added token 'Ƥ'+A..Y. Those twenty ids are disjoint "
            "from the text vocabulary, which makes the residue probability mass "
            "at a text position a measurable statistic and is what distinguishes "
            "a collapsed text mode from a merely poor one"
        ),
    ),
    "prollama": JointRendering(
        name="prollama",
        # Measured on the staged ProLLaMA_Stage_1 tokenizer, not assumed: the
        # unmodified 32000-piece LLaMA-2 SentencePiece vocabulary spells a
        # 22-residue probe as 17 pieces (TA, IA, IS, SH and FS are single
        # tokens), and 1.536 residues per token over a 64-record Swiss-Prot draw
        # in band 64-246 at seed 20260728. There is no escape that reaches a
        # per-residue alphabet -- the merged rendering IS the trained format.
        symbol_unit=TOKEN_UNIT,
        protein_start="Seq=<",
        protein_end=">",
        residue_escape=None,
        escape_before_end_delimiter=False,
        # Stage 2's instruction form. Stage 1 saw the bare block, so a run that
        # names no context measures the format stage 1 was trained on and a run
        # that names one measures stage 2's; whichever was used reaches the
        # artefact.
        protein_context_template="[Generate by superfamily] Superfamily=<{context}> ",
        # No prefix marker is declared. The tokenizer does prepend one -- ``<s>``,
        # id 1, under the same default ``encode`` call -- but no measurement prices
        # that prefix on this lineage's text mode, and declaring it here would
        # report an unpriced rendering as an established one.
        prefix_marker=None,
        # 'Seq=<' is three pieces of this vocabulary and '>' merges with the
        # preceding word when it stands alone, so neither delimiter is a token
        # whose id could bound the span. The span is the token run that spells
        # the sequence, which also refuses a rendering where a residue merged
        # into a delimiter or into the instruction prefix.
        scored_target_rule=SPELLED_SEQUENCE_RUN,
        residue_subspace_disjoint_from_text=False,
        note=(
            "protein is Seq=<...> written straight into the unmodified LLaMA-2 "
            "SentencePiece vocabulary, which merges residue runs into single "
            "pieces. That is the format all three checkpoints of this lineage "
            "were trained on, so the merge is the declared behaviour and not a "
            "rendering defect -- this family therefore declares the TOKEN as its "
            "symbol unit, and its context information is in nats per token. A "
            "magnitude from this family is NOT comparable with a per-residue "
            "family's (Appendix B rule 26, limitation L23); read it beside the "
            "measured residues per scored token the artefact records with it. "
            "The residues are ordinary pieces of the text vocabulary, so this "
            "family declares NO disjoint residue subspace, and it has no "
            "per-residue escape, so the naive rendering control is withheld"
        ),
    ),
}

RENDERING_NAMES: tuple[str, ...] = tuple(JOINT_RENDERINGS)


def rendering(name: str) -> JointRendering:
    """The declaration for one family, refusing a name nobody declared."""

    if name not in JOINT_RENDERINGS:
        raise KeyError(
            f"unknown joint rendering {name!r}; declared families are "
            f"{sorted(JOINT_RENDERINGS)}. A joint checkpoint whose format is not "
            "declared here cannot be measured: rule 4 requires the rendering to be "
            "established before any number is read from it"
        )
    return JOINT_RENDERINGS[name]


# --------------------------------------------------------------- tokenisation


def encode(tokenizer: Any, text: str) -> list[int]:
    """Token ids for one string, as a plain list.

    ``add_special_tokens`` stays at its default, so the ids are the string plus
    whatever this tokenizer's post-processor puts in front of it, and that prefix
    is a declared fact rather than a detail of the call:
    :attr:`JointRendering.prefix_marker` names it for a family whose price has
    been measured, and :func:`prefix_marker_ids` resolves it to ids. InstructProtein's
    is ``</s>`` (id 2), which its protein rendering is declared to open with and
    does, and which costs 3.90703 nats/token in that same checkpoint's *text* mode
    over 200 OpenWebText documents of at least 800 characters at the 384-token
    window (23.74888 against 19.84184; the unprefixed span starts one document
    token later, so the two are not the same target set).

    This one call governs every family, and turning the default off would move the
    ids of all of them at once -- InstructProtein's protein rendering, and
    ProLLaMA's, ProLLaMA Stage 1's and LLaMA-2's text renderings, whose prefixes
    are unmeasured. So the call stays as it is and each family declares what its
    own measurement found, which keeps the ids identical for every checkpoint this
    line governs.
    """

    return [int(value) for value in tokenizer(text, return_tensors=None)["input_ids"]]


def prefix_marker_ids(tokenizer: Any, declaration: JointRendering) -> tuple[int, ...]:
    """Ids of the token a rendering's post-processor puts in front of its text.

    The joint families' door to the question
    :func:`src.capability.core.arms.rendering_marker_ids` answers for an ``Arm``,
    resolved from the rendering's own declaration so that no caller spells a
    marker's id by hand. A measurement that has to know which positions of a
    rendering are not its own text cannot read them off the ids: the prefix
    carries no spelling that separates it from a content token on this vocabulary.

    Raises for a family that declares no prefix marker rather than returning an
    empty tuple, because an empty tuple is the answer "this rendering prefixes
    nothing" and that answer has not been established for ProLLaMA or LLaMA-2:
    their tokenizers prepend ``<s>`` (id 1) under the same ``encode`` call, and
    absence of a declaration is not absence of a prefix. The token itself is
    round-tripped through :func:`_declared_token_id`, which refuses one the
    tokenizer does not carry as a token of its own.
    """

    if declaration.prefix_marker is None:
        raise ValueError(
            f"{declaration.name} declares no prefix marker, so the tokens its "
            "tokenizer puts in front of the rendering cannot be reported. That is "
            "a different statement from 'this rendering prefixes nothing': "
            "ProLLaMA and LLaMA-2 take the same encode call and their tokenizers "
            "prepend <s> (id 1), and neither family's prefix has been priced"
        )
    return (
        _declared_token_id(tokenizer, declaration.prefix_marker, role="prefix marker"),
    )


def _declared_token_id(tokenizer: Any, token: str, *, role: str) -> int:
    """The id of a token the rendering names, refusing an absent one.

    Round-tripped rather than compared against the unknown-token id: on
    InstructProtein the unknown token *is* the beginning-of-sequence token, so an
    ``== unk_token_id`` test would refuse a legitimate id on one family and
    accept a bogus one on another.
    """

    resolved = tokenizer.convert_tokens_to_ids(token)
    if resolved is None:
        raise ValueError(f"this tokenizer has no id for the {role} {token!r}")
    resolved = int(resolved)
    if tokenizer.convert_ids_to_tokens(resolved) != token:
        raise ValueError(
            f"the {role} {token!r} resolves to id {resolved}, which decodes back to "
            f"{tokenizer.convert_ids_to_tokens(resolved)!r}; this tokenizer does not "
            "carry it as a token of its own"
        )
    return resolved


def residue_spelling_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Every vocabulary id whose token spells a non-empty string of canonical residues.

    This is the support a scored protein target can take when the symbol unit is
    the token. It is neither the twenty-letter alphabet -- a scored target there
    carries a whole residue run -- nor the full vocabulary, which would put more
    additive-smoothing bias into the held-out baseline than the effect being
    measured: 463 ids on the staged LLaMA-2 vocabulary against 32000, which is
    0.011 nats of normaliser inflation instead of 0.570 on the 41682-target
    reference this stage's defaults draw.

    Enumerated from the tokenizer rather than declared, because which residue runs
    the vocabulary happens to carry is a property of the checkpoint.
    """

    allowed = set(AA20)
    ids = tuple(
        value
        for value in range(len(tokenizer))
        if (token := tokenizer.convert_ids_to_tokens(value)) and set(token) <= allowed
    )
    if not ids:
        raise ValueError(
            "this tokenizer carries no token spelled purely of canonical residues, so "
            "a protein target has no support to be scored against"
        )
    return ids


def _spelled_sequence_run(
    tokenizer: Any, token_ids: Sequence[int], sequence: str
) -> tuple[int, ...]:
    """Positions of the one contiguous token run whose spellings are exactly ``sequence``.

    The rule for a family whose delimiters are not tokens of its vocabulary. It
    refuses the two ways it can be wrong instead of returning a plausible span:
    no run means the sequence did not start and end on token boundaries -- a
    residue merged into a delimiter or into the document context -- and more than
    one means the rendered string carries the sequence twice, so the span is
    ambiguous.
    """

    pieces = [tokenizer.convert_ids_to_tokens(int(value)) for value in token_ids]
    if any(piece is None for piece in pieces):
        raise ValueError(
            "this tokenizer does not spell every id it produced, so the scored span "
            "cannot be located by spelling"
        )
    spelling = "".join(pieces)
    starts: dict[int, int] = {}
    ends: dict[int, int] = {}
    offset = 0
    for index, piece in enumerate(pieces):
        starts[offset] = index
        offset += len(piece)
        ends[offset] = index
    hits = []
    cursor = spelling.find(sequence)
    while cursor != -1:
        if cursor in starts and cursor + len(sequence) in ends:
            hits.append((starts[cursor], ends[cursor + len(sequence)]))
        cursor = spelling.find(sequence, cursor + 1)
    if len(hits) != 1:
        raise ValueError(
            f"the rendered protein's tokens spell the {len(sequence)}-residue sequence "
            f"on token boundaries {len(hits)} times; exactly one is required. Zero "
            "means a residue merged into a delimiter or into the document context, so "
            "token position is not residue position; more than one means the rendered "
            "string carries the sequence twice and the scored span is ambiguous"
        )
    first, last = hits[0]
    return tuple(range(first, last + 1))


def _delimited_span(token_ids: Sequence[int], start_id: int, end_id: int) -> tuple[int, ...]:
    """Positions strictly between the one start delimiter and the one end delimiter."""

    ids = list(token_ids)
    if ids.count(start_id) != 1 or ids.count(end_id) != 1:
        raise ValueError(
            f"a rendered protein must carry exactly one start delimiter (id "
            f"{start_id}) and one end delimiter (id {end_id}); this one carries "
            f"{ids.count(start_id)} and {ids.count(end_id)}"
        )
    opening, closing = ids.index(start_id), ids.index(end_id)
    if closing <= opening + 1:
        raise ValueError("the rendered protein has no content between its delimiters")
    return tuple(range(opening + 1, closing))


def scored_target_positions(
    tokenizer: Any,
    declaration: JointRendering,
    rendered: str,
    *,
    rule: str | None = None,
    residue_ids: Mapping[str, int] | None = None,
    sequence: str | None = None,
) -> tuple[int, ...]:
    """Positions of the rendered string whose target token carries residues.

    ``rule`` defaults to the family's own declaration. It is an argument at all
    because the naive control has no residue tokens by construction, so the
    identity rule selects nothing there and
    :attr:`JointRendering.positional_scored_target_rule` is the substitute;
    :meth:`JointTokenisation.render` is what makes that substitution, and it makes
    it in one place. Each rule states what it needs -- ``residue_ids`` for the
    identity rule, ``sequence`` for the spelled run -- and refuses to guess.

    A position of zero is refused: nothing predicts the first token of a
    sequence, so scoring it would read a logit that does not exist.
    """

    rule = declaration.scored_target_rule if rule is None else rule
    if rule not in SCORED_TARGET_RULES:
        raise ValueError(f"unknown scored target rule {rule!r}; declared: {SCORED_TARGET_RULES}")
    token_ids = encode(tokenizer, rendered)
    if rule == SPELLED_SEQUENCE_RUN:
        if sequence is None:
            raise ValueError(
                f"{declaration.name} locates its scored positions by the spelling of "
                "the residue run, so the sequence being rendered must be supplied"
            )
        positions = _spelled_sequence_run(tokenizer, token_ids, sequence)
    else:
        start_id = _declared_token_id(
            tokenizer, declaration.protein_start, role="start delimiter"
        )
        end_id = _declared_token_id(tokenizer, declaration.protein_end, role="end delimiter")
        span = _delimited_span(token_ids, start_id, end_id)
        if rule == DECLARED_RESIDUE_IDS:
            if residue_ids is None:
                raise ValueError(
                    f"{declaration.name} identifies its scored positions by token "
                    "identity, so the declared residue ids must be supplied"
                )
            allowed = set(int(value) for value in residue_ids.values())
            positions = tuple(
                index for index, value in enumerate(token_ids) if value in allowed
            )
        else:
            positions = span
    if 0 in positions:
        raise ValueError(
            "position 0 was selected as a scored target, but nothing predicts the "
            "first token of a sequence"
        )
    return positions


@dataclass(frozen=True)
class RenderedProtein:
    """One sequence rendered, tokenised and reduced to its scored positions."""

    rendering: str
    variant: str
    text: str
    token_ids: tuple[int, ...]
    scored_positions: tuple[int, ...]
    n_residues: int

    @property
    def n_scored_tokens(self) -> int:
        return len(self.scored_positions)

    @property
    def residues_per_scored_token(self) -> float:
        """Residues per token **over the scored span**, never over the whole string.

        The denominator is named because it is the quantity that differs between
        the declared and the naive rendering: exactly 1.0 for a rendering that
        passed the per-residue verification, and about 1.79 for Galactica's
        unescaped form. A ratio taken over the whole rendered string would fold
        the delimiters and any document context into it (Appendix B rule 27).
        """

        return self.n_residues / len(self.scored_positions)


@dataclass(frozen=True)
class JointTokenisation:
    """A rendering resolved against one tokenizer, with its ids verified.

    Building this object is where a tokenizer that cannot carry the declaration
    is refused: each of the twenty canonical residues must render to exactly one
    token, all twenty distinct, and -- for a family declaring the residue as its
    symbol unit -- a multi-residue probe must survive the per-residue guard.
    Every later call reuses the resolved ids, so the probe encodings happen once
    per run rather than once per sequence.

    ``start_id`` and ``end_id`` are ``None`` exactly when the family's delimiters
    are not tokens of the vocabulary, which is also when its scored-target rule
    does not use them.

    ``scored_target_ids`` is the enumerated set of ids a scored protein target can
    take under this rendering, and it is the single source of the held-out
    unigram's support. For a residue-unit family it is the twenty residue ids; for
    a token-unit family it is every token spelled purely of canonical residues.

    ``prefix_marker_ids`` is the resolved prefix of every id list this rendering
    encodes to -- ``</s>`` (id 2) on InstructProtein -- and it is empty exactly
    when the declaration carries no prefix marker, which is a family whose prefix
    has not been priced rather than one that has none.
    """

    declaration: JointRendering
    tokenizer: Any
    start_id: int | None
    end_id: int | None
    prefix_marker_ids: tuple[int, ...]
    residue_ids: Mapping[str, int]
    scored_target_ids: tuple[int, ...]

    def render(
        self, sequence: str, *, context: str | None = None, variant: str = DECLARED
    ) -> RenderedProtein:
        """Render, tokenise, select the scored positions and verify the result.

        The declared variant is verified against the unit its family declares --
        see :meth:`verify_symbol_unit` -- and raises rather than recording a
        warning field. The naive variant is deliberately not verified: it exists
        to be wrong, and its scored span uses the family's positional rule because
        the identity rule would select nothing in it.

        Every id the tokenizer produced is checked against the tokenizer's own
        declared size before the record is built. A checkpoint may carry more
        embedding rows than its tokenizer has tokens -- InstructProtein pads
        50287 real tokens to 50304, a multiple of 128 -- and a padded tail is
        harmless only while nothing the tokenizer emits reaches it, which is the
        invariant the fitness loaders' ``len(tokenizer) <= embedding rows``
        comparison rests on.
        """

        text = self.declaration.render_protein(sequence, context=context, variant=variant)
        token_ids = tuple(encode(self.tokenizer, text))
        size = int(len(self.tokenizer))
        unrowed = sorted({value for value in token_ids if value >= size})
        if unrowed:
            raise ValueError(
                f"{self.declaration.name}: this tokenizer declares {size} tokens "
                f"but produced ids {unrowed}, which its own declared size does not "
                "cover. An embedding padded above that size is safe only while "
                "every emitted id has a row"
            )
        if self.prefix_marker_ids and token_ids[: len(self.prefix_marker_ids)] != self.prefix_marker_ids:
            raise ValueError(
                f"{self.declaration.name}: this rendering is declared to open with "
                f"{self.declaration.prefix_marker!r} (id {self.prefix_marker_ids[0]}), "
                f"and the encoding of {text!r} opens with "
                f"{list(token_ids[: len(self.prefix_marker_ids)])}. The declared "
                "prefix is what the tokenizer's post-processor puts in front of "
                "every encode() call this rendering makes, so an encoding that "
                "does not open with it means the declaration and the ids have "
                "drifted apart"
            )
        rule = (
            self.declaration.scored_target_rule
            if variant == DECLARED
            else self.declaration.positional_scored_target_rule
        )
        positions = scored_target_positions(
            self.tokenizer,
            self.declaration,
            text,
            rule=rule,
            residue_ids=self.residue_ids,
            sequence=sequence,
        )
        record = RenderedProtein(
            rendering=self.declaration.name,
            variant=variant,
            text=text,
            token_ids=token_ids,
            scored_positions=positions,
            n_residues=len(sequence),
        )
        if variant == DECLARED:
            self.verify_symbol_unit(record)
        return record

    def verify_symbol_unit(self, record: RenderedProtein) -> None:
        """Verify a declared rendering against the symbol unit its family declares.

        The dispatcher, and the only place the two cases are told apart. A family
        that declares the residue as its unit gets
        :meth:`verify_one_token_per_residue` unchanged and with no way around it.
        A family that declares the token gets the property that holds for it --
        every scored target inside the enumerated support -- rather than a
        weakened version of a property it was never going to have.
        """

        if self.declaration.symbol_unit == RESIDUE_UNIT:
            self.verify_one_token_per_residue(record)
        else:
            self.verify_scored_targets_in_support(record)

    def verify_scored_targets_in_support(self, record: RenderedProtein) -> None:
        """Refuse a scored target the held-out unigram's support cannot represent.

        The token-unit counterpart of the per-residue guard, and load-bearing for
        the same reason: the support is enumerated from the vocabulary while the
        span is located in the rendering, so a target outside it means the two
        disagree and the baseline would be fitted over a support that cannot
        carry the sample it normalises.
        """

        stray = sorted(
            {record.token_ids[position] for position in record.scored_positions}
            - set(self.scored_target_ids)
        )
        if stray:
            raise ValueError(
                f"{self.declaration.name}: scored target ids {stray} are outside the "
                f"{len(self.scored_target_ids)} ids this rendering's scored support "
                "enumerates, so the held-out unigram could not represent them"
            )

    def verify_one_token_per_residue(self, record: RenderedProtein) -> None:
        """Refuse a rendering that did not reach the model's per-residue alphabet.

        This is the guard the whole module exists for. Galactica's tokenizer
        merges amino acids into multi-residue BPE pieces without the split-marker
        escape, and the merged measurement is wrong by about 2.9 nats/token while
        looking exactly like a valid one in the artefact (EXP-R2-151). It raises;
        it does not warn, and there is no flag that downgrades it.
        """

        if record.n_scored_tokens != record.n_residues:
            raise ValueError(
                f"{self.declaration.name}: the rendering produced "
                f"{record.n_scored_tokens} scored tokens for {record.n_residues} "
                f"residues ({record.residues_per_scored_token:.3f} residues per "
                "token). The tokenizer merged residues into multi-residue pieces, "
                "so token position is not residue position and the cross-entropy "
                "is not the quantity this stage reports -- measured at about 2.9 "
                "nats/token on galactica-1.3b (EXP-R2-151, Appendix B rule 4)"
            )
        allowed = set(int(value) for value in self.residue_ids.values())
        stray = sorted(
            {record.token_ids[position] for position in record.scored_positions} - allowed
        )
        if stray:
            raise ValueError(
                f"{self.declaration.name}: scored target ids {stray} are outside the "
                "declared residue alphabet, so the scored positions are not residues"
            )

    def facts(self) -> dict[str, Any]:
        """The resolved declaration, as it reaches the artefact."""

        declaration = self.declaration
        return {
            "name": declaration.name,
            "symbol_unit": declaration.symbol_unit,
            "symbol_unit_definition": SYMBOL_UNIT_DEFINITIONS[declaration.symbol_unit],
            "protein_start": declaration.protein_start,
            "protein_end": declaration.protein_end,
            "residue_escape": declaration.residue_escape,
            "escape_before_end_delimiter": declaration.escape_before_end_delimiter,
            "protein_context_template": declaration.protein_context_template,
            "scored_target_rule": declaration.scored_target_rule,
            "residue_subspace_disjoint_from_text": (
                declaration.residue_subspace_disjoint_from_text
            ),
            "naive_control_available": declaration.naive_control_available,
            "start_token_id": None if self.start_id is None else int(self.start_id),
            "end_token_id": None if self.end_id is None else int(self.end_id),
            "prefix_marker": declaration.prefix_marker,
            "prefix_marker_ids": [int(value) for value in self.prefix_marker_ids],
            "prefix_marker_note": (
                "the token the tokenizer's post-processor puts in front of every "
                "encode() call this rendering makes, resolved from the rendering's "
                "own declaration. Empty exactly when the declaration carries no "
                "prefix marker, which is an unpriced prefix rather than an absent "
                "one"
            ),
            "delimiters_are_tokens": self.start_id is not None,
            "residue_token_ids": {
                residue: int(value) for residue, value in sorted(self.residue_ids.items())
            },
            "residue_alphabet": AA20,
            "scored_target_token_ids": [int(value) for value in self.scored_target_ids],
            "n_scored_target_token_ids": len(self.scored_target_ids),
            "scored_target_support_note": (
                "the ids a scored protein target can take under this rendering, and "
                "the support the held-out unigram is fitted over. For a residue-unit "
                "family it is the twenty residue ids; for a token-unit family it is "
                "every vocabulary token spelled purely of canonical residues"
            ),
            "declaration_module": "src/capability/models/joint_modes.py",
            "note": declaration.note,
        }


def resolve(tokenizer: Any, declaration: JointRendering | str) -> JointTokenisation:
    """Resolve a declared rendering against a tokenizer, or refuse the pair.

    The residue ids are *derived* rather than spelled a second time: each
    canonical residue is rendered as a one-residue sequence through the same
    declaration and the same span locator, and the single token that falls in the
    span is that residue's id. A tokenizer that spells one residue as several
    tokens, or that maps two residues onto one id, is refused here.

    Resolution's last act is to render a multi-residue probe through
    :meth:`JointTokenisation.render`, which runs the family's own verification.
    That is what makes resolution the refusal point rather than the first scored
    sequence: a tokenizer that handles single residues but merges adjacent ones is
    exactly the state a residue-unit claim must never be made in, and it is
    refused before any checkpoint is loaded and long before any cross-entropy
    exists to be misread.
    """

    if isinstance(declaration, str):
        declaration = rendering(declaration)
    delimiters_are_tokens = declaration.scored_target_rule != SPELLED_SEQUENCE_RUN
    start_id = end_id = None
    if delimiters_are_tokens:
        start_id = _declared_token_id(
            tokenizer, declaration.protein_start, role="start delimiter"
        )
        end_id = _declared_token_id(tokenizer, declaration.protein_end, role="end delimiter")
    residue_ids: dict[str, int] = {}
    for residue in AA20:
        rendered = declaration.render_protein(residue)
        probe = encode(tokenizer, rendered)
        span = scored_target_positions(
            tokenizer,
            declaration,
            rendered,
            rule=declaration.positional_scored_target_rule,
            sequence=residue,
        )
        if len(span) != 1:
            raise ValueError(
                f"{declaration.name}: the declared rendering of the single residue "
                f"{residue!r} produced {len(span)} scored tokens, so this tokenizer "
                "does not carry the per-residue alphabet the declaration claims"
            )
        residue_ids[residue] = probe[span[0]]
    if len(set(residue_ids.values())) != len(AA20):
        collisions = sorted(
            residue
            for residue, value in residue_ids.items()
            if list(residue_ids.values()).count(value) > 1
        )
        raise ValueError(
            f"{declaration.name}: residues {collisions} share a token id, so a scored "
            "position cannot be attributed to one residue"
        )
    scored_target_ids = (
        tuple(sorted(residue_ids.values()))
        if declaration.symbol_unit == RESIDUE_UNIT
        else residue_spelling_token_ids(tokenizer)
    )
    tokenisation = JointTokenisation(
        declaration=declaration,
        tokenizer=tokenizer,
        start_id=start_id,
        end_id=end_id,
        prefix_marker_ids=(
            ()
            if declaration.prefix_marker is None
            else prefix_marker_ids(tokenizer, declaration)
        ),
        residue_ids=residue_ids,
        scored_target_ids=scored_target_ids,
    )
    tokenisation.render(AA20)
    return tokenisation
