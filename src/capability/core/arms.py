"""The matched model panel and its frozen evaluation cohorts.

Every transfer measurement compares a text decoder against protein decoders, so
the panel must be matched where it can be and its mismatches must be recorded
rather than hidden. GPT-2-large and ProtGPT2 share depth, width and vocabulary
size exactly, which makes that pair the controlled comparison; ZymCTRL and
ProGen2-medium differ and their differences are declared in ``ArmSpec``.

Cohorts are content-addressed. A cohort is a frozen, hashed list of sequences
plus the per-arm input strings derived from it, so that two runs either use the
same cohort or fail loudly.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Iterator

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import os

from .amino_acids import AA20 as AA20
from ..models import proteinglm as _proteinglm
from .scoring import TargetTokenShuffle, target_rule


def env_path(variable: str, default: Path) -> Path:
    """Read one input location from the environment, defaulting to the L20 host.

    The same code runs on the local L20 host and inside H200 pods, which mount
    their checkpoints and corpora elsewhere. Every location is therefore a named
    variable, and relocating a host is a matter of exports rather than of edits.
    The defaults are the authoring host's historical layout and are not
    guaranteed to resolve anywhere, this host included.

    Existence is deliberately not checked here. A module-level check would make
    importing the package depend on data that a given measurement never touches,
    and would fail while a corpus is still being staged; it is checked at first
    use instead, by :func:`require_input_path`.

    The shell wins, then ``.env.local``, then the module default. The file is
    read into this module and never into ``os.environ``: a value written there
    is inherited by every subprocess the run spawns, so one host's layout would
    reach a child that had been given its own.
    """

    value = os.environ.get(variable)
    if value is None:
        value = _LOCAL_ENVIRONMENT.get(variable)
    return Path(default if value is None else value)


def require_input_path(path: Path, variable: str) -> Path:
    """Fail on a missing input, naming the variable that relocates it.

    A missing corpus or checkpoint has to stop the run, because both ways of
    tolerating it are invisible in the numbers that follow. ``transformers``
    treats an absent local directory as a Hub repository id and goes to the
    network; a glob over a missing directory returns an empty list, which reads
    downstream as "no eligible records" rather than as "wrong host".
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; set {variable} to its location on this host"
        )
    return path


def _read_local_environment() -> dict[str, str]:
    """This host's values from ``.env.local``, for :func:`env_path` to consult.

    The file is host-local and is not distributed. An entry with an empty value
    is skipped, so a copy of ``.env.example`` whose lines have not been filled
    in leaves every module default standing rather than resolving it to ``.``.
    """

    path = Path(__file__).resolve().parents[3] / ".env.local"
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and value:
            values[key] = value
    return values


#: Read once at import, read-only thereafter, and never copied into
#: ``os.environ``; see :func:`env_path` for why.
_LOCAL_ENVIRONMENT: dict[str, str] = _read_local_environment()

#: Repository root. It is only a default for the corpora below, each of which
#: carries its own variable, so a host that mounts data outside its checkout
#: does not have to pretend that the two live together.
REPO = env_path("REPO_ROOT", Path(__file__).resolve().parents[3])
MODEL_ROOT = env_path("MODEL_ROOT", REPO.parent / "models")
#: Root of the host's staged model tree, beneath which the released checkpoints
#: and tokenizer directories sit. Host-specific and not distributed.
MODEL_BASE_DIR = env_path("MODEL_BASE_DIR", MODEL_ROOT.parent)
#: Parent of the text checkpoints that are addressed by name rather than
#: declared one by one: the ByGPT5 rungs below and the GPT-2 ladder in
#: :mod:`src.capability.scaling`.
TEXT_MODEL_BASE = env_path("TEXT_MODEL_BASE_DIR", REPO.parent / "text_models")
TEXT_MODEL_ROOT = env_path("TEXT_MODEL_DIR", TEXT_MODEL_BASE / "gpt2-large")
OPENWEBTEXT = env_path(
    "OPENWEBTEXT_DIR",
    MODEL_BASE_DIR / "datasets/openwebtext-screen/plain_text",
)
SWISSPROT_FASTA = env_path(
    "SWISSPROT_FASTA", REPO / "data/swissprot/uniprot_sprot.fasta.gz"
)
ZYMCTRL_FASTA = env_path(
    "ZYMCTRL_FASTA", REPO / "data/zymctrl/ec_labeled_swissprot.fasta"
)
#: ProGen3's own pretraining-scale corpus, and the one ProGenMech trains its
#: transcoders on. ``UNIREF50_FASTA`` was already exported to the pod
#: and named in the resource manifest while no module declared it, so every
#: caller resolved it by hand or not at all.
UNIREF50_FASTA = env_path(
    "UNIREF50_FASTA", REPO / "data/uniref50/uniref50.fasta"
)

#: Named in the failure message when an arm's checkpoint is absent. Which of the
#: three applies depends on the arm, and an operator needs the candidate list
#: rather than the one this module happened to resolve.
_MODEL_PATH_VARIABLES = "MODEL_ROOT, TEXT_MODEL_DIR or TEXT_MODEL_BASE_DIR"

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

#: Value of :attr:`ArmSpec.pretraining_corpus` for a checkpoint whose model card
#: does not state its training corpus. It is a sentinel rather than a guess:
#: inventing a corpus would put a false fact into every artefact that records the
#: panel, and the corpus contrasts below are only valid between arms that declare
#: one.
PRETRAINING_UNDECLARED = "undeclared"

#: Value of :attr:`ArmSpec.input_format` for a staged checkpoint whose native
#: rendering -- what a scored record is wrapped in, and which of the resulting
#: positions are cohort content -- this repository has **not** established from
#: the checkpoint's own card. It is a sentinel for the same reason
#: :data:`PRETRAINING_UNDECLARED` is: the alternative is to borrow another arm's
#: rendering, which produces a number the checkpoint was never trained behind and
#: no downstream check can see. Every renderer in this module raises on it by
#: falling through to its own "unsupported input format" refusal, so a
#: checkpoint declared this way is loadable and shape-checkable and cannot be
#: scored until its rendering is evidenced and declared.
INPUT_FORMAT_UNDECLARED = "undeclared_native_rendering"

#: ProteinGLM continuation rendering. Single source:
#: :data:`src.capability.models.proteinglm.INPUT_FORMAT`. Not ``raw`` and not EC-conditioned.
INPUT_FORMAT_GMASK_SOP_EOS = _proteinglm.INPUT_FORMAT

#: BOS-then-direction rendering: ``<|bos|>``, then the checkpoint's direction
#: token, then the sequence. This is ProtGPT3-1.3B's native format, and it has to
#: be declared because the checkpoint's own ``tokenizer_config.json`` cannot
#: answer the question it looks like it answers. That file's post-processor is
#: Sequence-only and ``add_bos_token`` is False, so a bare residue string encodes
#: with no BOS -- but the file states what the tokenizer does *by default*, not
#: what the model was trained on, and the model card's own loading recipe passes
#: ``add_bos_token=True`` while stating that the direction tokens "should be
#: placed at the start of the protein sequence (i.e., after the BOS token) to
#: select the direction". Reading the first as evidence about the second is what
#: left this arm rendered as ``raw``, two tokens short of its training format.
#:
#: Measured on the staged checkpoint, block 0 of the gate's own Swiss-Prot draw
#: (band 64-246, n=200, digest ``17f1ca32b288``): ``raw`` costs 3.0495
#: nats/residue against this rendering's 1.7957, and after ``<|bos|>`` the model
#: places 0.499998 of its next-token mass on ``"1"``, 0.499998 on ``"2"`` and
#: ~0 on every other token of its 31-wide head -- which is a model that expects a
#: direction token there and nothing else.
INPUT_FORMAT_BOS_DIRECTION_SEQ = "bos_direction_seq"

#: Document-boundary rendering: the checkpoint's own document-boundary token --
#: :data:`EOS_BOUNDED_BOUNDARY` -- then the sequence, with the tokenizer's
#: post-processor still appending its terminal ``<EOS>``, so a scored record is
#: ``<EOS> seq <EOS>``. This is RITA-xl's native format. Its published
#: post-processor is Sequence-then-``<EOS>`` and prefixes nothing, so a bare
#: residue string encodes with no boundary at all -- but that file states what
#: the tokenizer does unconfigured, not what the checkpoint was trained behind,
#: and RITA's training stream is documents separated by that boundary token.
#:
#: Measured on the staged checkpoint over the gate's own draw (200 Swiss-Prot
#: records of 64-246 residues, seed 20260728, float32): at the position after one
#: ``<EOS>`` the model places 0.99963 of its next-token mass on the 20 residue
#: ids (mean over 64 records) and assigns the first residue a mean NLL of 0.7973
#: +/- 0.0333 nats; scoring the tokens both renderings score -- residues 2..L and
#: the terminal ``<EOS>`` -- the boundary-prefixed rendering costs 1.24520
#: nats/token against the bare rendering's 1.40124, a paired per-record mean
#: difference of -0.17384 +/- 0.01383 nats/token over 200 records, against
#: -0.00618 for a ``<PAD>``-prefixed control. Reading the post-processor's
#: silence as evidence about the training format is what left this arm short of
#: its own format, as it did for :data:`INPUT_FORMAT_BOS_DIRECTION_SEQ`.
INPUT_FORMAT_EOS_BOUNDED_SEQ = "eos_bounded_seq"

#: How a cohort's records were drawn from their corpus. ``seeded_permutation``
#: is the only mode Appendix B rule 1 of the transfer audit permits for a
#: reported number; ``file_order`` is retained because several frozen artefacts
#: were produced with it and must remain reproducible, and because a census over
#: a whole corpus is order-independent. ``seeded_permutation_group_disjoint``
#: walks the same seeded permutation but fills later slots only with records
#: that have no exact or near-duplicate edge to an earlier slot; it is the
#: near-duplicate-free draw and is not a skip window. The mode is written into
#: every cohort's metadata so that no artefact can be read without knowing which
#: one produced it -- that invisibility, not the file order itself, is what
#: manufactured an effect three times.
SAMPLING_MODES = (
    "seeded_permutation",
    "file_order",
    "seeded_permutation_group_disjoint",
)
GROUP_DISJOINT_SAMPLING_MODE = "seeded_permutation_group_disjoint"

#: The seed every campaign stage draws its corpus under, declared once.
#:
#: Two facts make a shared constant the right object rather than a per-stage
#: default. A stage's cohort must be comparable with the cohort
#: ``cohort_power.py`` *qualified* its arms on, and cohorts drawn under
#: different seeds are different populations -- EXP-R2-060 priced protein
#: cohort-block sensitivity at 0.16-0.60 nats, which is the size of several
#: effects this programme has reported. And a per-stage default is a second
#: declaration of the same decision, which is the hazard Appendix B rule 12 was
#: written about.
#:
#: A stage may still be pointed at a different draw from the command line, and
#: ``0`` selects the historical file-order draw. Both are declared choices that
#: reach the artefact through :attr:`Cohort.sampling`; neither is a default that
#: nobody notices.
DEFAULT_CORPUS_DRAW_SEED = 20260728

#: Named in the sampling record of a file-order cohort so the hazard travels
#: with the number rather than living only in a document.
FILE_ORDER_HAZARD = (
    "records are the first eligible entries of the corpus in file order; "
    "biological corpora are grouped by family, so a head-of-file draw is a set "
    "of near-clonal homologues rather than a sample. Pass seed= to draw under a "
    "seeded permutation (transfer audit, Appendix B rule 1)"
)


#: Measurement families an arm can legitimately enter. These are capabilities,
#: not preferences: an arm without a capability must raise rather than return a
#: number that is not commensurate with the rest of the panel.
CAPABILITIES = frozenset({"budget", "lens", "pathway", "circuits", "relational"})

#: Architectures that follow the Llama module convention: a block list at
#: ``model.layers``, attention at ``block.self_attn``, RMSNorm in place of
#: LayerNorm, a gated feed-forward, grouped-query attention and rotary position
#: embeddings in place of a learned position table. They are grouped here because
#: the panel resolves all of them identically, and are still declared one by one
#: on each :class:`ArmSpec` because they are different pretraining runs whose
#: differences a downstream fit has to be able to separate.
_ROTARY_DECODERS = frozenset({"llama", "qwen2"})

#: Causal-LM classes that this Transformers build already ships. ``load_arm_spec``
#: must not take ``trust_remote_code`` as a serving path for them: the class is
#: here, and remote code would be a workaround rather than a load. They are also
#: not members of :data:`_ROTARY_DECODERS`: sharing rotary embeddings does not
#: grant a decomposition, a lens, or an attention-intervention contract.
_BUILTIN_CAUSAL_LM_ARCHITECTURES = frozenset({"qwen3", "mixtral"})

#: Where each architecture keeps a block's attention submodule, as the path from
#: the block down to it. Declared per architecture rather than found by trying
#: attribute names in turn: a panel member whose attention cannot be named is a
#: panel change that must be declared, and a search would silently resolve a new
#: architecture to whichever candidate attribute happened to exist on it.
#:
#: A path rather than a single attribute because ByGPT5's attention is not an
#: attribute of its block at all: a ``ByGPT5Block`` holds a ``ModuleList`` whose
#: first entry is the ``T5LayerSelfAttention`` wrapper and whose second is the
#: ``nn.Identity`` that replaces cross-attention, and the module that computes
#: the pattern is the ``T5Attention`` inside that wrapper. Integers index, strings
#: are attributes.
#:
#: ``opt`` is absent although its attention *is* nameable -- ``self_attn`` on the
#: block, an ``OPTAttention`` whose eager forward materialises a
#: ``(batch, head, token, token)`` pattern, measured. It is left out because
#: nothing can reach it. :meth:`Arm.attention` additionally requires
#: :data:`_DECOMPOSABLE`, which refuses ``opt``;
#: :meth:`Arm.attention_pattern_module` is reached only by
#: ``src.capability.prediction_addressed``, whose own ``PAA_ARCHITECTURES`` is what
#: schedules that census and does not name ``opt`` either -- and admitting it
#: there is a separate verification of the knockout path, not a table entry. An
#: entry here alone would be a claim of support with no path to it.
_ATTENTION_PATH: dict[str, tuple[str | int, ...]] = {
    "gpt2": ("attn",),
    "progen": ("attn",),
    **{architecture: ("self_attn",) for architecture in _ROTARY_DECODERS},
    "t5_decoder": ("layer", 0, "SelfAttention"),
}

#: Architectures whose per-sublayer decomposition is commensurate with a
#: standard causal decoder.
#:
#: Declared explicitly rather than derived from :data:`_ATTENTION_PATH`, because
#: the two questions have come apart and it is now their *difference* that has to
#: be kept honest. Naming an architecture's attention submodule is what a pattern
#: read needs; claiming its sublayers are the same objects the rest of the panel
#: decomposes is a stronger statement, and ByGPT5's T5 decoder satisfies the first
#: and not the second -- its relative position bias and gated feed-forward make
#: its sublayers different objects, so it is reachable for a pattern and refused
#: for a decomposition. Reformer is absent from both: LSH attention and reversible
#: layers mean its "attention share" and "residual stream" are not the same
#: quantities the rest of the panel measures. Grouped-query attention is not such
#: a reason: sharing one value projection across a group of query heads changes
#: how a per-head decomposition must be built but not what the attention pathway
#: is, and the pathway measurements ablate a whole sublayer output rather than a
#: head.
#:
#: ``opt`` is absent, and measured rather than assumed. An ``OPTDecoderLayer``
#: has no feed-forward *submodule* at all: ``fc1`` and ``fc2`` are attributes of
#: the block itself, and the block flattens batch and time before them
#: (``hidden_states.reshape(-1, d)``) and restores the shape only after the
#: residual add. So the one module whose output is the feed-forward residual term
#: is ``fc2``, and it emits a ``(batch * token, d_model)`` tensor where every
#: other arm's MLP emits ``(batch, token, d_model)``. The values are exact --
#: block input + attention output + ``fc2`` output reproduces the block output to
#: 0.0 on galactica-125m -- but the rank is not the one the fifteen call sites
#: that hook :meth:`Arm.mlp` read, so declaring it here would hand a decomposition
#: a tensor of a different shape from every other arm's. Commensurate sublayers
#: is precisely what this set claims, and that claim would not hold.
_DECOMPOSABLE = frozenset({"gpt2", "progen", *_ROTARY_DECODERS})


@dataclass(frozen=True)
class MlpNeuronTensor:
    """How one architecture's MLP **hidden** activation is reached and checked.

    **The tensor is the MLP's hidden layer, never the MLP's output**, and the
    distinction is the whole content of this declaration. A GPT-2 feed-forward is
    ``c_proj(act(c_fc(x)))``: ``c_fc`` lifts the residual stream from ``d_model``
    to ``d_mlp``, the nonlinearity is applied there, and ``c_proj`` projects back
    down. A "neuron" is a coordinate of that ``d_mlp``-wide post-nonlinearity
    tensor. The ``d_model``-wide output is a different object -- it is a dense
    mixture of every neuron and is far less sparse -- so a neuron-basis circuit
    measured there would understate what a sparse basis can recover on *any*
    arm, and on this panel that would manufacture the conclusion that protein
    models specifically need a learned dictionary.

    Declared per architecture and reached by walking a path, exactly as
    :data:`_ATTENTION_PATH` is and for the same reason: an undeclared
    architecture must raise rather than resolve to whichever attribute happens to
    exist on it. The gated rotary lineages are absent on purpose -- their hidden
    tensor is the product of two projections, so its coordinates are not the same
    object a non-gated GELU neuron is -- and so is ``progen``, whose parallel
    residual block this programme's replacement estimand already excludes.
    """

    #: Path from the MLP module down to the module whose **input** is the hidden
    #: tensor. Hooking the down-projection's input rather than the nonlinearity's
    #: output is what makes the tensor definitionally the one the projection
    #: consumes, with no second opinion about where the MLP's stages begin.
    down_projection: tuple[str, ...]
    #: Config attributes that declare ``d_mlp``, in the order they are consulted.
    width_attributes: tuple[str, ...]
    #: What ``d_mlp`` is when every one of those is absent or ``None``. Every
    #: GPT-2 checkpoint on this panel leaves ``n_inner`` unset and means
    #: ``4 * n_embd`` by it. ``None`` here means an implicit width is not defined
    #: for the architecture and an absent attribute must raise.
    implicit_width_multiple: int | None
    #: The config attribute naming the nonlinearity.
    activation_attribute: str
    #: Greatest lower bound of each **declared** nonlinearity's output. The
    #: hidden tensor is post-nonlinearity by construction, so a measured minimum
    #: below this bound means something else was read: a pre-activation tensor is
    #: unbounded below and gives itself away immediately. An activation that is
    #: not declared raises rather than being checked against a bound that may not
    #: hold for it, which is also what keeps the non-gated claim honest -- only
    #: non-gated variants appear here.
    activation_lower_bound: dict[str, float]


#: Where each architecture keeps the activation its down-projection consumes.
#: ``gpt2`` only, and see :class:`MlpNeuronTensor` for why the others are absent.
#:
#: ``opt`` is absent for the reason it is absent from :data:`_DECOMPOSABLE`: its
#: block has no MLP module for this path to be walked from, and its hidden tensor
#: -- the input of ``fc2``, measured on galactica-125m at a minimum of -0.16997,
#: just inside the GELU bound below -- carries the block's flattened
#: ``(batch * token, d_mlp)`` shape rather than the panel's.
#:
#: ``gelu_new`` is the tanh approximation GPT-2, ProtGPT2 and ZymCTRL all declare;
#: its minimum is -0.169 at x = -0.752, so -0.17 is a true lower bound and any
#: value materially below it is not a GELU output. ``gelu`` and
#: ``gelu_pytorch_tanh`` are the same function to within 1e-3 and share the bound;
#: ``relu`` is floored at zero.
_MLP_NEURON_TENSOR: dict[str, MlpNeuronTensor] = {
    "gpt2": MlpNeuronTensor(
        down_projection=("c_proj",),
        width_attributes=("n_inner", "intermediate_size"),
        implicit_width_multiple=4,
        activation_attribute="activation_function",
        activation_lower_bound={
            "gelu_new": -0.17,
            "gelu": -0.17,
            "gelu_pytorch_tanh": -0.17,
            "relu": 0.0,
        },
    ),
}


def mlp_neuron_declaration(architecture: str) -> MlpNeuronTensor:
    """The MLP hidden-tensor declaration for an architecture, or a refusal.

    Answerable from an architecture name alone, so a stage refuses an arm it
    cannot measure before a checkpoint reaches the GPU rather than after.
    """

    declared = _MLP_NEURON_TENSOR.get(architecture)
    if declared is None:
        raise TypeError(
            f"no MLP hidden-activation tensor is declared for {architecture!r}, so "
            "a neuron basis cannot be resolved for it; declared: "
            f"{sorted(_MLP_NEURON_TENSOR)}"
        )
    return declared


@dataclass(frozen=True)
class ArmSpec:
    """Declared properties of one panel member.

    ``capabilities`` encodes which measurement families this arm may enter. It
    exists because the panel deliberately spans architectures in order to break
    the modality/tokenisation collinearity, and that breadth is only safe if a
    non-commensurate arm cannot silently reach a metric that assumes a standard
    decoder. Prose caveats get lost; a raised exception does not.

    **Two corpora, two fields.** This class used to carry one field named
    ``source``, holding the corpus the arm's *evaluation cohort* is drawn from.
    Every text arm carried ``"openwebtext"``, which is true of the cohort and
    false of the pretraining data for six of the seven text arms -- and the bare
    name ``source`` reads as provenance, so the field invited exactly the
    misreading it could not support. The two facts are now separate fields:

    ``evaluation_cohort_source``
        The corpus this arm is *scored* on. It selects a rendering and a length
        band and it is what makes two arms' cross-entropies comparable; it says
        nothing about how the arm was trained.
    ``pretraining_corpus``
        The corpus the checkpoint was *trained* on, as stated by its own model
        card, or :data:`PRETRAINING_UNDECLARED` where no card states one. This is
        the field the corpus contrasts (:data:`MATCHED_DATA_CONTRAST`,
        :data:`TEXT_DATA_CONTRAST`) are defined against, and the one an
        interpretation like "the deficit is a property of the training data"
        depends on.

    ``source`` remains as a read-only alias of ``evaluation_cohort_source`` so
    that frozen artefact schemas and existing runners keep working; it should not
    be used in new code.

    ``scoring_target_alphabet_size`` is the support of scored next-token targets,
    not the width of the logit matrix. ``None`` means the live ``config.vocab_size``
    is that support, which is the panel default. A positive integer is an explicit
    declaration and is the only way a checkpoint whose ``config.vocab_size`` is
    missing or far larger than its reachable tokens may enter a budget measurement.
    Tokenizer length is never consulted: guessing from it would hide the mismatch
    this field exists to record. The model's output columns stay untouched;
    nothing here crops logits.
    """

    name: str
    path: Path
    #: The environment variable ``path`` is built from, declared rather than
    #: inferred. It used to be recovered by comparing the resolved ``path``
    #: against the three constants, which is correct only while the three resolve
    #: to *different* directories. The H200 pod sets
    #: ``TEXT_MODEL_BASE_DIR="${MODEL_ROOT}"`` because every
    #: checkpoint sits in one GPFS directory, so on that host the comparison
    #: aliased and six text arms classified as protein-root arms. The worker
    #: re-derives the generated contract inside the pod and refused the campaign
    #: -- correctly, and only because that check exists. Appendix B rule 12: the
    #: declaration is made where the path is made.
    path_variable: str
    modality: str
    n_layer: int
    d_model: int
    tokenisation: str
    input_format: str
    evaluation_cohort_source: str
    architecture: str
    pretraining_corpus: str = PRETRAINING_UNDECLARED
    capabilities: frozenset[str] = CAPABILITIES
    scoring_target_alphabet_size: int | None = None

    def __post_init__(self) -> None:
        size = self.scoring_target_alphabet_size
        if size is None:
            return
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(
                f"{self.name}: scoring_target_alphabet_size must be a positive "
                f"integer, got {size!r}"
            )

    @property
    def source(self) -> str:
        """Deprecated alias of :attr:`evaluation_cohort_source`.

        Kept because it is the spelling several frozen artefacts and runners
        use. It is *not* the pretraining corpus and never was.
        """

        return self.evaluation_cohort_source


PANEL: dict[str, ArmSpec] = {
    "gpt2-large": ArmSpec(
        name="gpt2-large",
        path=TEXT_MODEL_ROOT,
        path_variable="TEXT_MODEL_DIR",
        modality="text",
        n_layer=36,
        d_model=1280,
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        architecture="gpt2",
        pretraining_corpus="webtext",
    ),
    "protgpt2": ArmSpec(
        name="protgpt2",
        path=MODEL_ROOT / "ProtGPT2",
        path_variable="MODEL_ROOT",
        modality="protein",
        n_layer=36,
        d_model=1280,
        tokenisation="multi_residue_bpe",
        input_format="fasta_wrapped",
        evaluation_cohort_source="swissprot",
        architecture="gpt2",
        pretraining_corpus="uniref50",
    ),
    "zymctrl": ArmSpec(
        name="zymctrl",
        path=MODEL_ROOT / "ZymCTRL",
        path_variable="MODEL_ROOT",
        modality="protein",
        n_layer=36,
        d_model=1280,
        tokenisation="residue",
        input_format="ec_conditioned",
        evaluation_cohort_source="zymctrl_ec",
        architecture="gpt2",
        pretraining_corpus="uniprot_ec_annotated",
    ),
    "progen2-medium": ArmSpec(
        name="progen2-medium",
        path=MODEL_ROOT / "progen2-medium",
        path_variable="MODEL_ROOT",
        modality="protein",
        n_layer=27,
        d_model=1536,
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        architecture="progen",
        pretraining_corpus="uniref90_bfd30",
    ),
    # The protein-side scale rung the panel lacked. 151M against ProGen2-medium's
    # 765M, same architecture, same residue tokeniser, same UniRef90+BFD30
    # pretraining mixture -- so ProGen2-small / ProGen2-medium is a within-lineage
    # scale contrast on the PROTEIN side, holding corpus and tokenisation fixed.
    #
    # Why that matters more than one more arm. Until it was admitted, scale was
    # measurable only on the text side: the GPT-2 ladder falls monotonically
    # (0.1597 -> 0.0850 on the induction fraction) and the audit's scale-adjusted
    # restatement of the head-count shortfall rests entirely on that text-side
    # slope being transportable to protein, which no measurement had tested. This
    # rung tests it inside the protein lineage.
    #
    # Load-checked on the pod before admission (EXP-R2-068): 151.1M parameters,
    # 12 blocks of width 1024, 16 heads, vocab_size 32, n_positions 1024,
    # ProGenAttention, and a forward pass returning logits of width 32 against a
    # 31-token tokenizer -- the same shape as the two ProGen2 arms already here.
    "progen2-small": ArmSpec(
        name="progen2-small",
        path=MODEL_ROOT / "progen2-small",
        path_variable="MODEL_ROOT",
        modality="protein",
        n_layer=12,
        d_model=1024,
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        architecture="progen",
        pretraining_corpus="uniref90_bfd30",
    ),
    # Architecturally identical to progen2-medium down to the parameter count
    # (764,803,616), differing only in pretraining corpus. That makes the pair a
    # controlled contrast on training data with architecture, scale and
    # tokenisation all held fixed, which is the cleanest available test of
    # whether an interpretability metric tracks data rather than modality.
    "progen2-base": ArmSpec(
        name="progen2-base",
        path=MODEL_ROOT / "progen2-base",
        path_variable="MODEL_ROOT",
        modality="protein",
        n_layer=27,
        d_model=1536,
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        architecture="progen",
        pretraining_corpus="progen2_base_mixture",
    ),
}

# DialoGPT-small is GPT-2's architecture, GPT-2's tokenizer and GPT-2's size,
# pretrained on conversational Reddit threads instead of WebText. It is therefore
# the text-side analogue of the progen2-base/progen2-medium pair: corpus varies,
# everything else is held. It exists here because the panel had three protein
# arms spanning corpora, tokenisers and architectures against a single GPT-2
# lineage on the text side, which left every cross-modality difference open to
# the reading that it is a GPT-2-large idiosyncrasy rather than a modality one.
PANEL["dialogpt-small"] = ArmSpec(
    name="dialogpt-small",
    # Addressed through TEXT_MODEL_BASE like every other checkpoint named rather
    # than declared, so that a host which mounts its text models elsewhere moves
    # this arm with the rest instead of failing on it alone.
    path=TEXT_MODEL_BASE / "DialoGPT-small",
    path_variable="TEXT_MODEL_BASE_DIR",
    modality="text",
    n_layer=12,
    d_model=768,
    tokenisation="bpe",
    input_format="raw",
    evaluation_cohort_source="openwebtext",
    architecture="gpt2",
    pretraining_corpus="reddit_dialogue",
)

# The within-lineage scale ladder, as measurable arms rather than only as
# convergence-control rungs.
#
# These three checkpoints already existed in :data:`src.capability.scaling.
# DEFAULT_LADDER`, but a ``LadderMember`` is not an ``ArmSpec``: the ladder
# feeds the convergence fit, and nothing in it can be handed to the circuit
# census. That gap is why the induction-prevalence result is still confounded.
# ``llama-3.2-3b`` is simultaneously the lowest-scoring text arm and the widest
# and deepest, so "protein decoders have fewer induction heads" and "larger
# decoders have a lower fraction" currently predict the same table.
#
# gpt2 / gpt2-medium / gpt2-large / gpt2-xl separate them, because they vary
# scale over an order of magnitude -- 124M, 355M, 774M, 1558M; 12, 24, 36 and
# 48 layers; 12, 16, 20 and 25 heads -- while holding architecture, the
# 50257-piece BPE and the WebText corpus fixed. Whatever slope the induction
# fraction has against scale is therefore measured with lineage controlled, and
# it is measured *within* the text side, so it does not borrow identification
# from the modality contrast it is meant to adjudicate. If the fraction is flat
# or rising across this ladder, a scale explanation for the protein deficit is
# refuted on the text side alone; if it falls steeply, the deficit must be
# restated against a scale-matched expectation rather than against a mean.
#
# ProtGPT2 sits inside this ladder's range at 774,030,080 parameters and
# 36x1280 -- not merely the same depth and width as gpt2-large but the same
# parameter count to the unit, verified from both checkpoints -- so the ladder
# also yields a point prediction for what ProtGPT2's fraction *should* be under
# a pure scale account.
for _name, _dir, _n_layer, _d_model in (
    ("gpt2", "gpt2", 12, 768),
    ("gpt2-medium", "gpt2-medium", 24, 1024),
    ("gpt2-xl", "gpt2-xl", 48, 1600),
):
    PANEL[_name] = ArmSpec(
        name=_name,
        path=TEXT_MODEL_BASE / _dir,
        path_variable="TEXT_MODEL_BASE_DIR",
        modality="text",
        n_layer=_n_layer,
        d_model=_d_model,
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        architecture="gpt2",
        pretraining_corpus="webtext",
    )
del _name, _dir, _n_layer, _d_model

# Architecturally diverse text decoders. Every text arm above is one lineage:
# GPT-2's architecture, GPT-2's 50257-piece BPE and either WebText or a Reddit
# corpus scraped the same way, at five sizes. The protein side meanwhile spans
# three architectures, two tokenisation families and four corpora, so a
# text-versus-protein difference measured against that text side is not
# separable from a GPT-2-versus-everything-else difference. These two arms exist
# to make it separable, with models rather than with caveats.
#
# Both are rotary, RMSNorm, gated-feed-forward, grouped-query decoders -- a
# family that postdates every other member of the panel -- from two different
# laboratories, with byte-level BPE vocabularies two-and-a-half to three times
# GPT-2's learned over corpora GPT-2 never saw. Their tokenisation *family* is
# still ``bpe``, because a byte-level BPE over 128k or 152k pieces is the same
# kind of object on the subword/symbol axis the design identifies against; what
# they add is architecture, corpus and vocabulary scale, and the measured
# characters-per-token is what says how much.
#
# Rotary position embeddings were audited rather than assumed harmless. Nothing
# the granted capabilities touch reads a position table: `budget` and `pathways`
# see only forward passes and module outputs, and the two places in
# `src.capability.circuits` that would care -- `direct_logit_attribution`, whose
# embedding term is `transformer.drop`'s output and therefore the token plus
# position sum, and `ov_copying_scores`, which reads `transformer.wte` and states
# that it drops positional embeddings -- both raise on the missing `transformer`
# attribute before reaching a position table. Position ids are the default
# `arange` rather than a cumulative sum of the attention mask, so right padding
# behaves exactly as it does for a learned table: pad positions sit after the
# content and are masked out of every causal query.
#
# Three further candidates were rejected on the evidence of their own model
# cards rather than on preference. Qwen3-0.6B, Qwen3-1.7B and
# Llama-3.2-1B-Instruct are all post-trained (SFT, preference optimisation and,
# for Qwen3, a thinking mode), while every other panel member is a pure
# next-token pretrained decoder. Admitting one would confound the architecture
# and corpus contrast these arms exist to draw with a training-objective
# contrast that no ``ArmSpec`` field records, and it would do so on the very
# axis the budget stage measures: post-training moves a model's cross-entropy on
# raw web text, which is the denominator of the convergence axis. Those
# post-trained Qwen3 rungs remain excluded from :data:`PANEL`. The pretrained
# Qwen3-8B-Base checkpoint is declared as a budget-only candidate arm, not a
# panel member: candidate serving is not panel admission and is not a widening
# of :data:`TEXT_ARCHITECTURE_CONTRAST`.

#: What a rotary decoder may enter. ``budget`` needs only a forward pass and a
#: tokenizer. ``lens`` is granted on the same footing as the ByGPT5 rungs: the
#: output aperture of a final normalisation followed by a linear unembedding is
#: the same quantity here as in GPT-2, but ``src.capability.lenses.lens_head``
#: requires an ``nn.LayerNorm`` with a learned bias at the path
#: ``src.capability.lenses.FINAL_LAYER_NORM_PATH`` declares, so until that head
#: grows an RMSNorm form the capability is an intent that
#: ``src.capability.scaling.lens_supported`` records as a reasoned skip.
#: ``pathway`` is granted outright: it ablates a whole sublayer output and needs
#: nothing but the block, MLP and attention modules this file resolves.
#:
#: ``circuits`` is withheld. Grouped-query attention is *not* the obstacle -- a
#: correct per-head decomposition replicates each key/value head's ``W_V`` across
#: its group of query heads, and rebuilding an attention layer from those weights
#: reproduces the live forward pass to a relative maximum error of 2.6e-03 in
#: bfloat16 and 5e-07 in float32, far inside the 5e-02 tolerance
#: ``circuits.verify_head_decomposition`` enforces. The obstacle is that
#: ``src.capability.circuits`` cannot express that decomposition or reach these
#: models at all: ``head_ov_weights`` knows only GPT-2's fused ``c_attn`` and
#: ProGen2's ``qkv_proj`` and infers ``d_head`` as ``d_model / n_head``;
#: ``verify_head_decomposition`` hooks ``block.ln_1``;
#: ``ov_copying_scores`` reads ``model.transformer.wte``; and
#: ``direct_logit_attribution`` requires ``transformer.drop``,
#: ``transformer.ln_f`` and a LayerNorm bias, none of which exist on an RMSNorm
#: decoder. Declaring the capability would not produce a wrong number, it would
#: produce an exception at an arbitrary depth of the run; withholding it produces
#: the panel's own refusal, with the reason attached.
# Updated 2026-07-28: ``circuits`` granted. The module now resolves the
# q_proj/v_proj/o_proj layout with the grouped-query index mapping, the
# per-architecture pre-attention norm, the tied embedding, and an explicit
# RMSNorm linearisation. Verified per architecture at bfloat16 against a 5e-2
# tolerance: OV rebuild error 4.11e-3 (Qwen2) and 5.69e-3 (Llama), about 3e-7 in
# float32, so those are rounding rather than structure.
#
# Two failure modes were checked rather than assumed, and they differ in a way
# that matters. Omitting Qwen2's ``v_proj`` bias gives a rebuild error of 0.518,
# ten times over tolerance, so it fails loudly. Applying LayerNorm's algebra to
# an RMSNorm decoder *passes* the reconstruction gate at 0.49% logit error while
# producing a systematically wrong attribution -- silent, and the reason the
# explicit normalisation-form resolution is load-bearing rather than cosmetic.
_ROTARY_CAPABILITIES = frozenset({"budget", "lens", "pathway", "circuits"})

# Qwen2.5-0.5B: the base checkpoint, pretrained only. 24 layers of width 896 is
# a depth-to-width ratio no other panel member has, and its 151936-piece
# vocabulary is the largest in the panel by a factor of three. Corpus: the
# Qwen2.5 pretraining mixture, multilingual web text with a heavy code and
# mathematics component, which shares neither language distribution nor
# provenance with WebText.
PANEL["qwen2.5-0.5b"] = ArmSpec(
    name="qwen2.5-0.5b",
    path=TEXT_MODEL_BASE / "Qwen2.5-0.5B",
    path_variable="TEXT_MODEL_BASE_DIR",
    modality="text",
    n_layer=24,
    d_model=896,
    tokenisation="bpe",
    input_format="raw",
    evaluation_cohort_source="openwebtext",
    pretraining_corpus="qwen2.5_pretraining_mixture",
    architecture="qwen2",
    capabilities=_ROTARY_CAPABILITIES,
)

# Llama-3.2-3B: the base checkpoint, not the instruction-tuned sibling that is
# staged beside it. 28 layers matches ProGen2-medium's 27 almost exactly while
# differing in everything else, and at 3.2B parameters it extends the panel's
# scale range above GPT-2-xl. Corpus: up to 9T tokens of public web data with
# Llama-3.1 logits used as token-level targets during pretraining, per its model
# card -- a distillation signal no other panel member carries, recorded here
# because it is the one respect in which this arm is not a plain next-token run.
PANEL["llama-3.2-3b"] = ArmSpec(
    name="llama-3.2-3b",
    path=TEXT_MODEL_BASE / "Llama-3.2-3B",
    path_variable="TEXT_MODEL_BASE_DIR",
    modality="text",
    n_layer=28,
    d_model=3072,
    tokenisation="bpe",
    input_format="raw",
    evaluation_cohort_source="openwebtext",
    pretraining_corpus="llama3_web_corpus_with_llama3.1_logit_distillation",
    architecture="llama",
    capabilities=_ROTARY_CAPABILITIES,
)

#: Same architecture and parameter count, different pretraining corpus.
MATCHED_DATA_CONTRAST = ("progen2-base", "progen2-medium")

#: The protein-side scale ladder: same architecture, same residue tokeniser, same
#: UniRef90+BFD30 mixture, 151M against 765M. The text side has had a four-rung
#: ladder since EXP-R2-057 and the protein side had none, which is why every
#: scale-adjusted statement about the head-count shortfall has had to assume the
#: text-side slope transports. This pair is the first protein-internal test of
#: that assumption. Two rungs is a slope estimate with no curvature, and it is
#: declared as a contrast rather than a ladder for that reason.
PROTEIN_SCALE_CONTRAST = ("progen2-small", "progen2-medium")

#: The text-side equivalent: gpt2 and DialoGPT-small share architecture,
#: tokeniser and size (12 layers, width 768, 50257 vocabulary) and differ only in
#: pretraining corpus. Comparing the two contrasts bounds how much of any
#: cross-modality difference is corpus rather than modality.
TEXT_DATA_CONTRAST = ("gpt2", "dialogpt-small")

#: The text arms that are outside the GPT-2 lineage in architecture, tokeniser
#: and corpus at once. A cross-modality difference has to survive replacing
#: GPT-2-large with these before it can be read as a text/protein difference
#: rather than a property of GPT-2, which is the objection they exist to answer.
#: Named here so that the arms carrying that argument are a value the analysis
#: can select on, not a fact a reader has to reconstruct from the panel.
TEXT_ARCHITECTURE_CONTRAST = ("qwen2.5-0.5b", "llama-3.2-3b")

# Byte-level text decoders. These exist to populate the text x symbol-level cell
# of the modality x tokenisation design, which was empty and left the two
# indicators nearly collinear. ProtGPT2 is the only public protein model with
# genuine subword tokenisation, so the protein x subword cell cannot be grown
# past n=1 and this is the only side of the design that can be repaired.
#
# The cost is an architecture difference, declared here rather than in prose:
# ByGPT5 is T5-derived (relative position bias, T5 layer norm, gated GELU) and
# admits no GPT-2-style sublayer decomposition, so it carries no `pathway`
# capability and no arm of it may enter the per-head circuit decomposition of
# `src.capability.circuits`. Reformer uses LSH attention with reversible layers and
# is restricted to budget alone -- its attention share and residual stream are not
# the quantities the rest of the panel measures.
#
# Updated 2026-08-05: `circuits` granted, `pathway` still withheld, and the gap
# between the two is the point. The prediction-addressed-attention census reads
# and overrides *attention patterns*; it never splits a block into an attention
# and an MLP term, never rebuilds an OV circuit and never touches a position
# table, so the decomposition objection above does not reach it. Withholding
# `circuits` on that objection left that census with no byte-level TEXT arm at
# all, which is the one control that separates "symbol-level tokenisation" from
# "protein model" in its head-retrieval result: every symbol-level arm in the
# panel is otherwise a protein decoder (transfer audit, EXP-R2-114).
#
# What the grant admits and what it does not was checked rather than assumed.
# `circuit_primitives` and `induction_path_patching` both gate on their own
# module's architecture declaration -- `circuits._CIRCUIT_ARCHITECTURES` and
# `path_patching.SUPPORTED_ARCHITECTURES` -- and neither contains `t5_decoder`,
# so those stages still refuse these arms with their own reason attached.
for _name, _layers, _width in (
    ("bygpt5-small-en", 4, 1472),
    ("bygpt5-base-en", 6, 1536),
    ("bygpt5-medium-en", 12, 1536),
):
    PANEL[_name] = ArmSpec(
        name=_name,
        path=TEXT_MODEL_BASE / _name,
        path_variable="TEXT_MODEL_BASE_DIR",
        modality="text",
        n_layer=_layers,
        d_model=_width,
        tokenisation="byte",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        # No ByGPT5 model card on this host states a pretraining corpus, so no
        # corpus contrast is defined against these rungs. A guess here would be a
        # false fact in every artefact that records the panel.
        pretraining_corpus=PRETRAINING_UNDECLARED,
        architecture="t5_decoder",
        capabilities=frozenset({"budget", "lens", "circuits"}),
    )

# google/reformer-enwik8 was staged as an architecturally independent byte-level
# text model and is deliberately NOT in the panel. It ships no tokenizer: the
# checkpoint expects text encoded manually as byte+2, and AutoTokenizer resolves
# a ReformerTokenizer that then fails for want of a sentencepiece vocab file.
# Admitting it would mean a bespoke tokenizer shim in this shared module for a
# model that, being LSH-attention and reversible-layered, can only ever
# contribute to the `budget` family anyway. Three ByGPT5 rungs already populate
# the text x byte-level cell. The checkpoint remains on disk if an independent
# architecture check is later judged worth that cost.

#: Checkpoints staged beside the panel that are deliberately **not** panel
#: members, keyed by the name a measurement reaches them under.
#:
#: The distinction this table exists to keep is between "not staged" and "staged
#: and not admitted". ``scripts/capability/panel_contract.STAGED_BUT_NOT_ADMITTED``
#: records the second fact and its measured reason for the campaign; this table
#: is the declaration a *loader* needs. The two must name the same checkpoints,
#: and they do: eight keys on each side. No test asserts it any more -- the file
#: that did, ``tests/test_replaceable_arms.py``, did not survive the test
#: restructure -- so this is a stated requirement, not a checked one. A library
#: module must not import a stage script to validate itself, so the check belongs
#: in a test rather than here.
#:
#: **Why these two are not in** :data:`PANEL`. Both are ProGen2 rungs above
#: ``progen2-medium``, both load and run, and both would corrupt a panel-wide
#: statistic on admission. ``progen2-large`` declares ``vocab_size`` 51200
#: against a 31-token tokenizer, so every quantity derived from that key -- the
#: held-out unigram support, the plug-in entropy, the rank-(V-1) aperture -- would
#: be computed over a mostly dead alphabet; ``progen2-xlarge`` carries no
#: ``vocab_size`` key at all, only ``vocab_size_emb``/``vocab_size_lm_head``, so
#: the same code raises on it instead. Neither fact touches a measurement that
#: reads only the tokenizer and the block outputs, which is why they are reachable
#: here rather than unreachable everywhere.
#:
#: ``capabilities`` now include ``budget`` because the scoring-target alphabet
#: is declared as 32 rather than read from ``config.vocab_size``. That grant does
#: not admit either rung to :data:`PANEL`: the 51200-column large head and the
#: xlarge config that omits ``vocab_size`` remain reasons to keep them out of
#: every panel-wide statistic that still defaults to the config key. An opt-in
#: scale measurement that goes through :func:`scoring_target_alphabet` can score
#: them; a campaign stage that indexes :data:`PANEL` still cannot.
STAGED_ARMS: dict[str, ArmSpec] = {
    # 2779.4M parameters, 32 blocks of width 2560, 32 heads (EXP-R2-068).
    "progen2-large": ArmSpec(
        name="progen2-large",
        path=MODEL_ROOT / "progen2-large",
        path_variable="MODEL_ROOT",
        modality="protein",
        n_layer=32,
        d_model=2560,
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        architecture="progen",
        pretraining_corpus="uniref90_bfd30",
        capabilities=frozenset({"budget", "pathway"}),
        scoring_target_alphabet_size=32,
    ),
    # 6443.6M parameters, 32 blocks of width 4096, 16 heads. Its config declares
    # ``embed_dim`` and no ``n_embd``, which :func:`load_arm_spec` already
    # resolves through its declared fallback order.
    "progen2-xlarge": ArmSpec(
        name="progen2-xlarge",
        path=MODEL_ROOT / "progen2-xlarge",
        path_variable="MODEL_ROOT",
        modality="protein",
        n_layer=32,
        d_model=4096,
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        architecture="progen",
        pretraining_corpus="uniref90_bfd30",
        capabilities=frozenset({"budget", "pathway"}),
        scoring_target_alphabet_size=32,
    ),
}

# ---------------------------------------------- EXP-R2-225 second-stage staging
#
# Four larger public checkpoints, staged for the independent second
# stage. They are declared here for the same reason the two ProGen2 rungs above
# are: a checkpoint that a measurement may reach needs one declaration of its
# depth, width, rendering, corpus and scoring alphabet, and inventing any of
# those at a call site is how a wrong number gets written. Declaring them is
# **not** admission: none is in :data:`PANEL`, :func:`load_arm` still refuses
# every one by name, and the only door that names them is
# :data:`STAGED_SECOND_STAGE_ARMS`, which every stage must be opted into
# explicitly.
#
# ``capabilities`` below is what each arm was measured to honour today, not what
# it would be convenient to run, and two of the four therefore declare nothing at
# all. Two reasons compose, and they are independent.
#
# Architecture: only ``qwen2`` is declared in :data:`_ATTENTION_PATH`,
# :data:`_DECOMPOSABLE`, ``scaling.LENS_ARCHITECTURES`` and
# ``circuits._CIRCUIT_ARCHITECTURES``, so only the two Qwen rungs could carry an
# interpretability family at all; ``rita`` and ``proteinglm`` are in none of
# those tables and a grant would raise at an arbitrary depth of a run instead of
# refusing at its start. Widening those tables is a separate decision with its
# own verification and is not taken here.
#
# Interface: ``budget`` needs only a forward pass and a tokenizer, so it is the
# one family an undeclared architecture could still honour. ``rita-xl`` still
# cannot: it ships an empty ``special_tokens_map.json``, declares no pad token
# and names an out-of-vocabulary end-of-sequence id, so :func:`tokenize_batch`
# refuses every batch. ``proteinglm-7b-clm`` now serves the continuation
# rendering through :mod:`src.capability.models.proteinglm` and may declare ``budget``;
# that is an implementation door, not 7B numerical qualification.

# **Galactica is deliberately NOT declared here, at either rung.** EXP-R2-225's
# joint 1.3->6.7->30B trajectory is a wave of this campaign, and the two upper
# rungs were briefly declared in this table; they are not, because
# ``joint_mode_qualification.py`` states that a joint checkpoint which has not
# passed it "must not be in ``arms.py`` at all" and no route needs them here.
#
# The rule holds because the checkpoint path is sufficient, which was checked
# stage by stage rather than assumed. Every stage that reads a joint
# checkpoint -- 15, 17, 23, 24, 25, 30, 32, 33 and 38 -- takes ``--checkpoint``
# (or ``--joint-checkpoint``) with ``--rendering`` and reaches the weights
# through ``21_joint_mode_qualification.load_tokenizer``/``load_model`` into
# :class:`src.capability.replaceable.JointReplaceable`; none of them resolves an
# ``ArmSpec``. The lens, circuit and pathway stages take ``--arms`` out of
# :data:`PANEL` and load through :func:`load_arm`, which refuses a staged name
# outright, so a row here would not have reached them either. And the budget
# estimand is not withheld by the absence: stage 21 computes stage 01's own
# context information on a joint checkpoint by path, deliberately, so that a
# joint reading is commensurable with the panel's.
#
# A row here would also have cost something. ``scaling.register_arm_spec`` --
# the one door that can put a rung under the lens without editing this file --
# refuses by name every checkpoint that :data:`STAGED_ARMS` declares, so
# declaring Galactica here *blocks* the scale-ladder route rather than opening
# it.
#
# And an ``ArmSpec`` cannot describe this checkpoint in the first place. One spec
# carries one modality, one ``input_format`` and one evaluation cohort; Galactica
# has two of each, and its protein rendering --
# ``[START_AMINO]...[END_AMINO]``, declared in
# :data:`src.capability.models.joint_modes.JOINT_RENDERINGS` -- is one
# :meth:`Cohort.input_strings` cannot produce. A row here could only ever have
# been the text half, which is not the modality contrast the joint rungs are
# staged for.
#
# What the architecture can do is a separate question from what may be declared,
# and it is answered where it belongs: ``opt`` is declared in
# :meth:`Arm.blocks` and in :data:`src.capability.lenses.FINAL_LAYER_NORM_PATH`,
# so a ladder rung declared through ``scaling.LadderMember`` reaches the lens
# family, and this module's tokenizer door now resolves Galactica's pad token
# from its own config (see :func:`adopt_config_declared_pad_token`, whose
# docstring records the measurement on all four rungs).

# Qwen2.5-7B and Qwen2.5-32B, the upper two rungs of EXP-R2-225's pure-text
# 0.5->7->32B trajectory, and the same architecture, tokenizer and pretraining
# mixture as the panel's ``qwen2.5-0.5b``. **Base checkpoints only**: the
# instruction-tuned siblings are a training-objective contrast no ``ArmSpec``
# field records, and the prereg forbids substituting one.
#
# These two may honestly carry :data:`_ROTARY_CAPABILITIES`, and they are the
# only two of the four that may carry anything past ``budget``: ``qwen2`` is
# declared in every table those families resolve through, and the per-head
# decomposition additionally verifies itself against the live forward pass, so a
# checkpoint whose layout differed from the 0.5B's would fail that check rather
# than produce a silent number. ``lens`` remains the intent
# ``scaling.lens_supported`` records as a reasoned skip, exactly as on the 0.5B.
#
# **The declared context is 131072 and must be capped wherever a length is
# chosen.** Both configs declare ``max_position_embeddings`` 131072, against the
# 1024 and 2048 of every protein arm here and the 384-token window stage 01
# scores at. Nothing in this module reads it -- :func:`tokenize_batch` takes
# ``max_len`` as an argument -- but the two fitness stages derive an arm's
# context as ``n_positions or max_position_embeddings`` and would read 131072
# for these rungs, so a caller that ever routes one of them through a
# context-derived length must cap it at the panel's own window rather than at
# this checkpoint's ceiling.
for _name, _dir, _layers, _width in (
    ("qwen2.5-7b", "Qwen2.5-7B", 28, 3584),
    ("qwen2.5-32b", "Qwen2.5-32B", 64, 5120),
):
    STAGED_ARMS[_name] = ArmSpec(
        name=_name,
        path=TEXT_MODEL_BASE / _dir,
        path_variable="TEXT_MODEL_BASE_DIR",
        modality="text",
        n_layer=_layers,
        d_model=_width,
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        architecture="qwen2",
        pretraining_corpus="qwen2.5_pretraining_mixture",
        capabilities=_ROTARY_CAPABILITIES,
        # config.vocab_size. 152064 on both, which is the 0.5B's 151936 padded
        # further; declared rather than read because every staged arm declares.
        scoring_target_alphabet_size=152064,
    )
del _name, _dir, _layers, _width

# ProteinGLM-7B-CLM, EXP-R2-225's new-lineage pure-protein decoder point. 36
# blocks of width 4096, a 128-symbol padded vocabulary, a 1024-position context,
# and **float32 weights on disk** -- the only checkpoint in this file that is not
# stored in a half precision, so a caller that wants half precision asks for it
# and the loader's observed-dtype check enforces the answer.
#
# **Native rendering is served as** :data:`INPUT_FORMAT_GMASK_SOP_EOS`.
# What was measured on the real 7B weights -- not read off the card -- is that
# the native rendering is the card's own generation prompt used as a prefix:
# ``<gmask><sop><eos>`` at ids ``[29, 32, 34]``, followed by the residue run,
# scored over residues 2..L. It reads **1.1277 nats/residue** against **2.8974**
# for the same residues shuffled and **16.9930** for a bare residue string with
# no prefix. Those 7B numbers remain a future reproduction target for a
# qualified 7B cell; they are not claimed by the tiny serving tests. Two
# independent corroborations recorded with that probe: the only special-token
# embedding rows with a trained norm are ``<gmask>`` 1.4439, ``<sop>`` 1.2600
# and ``<eos>`` 1.6526 against ~1.057 for every other special and all 92 unused
# rows, which also settles that the terminator is ``<eos>`` (34) and not
# ``<eop>`` (33); and the head is tied. Total mass on all untrained ids is
# 5.4e-08, so the uncropped 128-wide logits are safe to score.
#
# The served estimand is continuation NLL (prefix plus residues 2..L, no tail
# EOS). It is not a full protein CLM over 1..L plus EOS, and it is not
# ProteinGym admission.
#
# **What the tokenizer does, corrected.** This declaration previously recorded
# that ``ProteinGLMTokenizer`` splits on whitespace so an unspaced residue string
# reaches the model as one out-of-vocabulary word. That is measurably false and
# is retracted: the tokenizer sets ``unique_no_split_tokens`` to the whole
# vocabulary and updates its trie, so the trie splits the string before
# ``_tokenize`` is reached. ``tokenize('MLFVVL')`` returns the six residues and a
# 101-residue sequence through :func:`tokenize_batch`'s exact call returns 102
# ids with zero ``<unk>`` and an exact round trip. Its ``pad_token_id`` is 0, so
# unlike ``rita-xl`` it can build batches. The tokenizer is not a blocker.
#
# **The original AutoModel path is still unloadable.**
# ``modeling_proteinglm.py`` line 15 reads ``import torch, deepspeed``, and
# Transformers' AST-based ``check_imports`` fires on the presence of that name
# before the module body runs. The name is only used inside ``get_checkpoint_fn()``,
# reachable only from a training path guarded by ``gradient_checkpointing``,
# ``self.training`` and ``torch.is_grad_enabled()``. Serving copies those three
# Python files into a derived package, drops the top-level alias, and makes
# ``get_checkpoint_fn`` raise rather than substituting ``torch.checkpoint``.
# The checkpoint is cc-by-nc-4.0 and is not vendored.
#
# **Logits are batch-first; hidden states are not.** Budget scoring reads
# ``logits`` of shape ``[batch, seq, vocab]`` and does not add an axis adapter.
# Hidden states remain ``[seq, batch, hidden]``. ``output_attentions`` is dead
# and ``attn_implementation`` is inert -- the only switch is
# ``config.use_pytorch_sdpa``. ``lens``, ``circuits``, ``pathway`` and
# ``relational`` stay unggranted. ``capabilities={'budget'}`` means the
# continuation path is implemented, not that the 7B checkpoint is qualified.
STAGED_ARMS["proteinglm-7b-clm"] = ArmSpec(
    name="proteinglm-7b-clm",
    path=MODEL_ROOT / "proteinglm-7b-clm",
    path_variable="MODEL_ROOT",
    modality="protein",
    n_layer=36,
    d_model=4096,
    tokenisation="residue",
    input_format=INPUT_FORMAT_GMASK_SOP_EOS,
    evaluation_cohort_source="swissprot",
    architecture="proteinglm",
    # Card citations Chen et al. (arXiv:2401.06199) and Cheng et al.
    # (arXiv:2411.02142): UniRef50/S + UniRef90 + ColabFoldDB. The staged
    # UniRef50 snapshot is not identified as that mix.
    pretraining_corpus="uniref50s_uniref90_colabfolddb",
    capabilities=frozenset({"budget"}),
    # ``padded_vocab_size`` and ``vocab_size`` both read 128 and the head is
    # built at that width; the tokenizer's own file lists 128 symbols.
    scoring_target_alphabet_size=128,
)

# RITA-XL, EXP-R2-225's secondary new-architecture single point. 24 blocks of
# width 2048, a 26-symbol residue vocabulary, a 1024-position context, float16
# weights. Its rendering is :data:`INPUT_FORMAT_EOS_BOUNDED_SEQ` -- the
# document-boundary ``<EOS>`` its training stream separates documents with, then
# the sequence, with the tokenizer's own terminal ``<EOS>`` still appended -- and
# the reason it is not ``raw`` is the reason its context-information envelope was
# measured on an input the checkpoint does not use: the released post-processor
# prefixes nothing, which is what the tokenizer does unconfigured rather than
# what the checkpoint was trained behind. Its card's own table reports the
# UniRef-100 language-model loss, which is a loss on a document stream whose
# separator is that boundary; scored as a bare string the arm is short of it by
# one token, and the position that predicts residue 1 has no boundary to
# condition on.
#
# ``rita`` is in none of the architecture tables the interpretability families
# resolve through, so ``budget`` would be the most it could carry -- and it
# cannot carry that either, on a measured tokenizer fact. Its
# ``special_tokens_map.json`` is literally ``{}`` while its ``tokenizer.json``
# defines ``<PAD>`` at id 1 and ``<EOS>`` at id 2, so the loaded tokenizer
# reports no pad and no end-of-sequence token and :func:`load_arm_spec`'s
# ``pad_token = eos_token`` fallback has nothing to copy. Neither does the
# config-declared step behind it: this config declares **no** ``pad_token_id``
# at all, and the ``eos_token_id`` 50256 it does declare is an id its 26-symbol
# vocabulary does not contain. So the tokenizer this loader returns still reports
# no pad token and :func:`tokenize_batch` still refuses every batch -- one-record
# batches included -- with "tokenizer has no pad token". That is why the
# capability is withheld rather than deferred to a fallback: there is nothing
# here to resolve a pad token *from*.
#
# Measured on this host: loads strictly clean at 1,208,655,872 parameters,
# 24 x 2048 matching the declaration, live output width 26 from
# ``lm_head.out_features``, and a 16-residue probe returns finite logits at a
# mean next-token NLL of 2.2363 nats. The tokenizer appends ``<EOS>`` to a bare
# residue string of its own accord and prefixes nothing; that is why the
# boundary is rendered here rather than left to the tokenizer. Its remote
# modeling code additionally fails a float16 forward pass on CPU -- ``att @ v``
# mixes float32 and half inside its own attention -- so the numbers above are the
# float32 reading.
#
# ``scoring_target_alphabet_size`` is declared at the config's 26 rather than
# inherited, as every staged arm's is.
STAGED_ARMS["rita-xl"] = ArmSpec(
    name="rita-xl",
    path=MODEL_ROOT / "RITA_xl",
    path_variable="MODEL_ROOT",
    modality="protein",
    n_layer=24,
    d_model=2048,
    tokenisation="residue",
    input_format=INPUT_FORMAT_EOS_BOUNDED_SEQ,
    evaluation_cohort_source="swissprot",
    architecture="rita",
    pretraining_corpus="uniref100",
    capabilities=frozenset(),
    scoring_target_alphabet_size=26,
)

# ---------------------------------------------- budget-only candidate serving
#
# Two public checkpoints staged for the shared measurement premises' budget /
# loglikelihood path, and for that path only. They are a **third** staged door,
# not a widening of :data:`STAGED_SCALE_ARMS` or :data:`STAGED_SECOND_STAGE_ARMS`,
# not members of :data:`PANEL` or :data:`CAMPAIGN_PANEL`, and not an experiment
# admission. :func:`load_arm` still refuses both by name. Stage 01 reaches them
# only through ``--allow-candidate-arms``.
#
# ``capabilities`` is ``{budget}`` and nothing else. Both architectures are
# rotary, and that is not a grant: they are absent from :data:`_ROTARY_DECODERS`,
# :data:`_DECOMPOSABLE` and :data:`_ATTENTION_PATH`, so a lens, a pathway split
# or an attention intervention raises rather than silently resolving through the
# Qwen2/Llama walk. QK-Norm, GQA and Mixtral routing are separate qualifications.
#
# Builtin classes only: ``qwen3`` and ``mixtral`` load with
# ``trust_remote_code=False``. Missing local directories fail closed through
# :func:`require_input_path`; the Hub is not a fallback.

# Qwen3-8B-Base, the pretrained (not Instruct) 8B dense decoder. 36 blocks of
# width 4096, GQA 32/8, QK-Norm, RoPE, SwiGLU, vocab_size 151936. Tokenizer is
# Qwen2TokenizerFast: ``bos_token`` is None, ``add_bos_token`` is not set, pad
# and eos are both ``<|endoftext|>`` at 151643, and a raw document is encoded
# without a BOS prefix. ``config.pad_token_id`` is unset; the tokenizer already
# declares the pad, so :func:`load_arm_spec`'s eos-copy is not the source of it.
# Context ceiling 32768 must be capped at the evaluation window, as with the
# Qwen2.5 rungs. Path is ``TEXT_MODEL_BASE / "Qwen3-8B-Base"``, which is not
# the Instruct sibling ``Qwen3-8B``.
STAGED_ARMS["qwen3-8b-base"] = ArmSpec(
    name="qwen3-8b-base",
    path=TEXT_MODEL_BASE / "Qwen3-8B-Base",
    path_variable="TEXT_MODEL_BASE_DIR",
    modality="text",
    n_layer=36,
    d_model=4096,
    tokenisation="bpe",
    input_format="raw",
    evaluation_cohort_source="openwebtext",
    architecture="qwen3",
    pretraining_corpus=PRETRAINING_UNDECLARED,
    capabilities=frozenset({"budget"}),
    scoring_target_alphabet_size=151936,
)

# ProtGPT3-1.3B, a Mixtral-style protein decoder MoE: 17 blocks of width 1024,
# 8 experts top-2, GQA 16/4, vocab_size 31. Tokenizer is LlamaTokenizerFast over
# a WordLevel character split. Measured, not assumed from the vocabulary size:
# each of :data:`AA20` encodes as exactly one id and single-token decode returns
# that residue; a full-string decode inserts spaces (``"A C D E"``) which do not
# enter the AA20 per-symbol count, so reported units stay nats/token. Pad/bos/eos
# are ids 0/1/2 and the direction tokens are the ordinary entries 4/5. Non-alphabet
# symbols such as ``J`` map to ``[UNK]`` id 3, which is inside the 31-wide head;
# tokenizer.unk_token ``<unk>`` is id 33 and is **outside** that head, and AA20
# strings do not emit it. Weight licence is unknown: that is a disclosure, not
# a loader error, and it is not experiment admission.
#
# ``input_format`` is :data:`INPUT_FORMAT_BOS_DIRECTION_SEQ`, the format the card
# declares and the checkpoint was trained behind. It used to be ``raw``, on the
# reasoning that ``add_bos_token`` is False and the post-processor is
# Sequence-only -- which is true of the *published tokenizer default* and false of
# the *training format*: the card's loading recipe overrides that default with
# ``add_bos_token=True``, names two direction tokens, and says they follow the
# BOS. Scored as ``raw`` this arm was two tokens short of its own format and read
# below its held-out unigram baseline on all eight gate blocks; scored natively it
# reads +1.0825 nats/residue of context information on the gate's block 0 against
# ``raw``'s -0.1767 on the same records and the same held-out reference.
STAGED_ARMS["protgpt3-1.3b"] = ArmSpec(
    name="protgpt3-1.3b",
    path=MODEL_ROOT / "ProtGPT3-1.3B",
    path_variable="MODEL_ROOT",
    modality="protein",
    n_layer=17,
    d_model=1024,
    tokenisation="residue",
    input_format=INPUT_FORMAT_BOS_DIRECTION_SEQ,
    evaluation_cohort_source="swissprot",
    architecture="mixtral",
    pretraining_corpus=PRETRAINING_UNDECLARED,
    capabilities=frozenset({"budget"}),
    scoring_target_alphabet_size=31,
)


def _check_staged_arms() -> None:
    """A staged checkpoint is not a panel member, and says which it is."""

    both = sorted(set(STAGED_ARMS) & set(PANEL))
    if both:
        raise AssertionError(
            f"{both} are declared both as panel members and as staged non-members; "
            "one name must mean one thing"
        )
    for name, spec in STAGED_ARMS.items():
        if spec.name != name:
            raise AssertionError(f"staged arm {name!r} declares the name {spec.name!r}")
        if spec.scoring_target_alphabet_size is None:
            raise AssertionError(
                f"staged arm {name!r} must declare scoring_target_alphabet_size; "
                "config.vocab_size is not its scoring alphabet"
            )


_check_staged_arms()


def arm_spec(name: str) -> ArmSpec:
    """The declaration for a panel arm or a staged checkpoint.

    One resolver for both, so that a measurement which admits staged checkpoints
    reads their modality, rendering and evaluation corpus from the same field a
    panel arm's come from. It does **not** widen :func:`load_arm`: that door stays
    :data:`PANEL`-only, and a caller that means to load a staged checkpoint says
    so by resolving the spec here and passing it to :func:`load_arm_spec`.
    """

    if name in PANEL:
        return PANEL[name]
    if name in STAGED_ARMS:
        return STAGED_ARMS[name]
    if name in PROGEN3_ARMS:
        return PROGEN3_ARMS[name]
    raise KeyError(
        f"unknown arm {name!r}; the panel is {sorted(PANEL)}, the staged "
        f"non-members are {sorted(STAGED_ARMS)}, and the ProGen3 door is "
        f"{sorted(PROGEN3_ARMS)}"
    )


def scoring_target_alphabet(
    spec: ArmSpec, config: Any | None = None
) -> dict[str, Any]:
    """The scoring-target alphabet size and the declaration it came from.

    This is the support of next-token target ids, not the logit width. A panel
    member with no explicit size reads ``config.vocab_size``. A staged scale
    rung must declare the size on :class:`ArmSpec`; the live config is then
    ignored for this quantity so progen2-large's 51200-column head cannot enter
    a unigram support. Tokenizer length is not a fallback.
    """

    declared = spec.scoring_target_alphabet_size
    if declared is not None:
        size = int(declared)
        source = SCORING_TARGET_ALPHABET_DECLARED
    else:
        if config is None or not hasattr(config, "vocab_size"):
            raise ValueError(
                f"{spec.name}: scoring-target alphabet is not declared and "
                "config.vocab_size is absent; tokenizer length is not a substitute"
            )
        raw = config.vocab_size
        if raw is None:
            raise ValueError(
                f"{spec.name}: config.vocab_size is None; declare "
                "scoring_target_alphabet_size rather than guessing the tokenizer"
            )
        size = int(raw)
        source = SCORING_TARGET_ALPHABET_CONFIG
    if size < 1:
        raise ValueError(
            f"{spec.name}: scoring-target alphabet size must be positive, got {size}"
        )
    return {
        "size": size,
        "source": source,
        "arm": spec.name,
        "not_logit_width": (
            "this is the support of scored target ids; it is not the number of "
            "logit columns and does not authorise cropping the model output"
        ),
    }


def require_scoring_target_ids(
    target_ids: Any, alphabet: Mapping[str, Any], *, arm: str
) -> None:
    """Refuse target ids that fall outside the declared scoring alphabet."""

    size = int(alphabet["size"])
    if size < 1:
        raise ValueError(f"{arm}: scoring-target alphabet size must be positive")
    if hasattr(target_ids, "size"):
        if int(target_ids.size) == 0:
            return
        lowest = int(target_ids.min())
        highest = int(target_ids.max())
    else:
        values = [int(value) for value in target_ids]
        if not values:
            return
        lowest = min(values)
        highest = max(values)
    if lowest < 0 or highest >= size:
        raise ValueError(
            f"{arm}: target token id outside the scoring-target alphabet of "
            f"{size} (source {alphabet['source']}); range [{lowest}, {highest}]"
        )


def output_logit_width(arm: Arm) -> dict[str, Any]:
    """Live output-column count of a loaded checkpoint.

    This is the model's real interface width, not the scoring-target alphabet.
    Truncation-curve memory guards and similar checks must use this, never 32
    on a staged ProGen2 rung. The live head is preferred; config.vocab_size and
    then config.vocab_size_lm_head are fallbacks. Tokenizer length is not a
    substitute, and nothing here crops or renormalises logits.
    """

    name = arm.name
    model = arm.model
    head = getattr(model, "lm_head", None)
    if head is not None:
        out_features = getattr(head, "out_features", None)
        if out_features is not None:
            size = int(out_features)
            source = "lm_head.out_features"
        else:
            weight = getattr(head, "weight", None)
            shape = getattr(weight, "shape", None) if weight is not None else None
            if shape is None or len(shape) < 1:
                size = 0
                source = ""
            else:
                size = int(shape[0])
                source = "lm_head.weight.shape[0]"
        if size >= 1:
            return {"size": size, "source": source, "arm": name}
    config = getattr(model, "config", None)
    if config is not None:
        if hasattr(config, "vocab_size"):
            raw = config.vocab_size
            if raw is not None:
                size = int(raw)
                if size >= 1:
                    return {
                        "size": size,
                        "source": "config.vocab_size",
                        "arm": name,
                    }
        lm_head = getattr(config, "vocab_size_lm_head", None)
        if lm_head is not None:
            size = int(lm_head)
            if size >= 1:
                return {
                    "size": size,
                    "source": "config.vocab_size_lm_head",
                    "arm": name,
                }
    raise ValueError(
        f"{name}: cannot resolve the live logit width from the output head or "
        "config; tokenizer length is not a substitute"
    )


#: The four-rung protein scale ladder: one lineage, one 31-token residue
#: tokenizer, one UniRef90+BFD30 mixture, 151M -> 765M -> 2.78B -> 6.44B.
#:
#: :data:`PROTEIN_SCALE_CONTRAST` is its first two rungs and is what a *campaign*
#: may schedule, because the upper two are :data:`STAGED_ARMS` rather than panel
#: members. The ladder is declared separately because a measurement that needs
#: none of the panel's campaign obligations can run all four, and the set of
#: rungs it runs on is then a value rather than a list a reader has to reassemble.
#: The upper two now declare a scoring-target alphabet of 32, so a budget
#: measurement may score them through :func:`scoring_target_alphabet` without
#: reading ``config.vocab_size``. That is still not panel admission.
PROTEIN_SCALE_LADDER = (
    "progen2-small",
    "progen2-medium",
    "progen2-large",
    "progen2-xlarge",
)

#: The staged rungs of :data:`PROTEIN_SCALE_LADDER`. Opt-in scale stages may
#: name these and no other staged checkpoint.
STAGED_SCALE_ARMS = tuple(name for name in PROTEIN_SCALE_LADDER if name in STAGED_ARMS)

#: EXP-R2-225's second-stage checkpoints: the larger public models staged for the
#: independent second stage, in the prereg's own wave order.
#:
#: The campaign's joint wave -- ``galactica-6.7b`` and ``galactica-30b`` -- is
#: **not** here, and the comment above their would-be declaration says why: a
#: joint checkpoint is reached by path, and every stage that needs one already
#: does. This tuple is therefore this file's list, not the prereg's.
#:
#: A **separate** door from :data:`STAGED_SCALE_ARMS`, and separate on purpose.
#: That tuple is the two upper rungs of one ProGen2 lineage and is what
#: EXP-R2-224's first round is frozen on; the three stages that key on it -- stage
#: 01's ``--allow-staged-scale-arms``, stage 20's ``SCOREABLE_ARMS`` and stage
#: 29's staged-rung branch -- must keep refusing exactly what they refuse today,
#: which they do only if that tuple keeps meaning exactly what it meant. So this
#: campaign gets its own name and its own per-stage opt-in rather than widening
#: the first round's.
#:
#: Membership here is staging and nothing else. It does not qualify a
#: checkpoint, does not admit one to :data:`PANEL`, and above all does not grant
#: a capability beyond what the arm's ``capabilities`` field already states.
#: The two Qwen rungs keep the rotary family; ``rita-xl`` still declares an empty
#: set because no pad token can be established; ``proteinglm-7b-clm`` now declares
#: ``budget`` for the continuation serving path. That is not 7B qualification and
#: not an interpretability-family grant.
STAGED_SECOND_STAGE_ARMS = (
    "qwen2.5-7b",
    "qwen2.5-32b",
    "proteinglm-7b-clm",
    "rita-xl",
)

#: Budget-only candidate checkpoints. A **third** door, separate from
#: :data:`STAGED_SCALE_ARMS` (EXP-R2-224) and :data:`STAGED_SECOND_STAGE_ARMS`
#: (EXP-R2-225). Stage 01 admits these names only with ``--allow-candidate-arms``;
#: that flag admits only this tuple. Membership is not panel admission, not a
#: capability beyond ``budget``, and not an experiment ADMITTED digest.
STAGED_CANDIDATE_ARMS = (
    "qwen3-8b-base",
    "protgpt3-1.3b",
)

#: ProGen3 rungs for the stage-01 context-information door. **Not** in
#: :data:`STAGED_ARMS`: that table's keys are pinned to
#: ``panel_contract.STAGED_BUT_NOT_ADMITTED``, and putting a joint-ineligible
#: packed-MoE lineage there would either widen campaign membership or block the
#: megablocks loader behind AutoModel. Stage 01 admits these names only with
#: ``--allow-progen3-arms``. Membership is not panel admission. Hugging Face
#: eager MoE is random experts; the loader is :func:`src.capability.models.progen3.load_progen3`.
PROGEN3_ARMS: dict[str, ArmSpec] = {
    "progen3-112m": ArmSpec(
        name="progen3-112m",
        path=env_path("PROGEN3_DIR", MODEL_BASE_DIR / "progen3-112m"),
        path_variable="PROGEN3_DIR",
        modality="protein",
        n_layer=10,
        d_model=384,
        tokenisation="residue",
        input_format="raw",
        evaluation_cohort_source="swissprot",
        architecture="progen3",
        pretraining_corpus=PRETRAINING_UNDECLARED,
        capabilities=frozenset({"budget"}),
        scoring_target_alphabet_size=134,
    ),
    "progen3-3b": ArmSpec(
        name="progen3-3b",
        path=MODEL_ROOT / "progen3-3b",
        path_variable="MODEL_ROOT",
        modality="protein",
        n_layer=24,
        d_model=1280,
        tokenisation="residue",
        input_format="raw",
        evaluation_cohort_source="swissprot",
        architecture="progen3",
        pretraining_corpus="profluent_protein_atlas_v1",
        capabilities=frozenset({"budget"}),
        scoring_target_alphabet_size=134,
    ),
}
STAGED_PROGEN3_ARMS = tuple(PROGEN3_ARMS)


def _check_progen3_arms() -> None:
    """ProGen3 is a fourth door, not a silent widening of STAGED_ARMS."""

    if list(STAGED_PROGEN3_ARMS) != list(PROGEN3_ARMS):
        raise AssertionError("STAGED_PROGEN3_ARMS must be the keys of PROGEN3_ARMS")
    overlap = sorted(set(PROGEN3_ARMS) & (set(PANEL) | set(STAGED_ARMS)))
    if overlap:
        raise AssertionError(
            f"{overlap} cannot be both ProGen3 door members and PANEL/STAGED_ARMS"
        )
    for name, spec in PROGEN3_ARMS.items():
        if spec.name != name:
            raise AssertionError(f"progen3 arm {name!r} declares the name {spec.name!r}")
        if spec.architecture != "progen3":
            raise AssertionError(f"{name} must declare architecture progen3")
        if spec.scoring_target_alphabet_size is None:
            raise AssertionError(f"{name} must declare scoring_target_alphabet_size")
        if "budget" not in spec.capabilities:
            raise AssertionError(f"{name} must declare the budget capability")


_check_progen3_arms()


def _check_second_stage_arms() -> None:
    """The staged doors name real staged checkpoints and never each other.

    Three doors, checked at import: scale, second-stage, and candidate. The
    whole value of separate opt-ins is that EXP-R2-224's first round cannot be
    widened by an edit to a later list, and an overlap would widen it silently.
    A name behind no door would be unreachable by omission.
    """

    doors = (
        ("STAGED_SCALE_ARMS", STAGED_SCALE_ARMS),
        ("STAGED_SECOND_STAGE_ARMS", STAGED_SECOND_STAGE_ARMS),
        ("STAGED_CANDIDATE_ARMS", STAGED_CANDIDATE_ARMS),
    )
    for label, names in doors:
        unknown = [name for name in names if name not in STAGED_ARMS]
        if unknown:
            raise AssertionError(
                f"{label} names {unknown}, which are not declared in STAGED_ARMS"
            )
        if len(set(names)) != len(names):
            raise AssertionError(f"{label} repeats an arm")
    both = sorted(set(STAGED_SECOND_STAGE_ARMS) & set(STAGED_SCALE_ARMS))
    if both:
        raise AssertionError(
            f"{both} are in both staged doors; EXP-R2-224's first round and "
            "EXP-R2-225's second stage are separate opt-ins and an arm in both "
            "would widen the first round without an edit to it"
        )
    candidate_scale = sorted(set(STAGED_CANDIDATE_ARMS) & set(STAGED_SCALE_ARMS))
    if candidate_scale:
        raise AssertionError(
            f"{candidate_scale} are in both STAGED_CANDIDATE_ARMS and "
            "STAGED_SCALE_ARMS; the candidate door is not a widening of the "
            "first round"
        )
    candidate_second = sorted(
        set(STAGED_CANDIDATE_ARMS) & set(STAGED_SECOND_STAGE_ARMS)
    )
    if candidate_second:
        raise AssertionError(
            f"{candidate_second} are in both STAGED_CANDIDATE_ARMS and "
            "STAGED_SECOND_STAGE_ARMS; the candidate door is not a widening of "
            "the second stage"
        )
    covered = (
        set(STAGED_SCALE_ARMS) | set(STAGED_SECOND_STAGE_ARMS) | set(STAGED_CANDIDATE_ARMS)
    )
    undecided = sorted(name for name in STAGED_ARMS if name not in covered)
    if undecided:
        raise AssertionError(
            f"staged checkpoints {undecided} are behind neither opt-in door; a "
            "staged arm must be reachable through a named door or be removed, "
            "never be unreachable by omission"
        )


_check_second_stage_arms()

#: Source labels recorded beside every scoring-target alphabet.
SCORING_TARGET_ALPHABET_CONFIG = "config.vocab_size"
SCORING_TARGET_ALPHABET_DECLARED = "arm_spec.scoring_target_alphabet_size"

#: Identification of a UniRef90+BFD30 pretraining mixture searched only through
#: the staged UniRef50 snapshot. The bound runs in the model-favouring direction:
#: an under-supported LOOKUP inflates MODEL - LOOKUP / weakens a retrieval
#: exclusion. One note, reused by the retrieval-bound and designed-referent
#: stages so the two cannot drift.
UNIREF90_BFD30_INCOMPLETE_SEARCH = (
    "UniRef90 and BFD30 are not fully searched. The staged snapshot is UniRef50, "
    "whose representatives are members of both; LOOKUP and the homology certificate "
    "therefore under-count retrievable support. That inflates MODEL - LOOKUP and "
    "weakens a corpus-disjoint exclusion, so both bounds run in the model-favouring "
    "direction and are not identified measurements"
)


def _check_scale_ladder() -> None:
    """A scale ladder must vary scale and nothing else it declares.

    Checked at import for the reason :func:`_check_corpus_contrast` is: the whole
    value of these four rungs is that architecture, tokenisation, rendering,
    evaluation corpus and pretraining mixture are held fixed, so a rung that
    quietly differed in one of them would turn a scale curve into a curve of
    something else with nothing raising.
    """

    specs = [arm_spec(name) for name in PROTEIN_SCALE_LADDER]
    held = (
        "modality",
        "tokenisation",
        "input_format",
        "evaluation_cohort_source",
        "architecture",
        "pretraining_corpus",
    )
    for key in held:
        values = {getattr(spec, key) for spec in specs}
        if len(values) != 1:
            raise AssertionError(
                f"the protein scale ladder does not hold {key} fixed: {sorted(values)}"
            )
    widths = [spec.d_model for spec in specs]
    if widths != sorted(widths) or len(set(widths)) != len(widths):
        raise AssertionError(f"the protein scale ladder is not ordered by width: {widths}")


_check_scale_ladder()

#: The exactly matched pair. Depth, width and vocabulary size are identical, so
#: any difference between these two is a modality difference, not an
#: architecture difference.
MATCHED_PAIR = ("gpt2-large", "protgpt2")


def _check_corpus_contrast(pair: tuple[str, str]) -> None:
    """A corpus contrast must hold everything but the pretraining corpus fixed.

    The two pairs below are the only evidence in the panel for how much of a
    cross-modality difference is training data rather than modality, and the
    claim rests entirely on what is held fixed. That used to be a comment; it is
    now checked at import, because the field it depends on
    (:attr:`ArmSpec.pretraining_corpus`) is new and a future edit that leaves it
    at :data:`PRETRAINING_UNDECLARED` would turn the contrast into a comparison
    of two arms with no declared difference at all.
    """

    left, right = (PANEL[name] for name in pair)
    held = ("modality", "n_layer", "d_model", "tokenisation", "architecture")
    differing = [key for key in held if getattr(left, key) != getattr(right, key)]
    if differing:
        raise AssertionError(f"corpus contrast {pair} does not hold {differing} fixed")
    if PRETRAINING_UNDECLARED in (left.pretraining_corpus, right.pretraining_corpus):
        raise AssertionError(
            f"corpus contrast {pair} has an undeclared pretraining corpus, so the "
            "quantity it varies is not recorded"
        )
    if left.pretraining_corpus == right.pretraining_corpus:
        raise AssertionError(
            f"corpus contrast {pair} declares one pretraining corpus "
            f"{left.pretraining_corpus!r} for both arms, so it varies nothing"
        )


for _pair in (MATCHED_DATA_CONTRAST, TEXT_DATA_CONTRAST):
    _check_corpus_contrast(_pair)
del _pair


def _check_architecture_contrast() -> None:
    """The arms that answer "is this just a GPT-2 property?" must not be GPT-2.

    Checked rather than commented for the same reason as the corpus contrasts,
    and with more at stake: this is the declaration behind the audit's §5.05(d)
    argument that the pathway-budget separation "survives replacing GPT-2-large
    with a Qwen2 and a Llama decoder", and it is what killed the QK/OV finding
    when it turned out to be a GPT-2-lineage property. An edit that left a
    GPT-2-architecture arm in this tuple would leave that argument stated and
    unsupported, with nothing raising.
    """

    for name in TEXT_ARCHITECTURE_CONTRAST:
        spec = PANEL[name]
        if spec.modality != "text":
            raise AssertionError(
                f"{name} is in TEXT_ARCHITECTURE_CONTRAST but is not a text arm"
            )
        if spec.architecture == "gpt2":
            raise AssertionError(
                f"{name} declares the gpt2 architecture, so it cannot witness that a "
                "finding survives leaving the GPT-2 lineage"
            )


_check_architecture_contrast()


@dataclass
class Arm:
    """A loaded panel member, with the contracts a measurement can rely on.

    ``attn_implementation`` is the attention kernel the checkpoint was actually
    loaded with, read back from the built model rather than from the request, so
    that a build which ignored or overrode the request is visible.

    ``target_token_shuffle`` is off on every arm any loader builds and is set only
    by a caller that has asked for the E3 negative control by name. When it is
    set, :func:`tokenize_batch` permutes each record's scored target positions
    before the ids leave for the model; see :class:`.scoring.TargetTokenShuffle`
    for what that control does and does not establish. It sits on the arm rather
    than on the cohort because the permutation is defined over *tokens*, and the
    token grid is what this arm's tokenizer, its truncation and its padding
    produce -- a cohort holds strings, and no string-level shuffle preserves a
    BPE arm's target multiset.

    ``strict_load`` is the per-key Transformers loading-info count block that a
    ``strict=True`` load was verified clean against, and is ``None`` on every arm
    no strict load produced. It rides on the arm because the evidence is created
    at load time and is needed wherever a record is written: a caller that means
    to state "this checkpoint loaded strictly and cleanly" reads the counts the
    check actually returned, instead of asserting a fact it cannot see from an
    already-constructed arm.

    ``serving_provenance`` is optional load-time evidence for a path that derives
    source before serving. ProteinGLM fills it from this call's ``derived`` and
    ``max_length`` payload; every other architecture leaves it ``None``. It is
    not a scientific PASS, not a registry, and not a later re-derivation from
    disk.
    """

    spec: ArmSpec
    model: torch.nn.Module
    tokenizer: object
    device: str
    dtype: str
    attn_implementation: str | None = None
    target_token_shuffle: TargetTokenShuffle | None = None
    strict_load: dict[str, int] | None = None
    serving_provenance: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def modality(self) -> str:
        return self.spec.modality

    @property
    def n_layer(self) -> int:
        return self.spec.n_layer

    @property
    def d_model(self) -> int:
        return self.spec.d_model

    def supports(self, capability: str) -> bool:
        if capability not in CAPABILITIES:
            raise ValueError(f"unknown capability {capability!r}; known: {sorted(CAPABILITIES)}")
        return capability in self.spec.capabilities

    def scoring_target_alphabet(self) -> dict[str, Any]:
        """The scoring-target alphabet for this loaded arm, with its source."""

        config = getattr(self.model, "config", None)
        return scoring_target_alphabet(self.spec, config)

    def require(self, capability: str) -> None:
        """Refuse a measurement this arm cannot enter commensurably."""
        if not self.supports(capability):
            raise ValueError(
                f"{self.name} ({self.spec.architecture}) does not support the "
                f"{capability!r} measurement family; its declared capabilities are "
                f"{sorted(self.spec.capabilities)}. Producing a number here would "
                "not be commensurate with the rest of the panel."
            )

    def require_eager_attention(self, measurement: str) -> None:
        """Refuse a measurement that reads or overrides attention patterns on a
        non-eager kernel.

        The fused kernels never materialise the pattern, so they return ``None``
        for the weights and ignore a per-head additive mask. Every caller that
        needs a pattern already raises when it gets ``None`` back, but that check
        fires deep inside a run and only for the read path; an *override* -- a
        per-head knockout mask, a frozen pattern -- would be accepted by the
        fused kernel's signature and quietly not applied. The contract is
        therefore stated at the top of the measurement, against the
        implementation the model was actually built with.
        """

        if self.attn_implementation is None:
            raise ValueError(
                f"{self.name}: {measurement} needs a declared attention "
                "implementation; load the arm with attn_implementation='eager'"
            )
        if self.attn_implementation != "eager":
            raise ValueError(
                f"{self.name}: {measurement} reads or overrides attention patterns, "
                f"which the {self.attn_implementation!r} kernel does not materialise; "
                "load the arm with attn_implementation='eager'"
            )

    def blocks(self) -> torch.nn.ModuleList:
        """The transformer block list, resolved per architecture.

        Resolution is explicit rather than duck-typed: a panel member whose
        block list cannot be named is a panel change that must be declared, not
        guessed at.
        """
        architecture = self.spec.architecture
        if architecture in ("gpt2", "progen"):
            if not hasattr(self.model, "transformer") or not hasattr(self.model.transformer, "h"):
                raise TypeError(f"{self.name}: declared {architecture} but no transformer.h")
            return self.model.transformer.h
        if architecture == "t5_decoder":
            if not hasattr(self.model, "decoder") or not hasattr(self.model.decoder, "block"):
                raise TypeError(f"{self.name}: declared t5_decoder but no decoder.block")
            return self.model.decoder.block
        if architecture == "opt":
            # ``OPTForCausalLM`` keeps an ``OPTModel`` at ``.model`` and the
            # decoder one level below it, so the block list is two attributes
            # further down than either of the branches above: GPT-2's
            # ``.transformer.h`` and a rotary decoder's ``.model.layers`` both
            # stop one level short of it. Reaching ``.model`` and stopping would
            # find no ``layers`` at all, which is why the walk is spelled out.
            inner = getattr(getattr(self.model, "model", None), "decoder", None)
            if inner is None or not hasattr(inner, "layers"):
                raise TypeError(f"{self.name}: declared opt but no model.decoder.layers")
            return inner.layers
        if architecture in _ROTARY_DECODERS:
            # The causal-LM wrapper holds the bare decoder at ``.model``, one
            # level below where GPT-2 keeps ``.transformer``, and the block list
            # at ``.layers`` rather than ``.h``.
            inner = getattr(self.model, "model", None)
            if inner is None or not hasattr(inner, "layers"):
                raise TypeError(f"{self.name}: declared {architecture} but no model.layers")
            return inner.layers
        if architecture == "reformer":
            return self.model.reformer.encoder.layers
        raise TypeError(f"{self.name}: unsupported architecture {architecture!r}")

    def mlp(self, layer: int) -> torch.nn.Module:
        self.require("pathway")
        if self.spec.architecture not in _DECOMPOSABLE:
            raise TypeError(
                f"{self.name}: sublayer decomposition is not defined for "
                f"{self.spec.architecture!r}"
            )
        return self.blocks()[layer].mlp

    def _resolve_d_mlp(self) -> tuple[int, str]:
        """``d_mlp`` and the declaration it came from."""

        declared = mlp_neuron_declaration(self.spec.architecture)
        for attribute in declared.width_attributes:
            value = getattr(self.model.config, attribute, None)
            if value is not None:
                return int(value), f"config.{attribute}"
        if declared.implicit_width_multiple is None:
            raise TypeError(
                f"{self.name}: none of {list(declared.width_attributes)} is set on the "
                f"config and {self.spec.architecture!r} declares no implicit width, so "
                "the MLP hidden width is unknown"
            )
        multiple = int(declared.implicit_width_multiple)
        return multiple * int(self.d_model), (
            f"{multiple} x d_model ({'/'.join(declared.width_attributes)} unset)"
        )

    @property
    def d_mlp(self) -> int:
        """Width of the MLP hidden layer: the number of neurons in one block."""

        return self._resolve_d_mlp()[0]

    def mlp_down_projection(self, layer: int) -> torch.nn.Module:
        """The module whose **input** is this layer's MLP hidden activation.

        The neuron tensor is that input, and hooking the projection rather than
        the nonlinearity is what makes it definitionally so. Resolved by walking
        :attr:`MlpNeuronTensor.down_projection` from :meth:`mlp`, so an
        architecture nobody declared raises instead of being searched.
        """

        self.require("pathway")
        declared = mlp_neuron_declaration(self.spec.architecture)
        module: object = self.mlp(layer)
        for step in declared.down_projection:
            if not hasattr(module, step):
                raise TypeError(
                    f"{self.name}: declared {self.spec.architecture} but its layer-"
                    f"{layer} MLP has no {step} on the path {declared.down_projection}"
                )
            module = getattr(module, step)
        return module  # type: ignore[return-value]

    def mlp_neuron_facts(self) -> dict[str, object]:
        """What the artefact must carry about the tensor a neuron basis reads.

        Resolved once, from the declaration and the loaded config, so that the
        stage records the path it actually hooked and the width it expects rather
        than a sentence about them. An activation the declaration does not name
        raises here: its output bound is unknown, so the check that the hooked
        tensor is post-nonlinearity could not be applied to it.
        """

        declared = mlp_neuron_declaration(self.spec.architecture)
        width, source = self._resolve_d_mlp()
        activation = getattr(self.model.config, declared.activation_attribute, None)
        if activation not in declared.activation_lower_bound:
            raise TypeError(
                f"{self.name}: {declared.activation_attribute}={activation!r} is not a "
                f"declared nonlinearity for {self.spec.architecture!r} "
                f"({sorted(declared.activation_lower_bound)}); its output bound is "
                "unknown, so the hooked tensor cannot be verified as post-nonlinearity"
            )
        return {
            "architecture": self.spec.architecture,
            "tensor": "input of " + ".".join(("mlp", *declared.down_projection)),
            "declared_width": int(width),
            "width_source": source,
            "d_model": int(self.d_model),
            "activation": str(activation),
            "activation_lower_bound": float(declared.activation_lower_bound[activation]),
        }

    def _resolve_attention(self, layer: int) -> torch.nn.Module:
        """Walk :data:`_ATTENTION_PATH` from the block to the attention submodule.

        Searching a block for the first plausible attribute would resolve a newly
        admitted architecture silently, which is the one failure this panel
        cannot afford: an arm that reaches a measurement through an attribute
        nobody declared produces a number that looks like every other number in
        the table. One walker for both accessors, so the two cannot disagree
        about which module an arm's attention *is* while disagreeing, as they
        must, about what may be measured on it.
        """

        architecture = self.spec.architecture
        path = _ATTENTION_PATH.get(architecture)
        if path is None:
            raise TypeError(
                f"{self.name}: no attention submodule is declared for {architecture!r}; "
                f"declared: {sorted(_ATTENTION_PATH)}"
            )
        module: object = self.blocks()[layer]
        for step in path:
            if isinstance(step, int):
                try:
                    module = module[step]  # type: ignore[index]
                except (IndexError, TypeError) as error:
                    raise TypeError(
                        f"{self.name}: declared {architecture} but block {layer} has no "
                        f"entry {step} on the path {path}"
                    ) from error
                continue
            if not hasattr(module, step):
                raise TypeError(
                    f"{self.name}: declared {architecture} but block {layer} has no "
                    f"{step} on the path {path}"
                )
            module = getattr(module, step)
        return module  # type: ignore[return-value]

    def attention_pattern_module(self, layer: int) -> torch.nn.Module:
        """The attention submodule whose forward computes the pattern.

        Reading or overriding an attention pattern needs the module and nothing
        else. It does *not* need the block's sublayers to be commensurate with a
        standard causal decoder, which is what :meth:`attention` additionally
        asserts, and conflating the two kept ByGPT5 out of the prediction-addressed
        census on a decomposition ground the census never relies on.

        Gated on ``circuits`` rather than ``pathway``: everything that reads a
        pattern through this accessor computes a *per-head* statistic, which is
        the family ``circuits`` declares, while ``pathway`` declares that whole
        sublayer outputs are commensurate -- a claim this accessor's callers
        neither make nor need.
        """

        self.require("circuits")
        return self._resolve_attention(layer)

    def attention(self, layer: int) -> torch.nn.Module:
        """The attention submodule, as one term of a commensurate sublayer split.

        Same module as :meth:`attention_pattern_module`, stronger claim: callers
        of this accessor read or ablate it *as the attention sublayer* of a
        decomposition whose other term is the MLP, so an architecture whose
        sublayers are not those objects is refused even though its attention
        module can be named.
        """
        self.require("pathway")
        architecture = self.spec.architecture
        if architecture not in _DECOMPOSABLE:
            raise TypeError(
                f"{self.name}: sublayer decomposition is not defined for {architecture!r}"
            )
        return self._resolve_attention(layer)


#: Keys Transformers reports when ``output_loading_info=True``. A strict load
#: refuses any of them being non-empty, and refuses an API return that is not
#: exactly ``(model, {these four keys...})``.
LOADING_INFO_KEYS: tuple[str, ...] = (
    "missing_keys",
    "unexpected_keys",
    "mismatched_keys",
    "error_msgs",
)


def unpack_pretrained_loading_info(loaded: Any) -> tuple[Any, dict[str, Any]]:
    """Require Transformers' ``output_loading_info=True`` return shape.

    A model-only return, a tuple of the wrong length, or a non-dict second
    element is an API-shape failure. Silent degradation would treat a partial
    or unreported load as clean.
    """

    if not isinstance(loaded, tuple) or len(loaded) != 2:
        kind = type(loaded).__name__
        extra = f" of length {len(loaded)}" if isinstance(loaded, tuple) else ""
        raise TypeError(
            "strict load expected Transformers to return (model, loading_info); "
            f"got {kind}{extra}"
        )
    model, info = loaded
    if not isinstance(info, dict):
        raise TypeError(
            "strict load expected loading_info to be a dict; "
            f"got {type(info).__name__}"
        )
    missing = [key for key in LOADING_INFO_KEYS if key not in info]
    if missing:
        raise TypeError(
            "strict load expected loading_info keys "
            f"{list(LOADING_INFO_KEYS)}; missing {missing}"
        )
    return model, info


def require_clean_loading_info(
    info: Mapping[str, Any], *, arm: str
) -> dict[str, int]:
    """Refuse any non-empty Transformers loading-info list. Return zero counts."""

    counts: dict[str, int] = {}
    nonempty: list[str] = []
    for key in LOADING_INFO_KEYS:
        values = info[key]
        try:
            count = len(values)
        except TypeError as exc:
            raise TypeError(
                f"{arm}: loading_info[{key!r}] is not a sequence"
            ) from exc
        counts[key] = int(count)
        if count:
            nonempty.append(f"{key}={count}")
    if nonempty:
        raise ValueError(
            f"{arm}: strict weight load refused non-empty loading_info "
            f"({', '.join(nonempty)})"
        )
    return counts


def adopt_config_declared_pad_token(tokenizer: Any, config: Any, *, arm: str) -> None:
    """Give a tokenizer the pad token its **own config** declares, or leave it none.

    Several public checkpoints ship a tokenizer that carries the pad token in its
    vocabulary and declares it nowhere the tokenizer object reads. Galactica is
    the measured case in this repository: all four staged rungs ship
    ``special_tokens_map.json`` == ``{}`` and a ``tokenizer_config.json`` naming
    no special token, so the loaded ``PreTrainedTokenizerFast`` reports
    ``pad_token``, ``eos_token``, ``bos_token`` and ``unk_token`` all ``None`` --
    while the same directory's ``tokenizer.json`` defines ``<pad>`` at id 1 as a
    special token and ``config.json`` declares ``pad_token_id`` 1.
    :func:`tokenize_batch` then refuses every batch, one-record batches included.

    **The config is the checkpoint author's declaration, and it is checked rather
    than trusted.** The id is resolved back through the tokenizer's own
    vocabulary and the assignment is read back, so this adopts a token that
    exists in the vocabulary this tokenizer will encode with, not an integer that
    happened to be in a JSON file. Three outcomes, and only one of them proceeds:

    * the config declares no ``pad_token_id`` -- nothing to establish, so nothing
      is invented. The tokenizer is left with no pad token and the refusal stays
      exactly where it is today, in :func:`tokenize_batch`. ``rita-xl`` is this
      case: its config declares no pad id and its ``eos_token_id`` 50256 is not
      an id its 26-symbol vocabulary contains.
    * the declared id does not map to a token in this vocabulary, or the
      assignment does not read back at that id -- the checkpoint contradicts
      itself and the load stops here, with both ids named. Padding with an id
      nothing maps to would put a token the model never saw into every short row
      of every batch.
    * the declared id names a token -- it is adopted, and
      ``tokenizer.pad_token_id`` is that id.

    Why padding with the declared id is safe once established: :func:`tokenize_batch`
    right-pads, attention is causal, and the returned mask excludes every padding
    position from both the attention and the scored targets. What must not happen
    is padding with a *wrong* id, and the read-back is what rules that out.
    """

    declared = getattr(config, "pad_token_id", None)
    if declared is None:
        return
    declared = int(declared)
    token = tokenizer.convert_ids_to_tokens(declared)
    if not isinstance(token, str):
        raise ValueError(
            f"{arm}: config declares pad_token_id {declared}, which this "
            "tokenizer's vocabulary does not map to a token. A checkpoint whose "
            "config and tokenizer disagree about the padding symbol cannot be "
            "padded: the id would be encoded into every short row of every batch "
            "as a token the model never saw"
        )
    tokenizer.pad_token = token
    resolved = tokenizer.pad_token_id
    if resolved is None or int(resolved) != declared:
        raise ValueError(
            f"{arm}: config declares pad_token_id {declared} ({token!r}), but "
            f"setting it read back as {resolved!r}. The tokenizer's vocabulary "
            "and its config do not agree on this symbol's id, so no padding id "
            "is established for it"
        )


#: Config attributes that declare transformer depth, in the order consulted.
#: Three spellings appear among the checkpoints this module loads: ``n_layer``
#: (GPT-2, ProGen2), ``num_hidden_layers`` (Llama, Qwen2, OPT) and ``num_layers``
#: (RITA, ProteinGLM).
_DEPTH_ATTRIBUTES = ("n_layer", "num_hidden_layers", "num_layers")

#: Config attributes that declare residual width, in the order consulted. Four
#: spellings: ``n_embd`` (GPT-2, ProGen2-medium), ``hidden_size`` (Llama, Qwen2,
#: OPT, ProteinGLM), ``embed_dim`` (ProGen2-xlarge) and ``d_model`` (RITA).
_WIDTH_ATTRIBUTES = ("n_embd", "hidden_size", "embed_dim", "d_model")


def _first_declared(config: Any, attributes: tuple[str, ...], *, quantity: str) -> int:
    for attribute in attributes:
        value = getattr(config, attribute, None)
        if value is not None:
            return int(value)
    raise AttributeError(
        f"this config declares no {quantity}: none of {list(attributes)} is set "
        "on it. A checkpoint whose shape cannot be read must stop the load, "
        "because the alternative is to skip the shape check that is the only "
        "thing standing between a mis-declared arm and a plausible-looking number"
    )


def config_shape(config: Any) -> tuple[int, int]:
    """Depth and width of a checkpoint, from whichever key its config spells them in.

    One declared fallback order per quantity, resolved here rather than by a
    per-checkpoint exception at the loader. This is what makes admitting a new
    architecture a matter of extending a declaration -- and it is a declaration
    rather than a search, because the order is fixed, the candidates are named in
    the refusal, and a config that spells depth or width in none of them raises
    instead of resolving to whichever attribute happened to exist on it.

    ``config.vocab_size`` is deliberately absent from both orders and from every
    other shape question here: two ProGen2 checkpoints in this repository spell
    the output width differently, so depth and width -- which every checkpoint
    declares and which identify a rung of a scale ladder -- are what the loader
    verifies. See :func:`scoring_target_alphabet` for the separate question of
    what a scored target's support is.
    """

    return (
        _first_declared(config, _DEPTH_ATTRIBUTES, quantity="depth"),
        _first_declared(config, _WIDTH_ATTRIBUTES, quantity="width"),
    )


def require_declared_shape(spec: ArmSpec, config: Any) -> tuple[int, int]:
    """Refuse a config whose depth/width is not the arm's declaration.

    This is the check that keeps a mis-labelled checkpoint from producing a
    plausible-looking number. The ordinary loader runs it on the config before
    any weights load. The ProteinGLM path cannot share that AutoConfig pre-load,
    so it runs the same function on the config that was actually loaded, and
    also on the directory config before weights when that file is present.
    ProteinGLM is not exempt.
    """

    n_layer, d_model = config_shape(config)
    if (n_layer, d_model) != (spec.n_layer, spec.d_model):
        raise ValueError(
            f"{spec.name}: declared {spec.n_layer}L/{spec.d_model}d, loaded {n_layer}L/{d_model}d"
        )
    return n_layer, d_model


#: Config attributes that declare the position budget, in the order consulted.
#: Four spellings among the checkpoints this module reaches: ``n_positions``
#: (GPT-2, ProGen2), ``max_position_embeddings`` (Llama, Qwen2, OPT, ProGen3),
#: ``max_seq_len`` (RITA) and ``seq_length`` (ProteinGLM).
_CONTEXT_ATTRIBUTES = (
    "n_positions",
    "max_position_embeddings",
    "max_seq_len",
    "seq_length",
)


def config_context_length(config: Any) -> int:
    """How many positions a checkpoint declares, from whichever key spells it.

    :func:`config_shape`'s discipline, applied to the one other config quantity a
    stage derives behaviour from. The order is a declaration rather than a
    search: it is fixed, every candidate is named in the refusal, and a config
    that spells the budget in none of them raises **before any weight loads**
    rather than resolving to whichever attribute happened to exist. That last
    property is the point of resolving it here. The two fitness stages used to
    read ``n_positions or max_position_embeddings`` inline, which raises
    ``AttributeError`` on ``rita-xl`` (``max_seq_len``) and on
    ``proteinglm-7b-clm`` (``seq_length``) -- but only after the checkpoint has
    been loaded onto a card, because the config is read off the loaded model.

    **A declared budget is a ceiling, not a scoring window, and the two are not
    interchangeable.** The protein arms here declare 1024 or 2048 and the panel's
    own evaluation windows sit below that, so the distinction has never bitten;
    ``qwen2.5-7b`` and ``qwen2.5-32b`` declare **131072**, and ProGen3 declares
    65536. A caller that turns this number into a length must therefore cap it at
    the window its own measurement is declared on. Today three call sites choose a
    length from it -- ``retrieval_bound.py`` and ``designed_referent.py``,
    which use it as the token budget a rendered variant may not exceed and as
    ``tokenize_batch``'s ``max_len``, and ``epistasis_coupling.py``, which uses
    it the same way -- and none of them admits a Qwen rung: stages 20 and 29 are
    protein-only and stage 28 indexes :data:`PANEL` directly. The cap is owed the
    moment one does.
    """

    return _first_declared(config, _CONTEXT_ATTRIBUTES, quantity="context length")


def load_arm_spec(
    spec: ArmSpec,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    *,
    strict: bool = False,
) -> Arm:
    """Load a declared checkpoint and verify its declared shape and inference dtype.

    The loader takes a *declaration* rather than a name because two kinds of
    checkpoint reach it: a :data:`PANEL` member through :func:`load_arm`, and a
    :data:`STAGED_ARMS` non-member through :func:`arm_spec` at a call site that
    means to admit one. Splitting the name lookup from the load is what keeps the
    panel door panel-only while leaving one implementation of the load itself, so
    a staged checkpoint gets the same shape and dtype verification a panel arm
    does (Appendix B rule 12).

    ``config.vocab_size`` is deliberately not consulted. Two ProGen2 checkpoints
    in this repository spell the output width differently -- ``progen2-large``
    declares 51200 against a 31-token tokenizer and ``progen2-xlarge`` declares no
    ``vocab_size`` at all -- so the shape check reads depth and width, which every
    checkpoint here declares and which identify a rung of a scale ladder.

    ``strict=False`` is the historical default and does not request loading info.
    ``strict=True`` asks Transformers for ``output_loading_info`` and refuses any
    missing, unexpected, mismatched, or error-reported key. A return that is not
    exactly ``(model, loading_info)`` fails rather than being treated as clean.
    The verified counts are recorded on the returned arm as ``strict_load``, so a
    later caller can tell a strictly loaded checkpoint from one that was not; a
    non-strict load leaves that field ``None`` rather than a clean-looking zero.
    """

    name = spec.name
    if spec.architecture == "progen3":
        raise ValueError(
            f"{name}: architecture progen3 cannot be loaded through load_arm_spec; "
            "HF eager MoE leaves random experts. Use src.capability.models.progen3.load_progen3"
        )
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported inference dtype {dtype!r}")
    if spec.architecture == "proteinglm":
        if attn_implementation is not None:
            raise ValueError(
                f"{name}: attn_implementation is inert on ProteinGLM; refusing "
                f"{attn_implementation!r} rather than recording a contract the "
                "checkpoint does not honour"
            )
        if dtype != "float32":
            raise ValueError(
                f"ProteinGLM budget serving is FP32-only; refused dtype {dtype!r}"
            )
        path = require_input_path(spec.path, _MODEL_PATH_VARIABLES)
        serving_config = _proteinglm.load_serving_config(path)
        require_declared_shape(spec, serving_config)
        _proteinglm.require_serving_config(serving_config)
        loaded = _proteinglm.load_pretrained(
            path,
            device=device,
            dtype=dtype,
            strict=strict,
        )
        require_declared_shape(spec, loaded["config"])
        require_declared_shape(spec, loaded["model"].config)
        return Arm(
            spec=spec,
            model=loaded["model"],
            tokenizer=loaded["tokenizer"],
            device=device,
            dtype=dtype,
            attn_implementation=None,
            strict_load=loaded["strict_load"],
            serving_provenance={
                "derived": loaded["derived"],
                "max_length": loaded["max_length"],
            },
        )
    path = str(require_input_path(spec.path, _MODEL_PATH_VARIABLES))
    trust_remote_code = spec.architecture not in _BUILTIN_CAUSAL_LM_ARCHITECTURES

    config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    require_declared_shape(spec, config)

    extra: dict[str, object] = {}
    if attn_implementation is not None:
        # sdpa returns None for attention weights and cannot be intercepted;
        # anything that reads or overrides patterns must ask for eager.
        extra["attn_implementation"] = attn_implementation
    load_kwargs: dict[str, object] = {
        # ``torch_dtype`` rather than ``dtype``: the H200 pod runs transformers
        # 4.52.4, where ``dtype`` is not a recognised loading argument and would
        # be swallowed as a config keyword, leaving a float32 model. ``dtype``
        # is the newer spelling and 4.57.3 warns that ``torch_dtype`` is
        # deprecated, but it is the only spelling both versions honour, and the
        # observed-dtype check below is what actually enforces the outcome.
        "torch_dtype": _DTYPES[dtype],
        "trust_remote_code": trust_remote_code,
        "device_map": {"": device},
        **extra,
    }
    strict_load: dict[str, int] | None = None
    if strict:
        loaded = AutoModelForCausalLM.from_pretrained(
            path, output_loading_info=True, **load_kwargs
        )
        model, info = unpack_pretrained_loading_info(loaded)
        strict_load = require_clean_loading_info(info, arm=name)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
    model.eval()

    observed = sorted(
        {str(p.dtype).removeprefix("torch.") for p in model.parameters() if p.is_floating_point()}
    )
    if observed != [dtype]:
        raise ValueError(f"{name}: declared dtype {dtype}, observed {observed}")

    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        adopt_config_declared_pad_token(tokenizer, config, arm=name)
    # Read back rather than echo the request. A remote-code architecture that
    # never consults ``attn_implementation`` would otherwise be recorded as
    # eager on the strength of having been asked, and ``require_eager_attention``
    # would then vouch for a contract nothing enforced.
    resolved = getattr(model.config, "_attn_implementation", None)
    return Arm(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        attn_implementation=None if resolved is None else str(resolved),
        strict_load=strict_load,
    )


def load_arm(
    name: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
) -> Arm:
    """Load a panel member by name, refusing anything outside :data:`PANEL`.

    A staged non-member is not reachable here on purpose: admitting one by name
    would mean every stage that takes a free-text arm could schedule a checkpoint
    the panel deliberately excluded. Those callers resolve
    :func:`arm_spec` explicitly and load through :func:`load_arm_spec`.
    """

    if name not in PANEL:
        raise KeyError(f"unknown arm {name!r}; panel is {sorted(PANEL)}")
    return load_arm_spec(
        PANEL[name],
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )


# ------------------------------------------------------------------- cohorts


#: The FASTA corpora this package reads, and the variable that relocates each.
#: :func:`iter_fasta` is the single door to both, and several callers reach it
#: without going through :func:`protein_cohort`, so the existence check belongs
#: here rather than at each call site.
_FASTA_VARIABLE: dict[Path, str] = {
    SWISSPROT_FASTA: "SWISSPROT_FASTA",
    ZYMCTRL_FASTA: "ZYMCTRL_FASTA",
}


def iter_fasta(path: Path):
    path = Path(path)
    require_input_path(path, _FASTA_VARIABLE.get(path, "the TRANSFER_* variable naming it"))
    opener = gzip.open if str(path).endswith(".gz") else open
    header, chunks = None, []
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:].strip(), []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def sampling_record(
    *,
    seed: int | None,
    skip: int,
    requested: int,
    eligible: int | None,
    corpus: str,
) -> dict:
    """How a cohort was drawn, as a value that travels with the cohort.

    Recorded on every cohort this module builds. Which records were drawn is the
    single most expensive thing this programme has got wrong -- three separate
    incidents, one worth 1.01 nats -- and every one of them was invisible
    afterwards because the artefact said what the numbers were and not where the
    records came from. ``mode`` is therefore mandatory in the record, and the
    file-order mode carries its own hazard text.
    """

    if requested < 1:
        raise ValueError("a cohort must request at least one record")
    if skip < 0:
        raise ValueError("skip must be non-negative")
    mode = "file_order" if seed is None else "seeded_permutation"
    if mode not in SAMPLING_MODES:  # pragma: no cover - guards a future third mode
        raise AssertionError(
            f"sampling mode {mode!r} is not in SAMPLING_MODES {SAMPLING_MODES}; a "
            "mode that is not declared has no recorded hazard text and no reader "
            "knows how to interpret it"
        )
    record: dict = {
        "mode": mode,
        "seed": None if seed is None else int(seed),
        "skip": int(skip),
        "requested": int(requested),
        "corpus": corpus,
        "eligible_records": None if eligible is None else int(eligible),
    }
    if mode == "file_order":
        record["hazard"] = FILE_ORDER_HAZARD
    return record


@dataclass
class Cohort:
    """A frozen evaluation cohort, identified by the hash of its contents."""

    name: str
    kind: str
    records: list[str]
    min_symbols: int
    max_symbols: int
    metadata: dict = field(default_factory=dict)

    @property
    def digest(self) -> str:
        """Content hash: the records themselves, and nothing else.

        Deliberately unchanged, so that a digest quoted in a frozen artefact
        still identifies the same content. It does *not* separate two cohorts
        that hold the same records under different metadata -- an exact-repeat
        and an approximate-repeat cohort can coincide on records while being
        different measurements -- which is what :attr:`provenance_digest` is for.
        """

        payload = json.dumps(
            {
                "kind": self.kind,
                "min": self.min_symbols,
                "max": self.max_symbols,
                "records": self.records,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def provenance_digest(self) -> str:
        """Content hash extended with everything that decided the content.

        Two cohorts agreeing on :attr:`digest` can still be different objects:
        the nested exact and approximate repeat cohorts can hold identical
        records under different criteria, an EC-labelled draw carries labels a
        plain draw does not, and a seeded draw and a file-order draw that happen
        to coincide are not the same evidence. This digest separates them.
        Metadata that is not JSON-serialisable is represented by its repr rather
        than dropped, so nothing silently falls out of the hash.
        """

        def canonical(value: object) -> object:
            try:
                json.dumps(value)
            except TypeError:
                return repr(value)
            return value

        payload = json.dumps(
            {
                "content": self.digest,
                "metadata": {key: canonical(value) for key, value in sorted(self.metadata.items())},
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def sampling(self) -> dict:
        """The draw that produced this cohort, or an explicit "not recorded"."""

        recorded = self.metadata.get("sampling")
        if isinstance(recorded, dict):
            return dict(recorded)
        return {
            "mode": "unrecorded",
            "hazard": (
                "this cohort was constructed without a sampling record, so whether "
                "its records are a seeded sample or the head of a file is not "
                "knowable from the artefact"
            ),
        }

    def __len__(self) -> int:
        return len(self.records)

    def input_strings(self, arm: Arm) -> list[str]:
        """Render the cohort in the arm's native input format."""
        if self.kind == "text":
            if arm.modality != "text":
                raise ValueError(f"{arm.name}: text cohort given to a protein arm")
            return list(self.records)
        if arm.modality != "protein":
            raise ValueError(f"{arm.name}: protein cohort given to a text arm")
        fmt = arm.spec.input_format
        if fmt == "raw":
            return list(self.records)
        if fmt == "fasta_wrapped":
            # ProtGPT2 was pretrained on FASTA-formatted UniRef50: sequences are
            # hard-wrapped at 60 residues and separated by the end-of-text token,
            # and its BPE merges were learned over exactly that byte stream.
            # Feeding one unwrapped line is off-distribution and costs 1.42
            # nats/token, measured on 80 Swiss-Prot sequences of 600-2000
            # residues: 8.046 raw versus 6.652 wrapped versus 6.623 with the
            # end-of-text prefix. Getting this wrong makes the model look
            # untrained and drove a spurious modality effect.
            eot = arm.tokenizer.eos_token
            if eot is None:
                raise ValueError(f"{arm.name}: tokenizer has no end-of-text token")
            return [
                eot + "\n" + "\n".join(s[i : i + 60] for i in range(0, len(s), 60))
                for s in self.records
            ]
        if fmt == "n_to_c_control":
            return [N_TO_C_MARKER + s for s in self.records]
        if fmt == INPUT_FORMAT_EOS_BOUNDED_SEQ:
            return [eos_bounded_rendering(s) for s in self.records]
        if fmt == INPUT_FORMAT_BOS_DIRECTION_SEQ:
            return [bos_direction_rendering(arm, s) for s in self.records]
        if fmt == "ec_conditioned":
            labels = self.metadata.get("ec_labels")
            if labels is None or len(labels) != len(self.records):
                raise ValueError(
                    f"{arm.name} requires EC labels; cohort {self.name!r} has none"
                )
            return [
                f"{ec}<sep>{CONDITIONING_START}{seq}{CONDITIONING_END}"
                for ec, seq in zip(labels, self.records)
            ]
        if fmt == INPUT_FORMAT_GMASK_SOP_EOS:
            return [_proteinglm.render_budget_sequence(sequence) for sequence in self.records]
        raise ValueError(f"unsupported input format {fmt!r}")


#: The markers ``Cohort.input_strings`` wraps a conditioned arm's content in.
#: Declared beside the rendering that emits them: a measurement module that needs
#: to *find* them must resolve these, not a second pair spelled by hand. Three
#: modules used to spell them independently, and the third did not check the
#: tokenizer's unknown-token id, so a tokenizer without them would have returned
#: a valid-looking id for a token it does not have.
CONDITIONING_START = "<start>"
CONDITIONING_END = "<end>"

#: The generation-direction marker an ``n_to_c_control`` rendering prefixes to its
#: content. Declared beside the rendering that emits it for the reason the pair
#: above is, and with one hazard they do not have: ProGen2 declares only
#: ``<|pad|>``, ``<|bos|>`` and ``<|eos|>`` as special tokens, so this marker is an
#: **ordinary vocabulary entry** (id 3 on every released rung). A measurement that
#: must keep it out of a content span cannot find it through ``all_special_ids``
#: and has to resolve it from here, through :func:`rendering_marker_ids`.
N_TO_C_MARKER = "1"

#: The direction tokens an ``INPUT_FORMAT_BOS_DIRECTION_SEQ`` rendering places
#: after the BOS, as the checkpoint's card names them: ``"1"`` for N-to-C and
#: ``"2"`` for C-to-N. Both are **ordinary vocabulary entries** rather than
#: special tokens -- measured on the staged checkpoint they are ids 4 and 5, and
#: ``all_special_ids`` is ``[0, 1, 2, 33]`` -- so a measurement that must keep them
#: out of a content span resolves them through :func:`rendering_marker_ids`, not
#: through the tokenizer's special ids.
#:
#: :data:`BOS_DIRECTION_N_TO_C` is :data:`N_TO_C_MARKER` rather than a second copy
#: of ``"1"``: the string that selects N-to-C is one fact this panel holds for
#: both checkpoints that declare a direction, and spelling it twice would let the
#: two drift apart. It is also the direction this panel *scores*; its sibling is
#: the directional control, because a model trained in both directions need not
#: prefer the one a call site happened to pick.
BOS_DIRECTION_N_TO_C = N_TO_C_MARKER
BOS_DIRECTION_C_TO_N = "2"


#: The document-boundary token an ``eos_bounded_seq`` rendering prefixes: the
#: token RITA's training stream separates documents with, and the one the
#: checkpoint's tokenizer carries at id 2. Spelled once, because the rendering
#: that prefixes it and
#: :func:`src.capability.models.rita_fitness._native_special_ids`, which verifies its id,
#: must resolve the same token. Its sibling is the tokenizer's terminal
#: ``<EOS>``, which the post-processor appends and which carries this same id:
#: the boundary and the terminator are one token, so an id set cannot separate
#: them, and every position whose target is it is a marker rather than content.
EOS_BOUNDED_BOUNDARY = "<EOS>"


def eos_bounded_rendering(sequence: str) -> str:
    """``<EOS>`` + sequence, RITA-xl's native document rendering.

    The single declaration of that format; :meth:`Cohort.input_strings` renders
    through it and :func:`src.capability.models.rita_fitness.native_encode_for_budget`
    refuses any text that does not start with the boundary it places, so neither
    the gate nor the ProteinGym door can score the bare residue string this arm
    used to be rendered as. The boundary token needs no ``Arm``: it is a fixed
    declared token rather than a tokenizer attribute, and its id is verified
    against the native one when the checkpoint loads.
    """

    return EOS_BOUNDED_BOUNDARY + str(sequence)


def bos_direction_rendering(
    arm: Arm, sequence: str, direction: str = BOS_DIRECTION_N_TO_C
) -> str:
    """``<|bos|>`` + direction token + sequence, ProtGPT3-1.3B's native rendering.

    The single declaration of that format; :meth:`Cohort.input_strings` renders
    through it so that no stage spells the prefix itself, and its inverse is
    :func:`rendering_marker_ids`, which returns the two ids it puts in front of
    the content.

    ``direction`` takes either declared direction token. Passing
    :data:`BOS_DIRECTION_C_TO_N` is the directional control -- the same sequence
    under the opposite marker, which for a model trained in both directions is a
    different estimate of the same quantity and not a relabelling of this one.
    """

    if direction not in (BOS_DIRECTION_N_TO_C, BOS_DIRECTION_C_TO_N):
        raise ValueError(
            f"{arm.name}: {direction!r} is not a declared direction token; the "
            f"declared pair is {(BOS_DIRECTION_N_TO_C, BOS_DIRECTION_C_TO_N)}"
        )
    bos = arm.tokenizer.bos_token
    if bos is None:
        raise ValueError(
            f"{arm.name}: its rendering places a BOS before the direction token, "
            "but the tokenizer declares no BOS token"
        )
    return f"{bos}{direction}{sequence}"


def conditioning_boundary_ids(
    arm: Arm, *, ec_conditioning: str = "native"
) -> tuple[int | None, int | None]:
    """Token ids delimiting the scored content of a conditioned rendering.

    ``(None, None)`` when the rendering carries no conditioning prompt -- either
    the arm's input format has none, or ``ec_conditioning="unconditioned"``
    deliberately removed it -- so the result can be passed straight to
    :func:`src.capability.core.scoring.sequence_target_mask` beside the rule that
    :func:`src.capability.core.scoring.target_rule` selects for the same two inputs.

    Raises rather than returning a plausible id when the tokenizer does not carry
    the markers. The conditioning prompt is the span a measurement must *not*
    score -- EXP-R2-034 prices ZymCTRL's EC tag at 1.73 nats of leak -- so a
    silently wrong boundary id lands directly on the quantity being measured.
    """

    if arm.spec.input_format != "ec_conditioned" or ec_conditioning == "unconditioned":
        return None, None
    unknown = arm.tokenizer.unk_token_id
    ids: list[int] = []
    for token in (CONDITIONING_START, CONDITIONING_END):
        resolved = arm.tokenizer.convert_tokens_to_ids(token)
        if resolved is None or resolved == unknown:
            raise ValueError(
                f"{arm.name}: tokenizer has no {token!r} id, but its input format is "
                "ec_conditioned, so the span that must not be scored cannot be located"
            )
        ids.append(int(resolved))
    return ids[0], ids[1]


def rendering_marker_ids(arm: Arm) -> tuple[int, ...]:
    """Token ids of the non-content markers this arm's rendering prefixes.

    The counterpart of :func:`conditioning_boundary_ids` for the renderings that
    carry no conditioning prompt, and the one place a measurement may learn which
    positions of such a rendering are not content.
    :meth:`Cohort.input_strings` prefixes ``fasta_wrapped`` with the tokenizer's
    end-of-text token and ``n_to_c_control`` with :data:`N_TO_C_MARKER`; ``raw``
    prefixes nothing; ``bos_direction_seq`` prefixes two tokens, the tokenizer's
    BOS and then :data:`BOS_DIRECTION_N_TO_C`; ``eos_bounded_seq`` prefixes
    :data:`EOS_BOUNDED_BOUNDARY`, whose id is also the terminal token its
    tokenizer appends. The scored span therefore starts at the first residue, not
    at the second token: the BOS is never a target, and the direction token is
    excluded by the marker ids this function returns, not by the position it
    happens to occupy. ``eos_bounded_seq`` is the one rendering whose terminator
    carries a marker id, so an exclusion stated by id removes that target as well:
    what is content there is the sequence's residues, which is the statement
    :data:`src.capability.models.progen3.NON_RESIDUE_TOKENS` makes for ProGen3's terminus
    tokens and the one :func:`src.capability.core.budget.scored_tokens` applies.

    **A tokenizer's special ids do not cover this, and assuming they did was a
    defect.** ProGen2 declares only ``<|pad|>``, ``<|bos|>`` and ``<|eos|>``
    special, so ``all_special_ids`` never returns its direction marker and a
    content mask built from those ids alone keeps position 0 of every ProGen2
    record -- while ProGen3, whose
    :data:`src.capability.models.progen3.NON_RESIDUE_TOKENS` names ``"1"`` and ``"2"``
    explicitly, drops the equivalent position. That one position per record is not
    a rounding error where it lands: on ``progen2-small`` at bfloat16 over 32
    Swiss-Prot records of 64-246 residues, keeping it displaces
    ``perturbation_sensitivity.py``'s fully-ablated endpoint -- the denominator
    of every recovery ratio that stage reports -- by 0.68 in relative norm at
    layer 1 and 0.59 at layer 2, and moves the layer-1 norm anchor epsilon is
    scaled against from 1.1267 to 1.4993.

    The two branches resolve differently because the two markers are different
    kinds of token. An end-of-text token is a special token and the tokenizer
    declares its id, so it is read; a direction marker is an ordinary vocabulary
    entry that has to be looked up and can come back as the unknown id, which is
    refused rather than returned. The unknown check is not applied to the
    end-of-text branch on purpose: ProtGPT2's ``unk_token_id`` *is* its
    ``eos_token_id``, so an unknown check there would refuse a healthy arm.

    ``ec_conditioned`` raises. Its conditioning prompt is a span rather than a
    prefix -- the EC digits are ordinary vocabulary entries carrying no marker id
    at all -- so removing marker ids would leave seven digit positions inside the
    content span, and :func:`conditioning_boundary_ids` is what locates it.
    """

    fmt = arm.spec.input_format
    if fmt == "raw":
        return ()
    if fmt == "fasta_wrapped":
        resolved = arm.tokenizer.eos_token_id
        if resolved is None:
            raise ValueError(
                f"{arm.name}: tokenizer declares no end-of-text id, but its rendering "
                "prefixes the end-of-text token, so the prefix cannot be located"
            )
        return (int(resolved),)
    if fmt == "n_to_c_control":
        resolved = arm.tokenizer.convert_tokens_to_ids(N_TO_C_MARKER)
        if resolved is None or resolved == arm.tokenizer.unk_token_id:
            raise ValueError(
                f"{arm.name}: tokenizer has no {N_TO_C_MARKER!r} id, but its input "
                "format is n_to_c_control, so the direction marker its rendering "
                "prefixes cannot be kept out of the content span"
            )
        return (int(resolved),)
    if fmt == INPUT_FORMAT_EOS_BOUNDED_SEQ:
        resolved = arm.tokenizer.convert_tokens_to_ids(EOS_BOUNDED_BOUNDARY)
        if resolved is None or resolved == arm.tokenizer.unk_token_id:
            raise ValueError(
                f"{arm.name}: tokenizer has no {EOS_BOUNDED_BOUNDARY!r} id, but its "
                "input format is eos_bounded_seq, so the document boundary its "
                "rendering prefixes cannot be kept out of the content span"
            )
        return (int(resolved),)
    if fmt == INPUT_FORMAT_BOS_DIRECTION_SEQ:
        bos = arm.tokenizer.bos_token_id
        if bos is None:
            raise ValueError(
                f"{arm.name}: its rendering prefixes a BOS token, but the tokenizer "
                "declares no BOS id, so that position cannot be located"
            )
        resolved = arm.tokenizer.convert_tokens_to_ids(BOS_DIRECTION_N_TO_C)
        if resolved is None or resolved == arm.tokenizer.unk_token_id:
            raise ValueError(
                f"{arm.name}: tokenizer has no {BOS_DIRECTION_N_TO_C!r} id, but its "
                "input format is bos_direction_seq, so the direction token its "
                "rendering places after the BOS cannot be kept out of the content "
                "span"
            )
        return (int(bos), int(resolved))
    if fmt == "ec_conditioned":
        raise ValueError(
            f"{arm.name} renders a conditioning prompt rather than a marker prefix: "
            "its EC digits carry no marker id, so a set of ids cannot describe the "
            "span that is not content. Use conditioning_boundary_ids"
        )
    if fmt == INPUT_FORMAT_GMASK_SOP_EOS:
        return _proteinglm.PREFIX_IDS
    raise ValueError(f"unsupported input format {fmt!r}")


def selected_positions(
    eligible: int, *, n: int, skip: int, seed: int | None, label: str
) -> list[int]:
    """Which eligible records a draw selects, given a mode.

    ``seed is None`` reproduces the historical file-order draw exactly:
    positions ``skip .. skip + n``. With a seed the corpus is permuted first and
    the same half-open window is taken from the permutation, which makes two
    draws at the same seed and different ``skip`` genuinely disjoint -- the
    skip-offset sensitivity Appendix B rule 1 asks for is only a sensitivity if
    the offsets do not overlap.

    Returned in ascending corpus order so that a second pass over the corpus can
    collect them in one sweep; the *identity* of the selected set is what the
    seed decides, not the order they end up in.
    """

    if n < 1:
        raise ValueError("a cohort must request at least one record")
    if skip < 0:
        raise ValueError("skip must be non-negative")
    if eligible < skip + n:
        raise RuntimeError(
            f"cohort {label!r}: {eligible} eligible records cannot supply {n} "
            f"after a skip of {skip}"
        )
    if seed is None:
        return list(range(skip, skip + n))
    import numpy as np

    order = np.random.default_rng(seed).permutation(eligible)
    return sorted(int(index) for index in order[skip : skip + n])


def _eligible_protein_records(min_len: int, max_len: int, *, with_ec: bool):
    """Every corpus entry passing the length and alphabet filter, in file order.

    One generator serves both the counting pass and the collecting pass, so the
    two cannot disagree about what "eligible" means -- which is the way a
    two-pass sampler silently selects the wrong records.
    """

    allowed = set(AA20)
    if with_ec:
        for _, body in iter_fasta(ZYMCTRL_FASTA):
            if "<start>" not in body or "<end>" not in body:
                continue
            sequence = body.split("<start>")[1].split("<end>")[0]
            if not (min_len <= len(sequence) <= max_len) or not set(sequence) <= allowed:
                continue
            yield sequence, body.split("<sep>")[0]
    else:
        for _, sequence in iter_fasta(SWISSPROT_FASTA):
            if not (min_len <= len(sequence) <= max_len) or not set(sequence) <= allowed:
                continue
            yield sequence, None


def _eligible_text_documents(min_chars: int):
    """Every screening-subset document at or above ``min_chars``, in shard order."""

    import pyarrow.parquet as pq

    require_input_path(OPENWEBTEXT, "OPENWEBTEXT_DIR")
    shards = sorted(OPENWEBTEXT.glob("*.parquet"))
    if not shards:
        raise RuntimeError(f"no parquet shards under {OPENWEBTEXT}")
    for shard in shards:
        for value in pq.read_table(shard, columns=["text"]).column("text"):
            document = value.as_py()
            if document is None or len(document) < min_chars:
                continue
            yield document


#: The corpus each declared source name streams from, and the variable that
#: relocates it. Read by :func:`iter_corpus_records`, which serves the stages
#: that need more records than a frozen cohort holds -- a transcoder trainer
#: consumes millions, which the cohort constructors cannot supply because they
#: count the whole corpus before selecting.
#:
#: The names are the ones :attr:`ArmSpec.evaluation_cohort_source` already uses,
#: plus ``uniref50``, which no arm evaluates on and ProGen3's transcoders are
#: trained on.
CORPUS_SOURCES: dict[str, tuple[Path, str]] = {
    "uniref50": (UNIREF50_FASTA, "UNIREF50_FASTA"),
    "swissprot": (SWISSPROT_FASTA, "SWISSPROT_FASTA"),
    "zymctrl_ec": (ZYMCTRL_FASTA, "ZYMCTRL_FASTA"),
    "openwebtext": (OPENWEBTEXT, "OPENWEBTEXT_DIR"),
}


def corpus_location(source: str, *, path: Path | None = None) -> Path:
    """Where one corpus source lives, checked to exist before anything loads a model.

    ``path`` overrides the declared location for ``uniref50`` only, which is the
    one source a stage flag has ever relocated. The others refuse an override
    rather than accept and ignore one: ``swissprot`` is read through the same
    eligibility generator :func:`protein_cohort` uses and ``openwebtext`` through
    the parquet reader, and neither consults a caller's path -- so an accepted
    override would silently stream the declared corpus under the name of another.
    """

    if source not in CORPUS_SOURCES:
        raise KeyError(f"unknown corpus source {source!r}; declared: {sorted(CORPUS_SOURCES)}")
    declared, variable = CORPUS_SOURCES[source]
    if path is None:
        return require_input_path(declared, variable)
    if source != "uniref50":
        raise ValueError(
            f"the {source!r} stream is read through this module's own reader and "
            f"cannot be relocated by a caller's path; set {variable} instead"
        )
    return require_input_path(Path(path), variable)


def iter_corpus_records(
    source: str,
    *,
    min_symbols: int,
    max_symbols: int | None = None,
    path: Path | None = None,
) -> Iterator[tuple[str, str | None]]:
    """Every eligible record of one corpus, in file order, in its own symbol unit.

    Symbols are residues for a protein corpus and characters for a text one, so
    one band argument means the thing the corpus is made of rather than a token
    count that would differ between arms (Appendix B rule 21's shape, one level
    down: a band declared in tokens is not the same band on two tokenisers).

    Each record is yielded as ``(record, conditioning_label)``. The label is
    ``None`` for every corpus whose arms take an unconditional rendering and is
    the EC number for ``zymctrl_ec``, because :meth:`Cohort.input_strings` cannot
    render an ``ec_conditioned`` arm without it. Carrying the pair rather than
    the bare record is what keeps the conditioned arm on the panel's one
    rendering declaration instead of a second copy inside a stage (Appendix B
    rule 12).

    **Eligibility is not uniform across the four, and the difference is
    deliberate.** ``swissprot`` and ``zymctrl_ec`` apply the canonical-alphabet
    filter :func:`protein_cohort` applies -- and ``zymctrl_ec`` additionally
    requires the ``<start>``/``<end>`` markers its records carry -- so a stage
    streaming either and a stage drawing a cohort from it see one population.
    ``uniref50`` applies the length band only, because that is what the ProGen3
    transcoder campaign streamed and its published runs must stay reproducible.
    ``openwebtext`` takes a floor and no ceiling, because the text path truncates
    at tokenisation rather than discarding long documents -- which is what
    :func:`text_cohort` does, and a ceiling here would select a different
    population from the cohort a replacement is later scored on.

    File order is the *stream* order, not a sampling decision: the callers shuffle
    it. The hazard Appendix B rule 1 names is theirs to answer, and
    ``train_transcoder.py`` answers it with a seeded block shuffle whose block
    size it records.
    """

    location = corpus_location(source, path=path)
    if source == "openwebtext":
        if max_symbols is not None:
            raise ValueError(
                "the openwebtext stream takes no upper bound: a text record is "
                "truncated at tokenisation rather than filtered out, and a "
                "character ceiling here would select a different population from "
                "the one text_cohort draws"
            )
        return ((document, None) for document in _eligible_text_documents(min_symbols))
    if max_symbols is None:
        raise ValueError(f"the {source!r} stream needs an upper residue bound")
    if source in ("swissprot", "zymctrl_ec"):
        return _eligible_protein_records(
            min_symbols, max_symbols, with_ec=source == "zymctrl_ec"
        )
    return (
        (sequence, None)
        for _, sequence in iter_fasta(location)
        if min_symbols <= len(sequence) <= max_symbols
    )


def protein_cohort(
    n: int,
    min_len: int,
    max_len: int,
    *,
    skip: int = 0,
    name: str = "swissprot",
    with_ec: bool = False,
    seed: int | None = None,
) -> Cohort:
    """Canonical-alphabet Swiss-Prot sequences, drawn under a declared mode.

    ``with_ec`` draws from the EC-labelled source so that one cohort can serve
    both the unconditional arms and ZymCTRL, which needs its conditioning tag.

    ``seed`` selects the draw. **Pass one.** Swiss-Prot and the EC-labelled
    corpus are both grouped by family, so the first ``n`` eligible entries are a
    set of near-clonal homologues rather than a sample: they are unusually
    predictable, which shrinks the context information every share is divided by,
    and reading past them has moved a headline figure by 1.01 nats. ``seed=None``
    keeps the historical file-order draw, because several frozen artefacts were
    produced with it and must stay reproducible; it is recorded as
    ``sampling.mode == "file_order"`` with its hazard attached, so no artefact
    built on it can be read without seeing which draw produced it.

    Under a seed the whole corpus is counted first and the draw is a window of a
    seeded permutation, so ``skip`` produces a genuinely disjoint second sample
    of the same corpus rather than a different prefix of the same file.
    """

    corpus = "ec_labelled_swissprot" if with_ec else "plain_swissprot"
    if seed is None:
        records: list[str] = []
        labels: list[str] = []
        eligible: int | None = None
        seen = 0
        for sequence, label in _eligible_protein_records(min_len, max_len, with_ec=with_ec):
            seen += 1
            if seen <= skip:
                continue
            records.append(sequence)
            if label is not None:
                labels.append(label)
            if len(records) >= n:
                break
        if len(records) < n:
            raise RuntimeError(f"cohort {name!r}: only {len(records)}/{n} eligible sequences")
    else:
        eligible = sum(
            1 for _ in _eligible_protein_records(min_len, max_len, with_ec=with_ec)
        )
        wanted = set(selected_positions(eligible, n=n, skip=skip, seed=seed, label=name))
        records = []
        labels = []
        for position, (sequence, label) in enumerate(
            _eligible_protein_records(min_len, max_len, with_ec=with_ec)
        ):
            if position not in wanted:
                continue
            records.append(sequence)
            if label is not None:
                labels.append(label)
        if len(records) != n:
            raise RuntimeError(
                f"cohort {name!r}: the corpus changed between the counting and the "
                f"collecting pass ({len(records)} of {n} selected records found)"
            )
    metadata: dict = {
        "sampling": sampling_record(
            seed=seed, skip=skip, requested=n, eligible=eligible, corpus=corpus
        )
    }
    if with_ec:
        metadata["ec_labels"] = labels
    return Cohort(name, "protein", records, min_len, max_len, metadata)


def eligible_protein_population(
    min_len: int,
    max_len: int,
    *,
    with_ec: bool = False,
) -> tuple[list[str], list[str] | None, str]:
    """Every eligible protein record in file order, with optional EC labels.

    Positions in the returned list are the source positions a group-disjoint
    fill records. The generator is consumed once; callers that need a seeded
    permutation apply it to these lists rather than opening the FASTA again.
    """

    records: list[str] = []
    labels: list[str] = []
    for sequence, label in _eligible_protein_records(min_len, max_len, with_ec=with_ec):
        records.append(sequence)
        if label is not None:
            labels.append(label)
    corpus = "ec_labelled_swissprot" if with_ec else "plain_swissprot"
    if with_ec and len(labels) != len(records):
        raise RuntimeError(
            f"EC-labelled population {corpus!r} is missing labels on "
            f"{len(records) - len(labels)} eligible records"
        )
    return records, (labels if with_ec else None), corpus


def group_disjoint_sampling_record(
    *,
    seed: int,
    requested: int,
    eligible: int,
    corpus: str,
    algorithm: str,
    algorithm_version: str,
    containment_threshold: float,
    shingle_length: int,
    slot: str,
    source_positions: Sequence[int],
) -> dict[str, Any]:
    """How a group-disjoint slot was filled, as a value that travels with it."""

    if requested < 1:
        raise ValueError("a cohort must request at least one record")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    if GROUP_DISJOINT_SAMPLING_MODE not in SAMPLING_MODES:
        raise AssertionError(
            f"sampling mode {GROUP_DISJOINT_SAMPLING_MODE!r} is not in "
            f"SAMPLING_MODES {SAMPLING_MODES}"
        )
    return {
        "mode": GROUP_DISJOINT_SAMPLING_MODE,
        "seed": int(seed),
        "requested": int(requested),
        "corpus": corpus,
        "eligible_records": int(eligible),
        "algorithm": str(algorithm),
        "algorithm_version": str(algorithm_version),
        "containment_threshold": float(containment_threshold),
        "shingle_length": int(shingle_length),
        "slot": str(slot),
        "source_positions": [int(position) for position in source_positions],
    }


def protein_cohort_from_records(
    records: Sequence[str],
    min_len: int,
    max_len: int,
    *,
    name: str,
    sampling: Mapping[str, Any],
    labels: Sequence[str] | None = None,
) -> Cohort:
    """Rebuild a protein cohort from already-chosen records and sampling."""

    metadata: dict = {"sampling": dict(sampling)}
    if labels is not None:
        if len(labels) != len(records):
            raise ValueError("EC labels must align with the frozen records")
        metadata["ec_labels"] = list(labels)
    return Cohort(name, "protein", list(records), min_len, max_len, metadata)


def text_cohort(
    n: int,
    min_chars: int = 800,
    *,
    skip: int = 0,
    name: str = "openwebtext",
    seed: int | None = None,
) -> Cohort:
    """Documents from the frozen OpenWebText screening subset.

    ``seed`` has the same meaning as in :func:`protein_cohort`. The text control
    is drawn the same way as the protein cohorts on purpose: a control drawn
    under a different sampling rule from the arm it controls is not a control.
    """

    if seed is None:
        records: list[str] = []
        eligible: int | None = None
        seen = 0
        for document in _eligible_text_documents(min_chars):
            seen += 1
            if seen <= skip:
                continue
            records.append(document)
            if len(records) >= n:
                break
        if len(records) < n:
            raise RuntimeError(f"cohort {name!r}: only {len(records)}/{n} documents")
    else:
        eligible = sum(1 for _ in _eligible_text_documents(min_chars))
        wanted = set(selected_positions(eligible, n=n, skip=skip, seed=seed, label=name))
        records = [
            document
            for position, document in enumerate(_eligible_text_documents(min_chars))
            if position in wanted
        ]
        if len(records) != n:
            raise RuntimeError(
                f"cohort {name!r}: the corpus changed between the counting and the "
                f"collecting pass ({len(records)} of {n} selected documents found)"
            )
    metadata = {
        "sampling": sampling_record(
            seed=seed,
            skip=skip,
            requested=n,
            eligible=eligible,
            corpus="openwebtext_screen",
        )
    }
    return Cohort(name, "text", records, min_chars, 0, metadata)


def target_shuffle_for(arm: Arm, *, seed: int) -> TargetTokenShuffle:
    """The E3 token-shuffle control declaration for one arm, at one seed.

    Resolves the masking rule and the boundary ids from the same two
    declarations :func:`src.capability.core.budget.scored_tokens` resolves them from, so
    the control permutes exactly the span that arm is scored on and cannot drift
    from it. Attach the result to an arm with
    ``dataclasses.replace(arm, target_token_shuffle=...)`` and the ids that reach
    the model are shuffled; leave it unattached and nothing changes.

    An EC-conditioned arm gets ``between_boundaries`` with its ``<start>`` and
    ``<end>`` ids, so its EC tag stays in context, in place, and out of the
    permuted span.

    Every other rendering's marker ids travel with it too. The permuted set is
    recomputed inside :meth:`TargetTokenShuffle.apply`, so a control that resolved
    the rule but not the markers would permute a position the measurement does not
    score -- which for a rendering whose prefix is two tokens means shuffling the
    direction token into a residue's place.
    """

    start_id, end_id = conditioning_boundary_ids(arm)
    return TargetTokenShuffle(
        seed=int(seed),
        rule=target_rule(arm.spec.input_format),
        start_token_id=start_id,
        end_token_id=end_id,
        marker_token_ids=() if start_id is not None else rendering_marker_ids(arm),
    )


def tokenize_batch(
    arm: Arm, texts: list[str], max_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded ids and a validity mask, truncated to ``max_len``.

    This is the single door from a rendered string to the token grid a model is
    scored on, which is why the E3 negative control is applied here: an arm
    carrying a :class:`.scoring.TargetTokenShuffle` has each record's scored
    target positions permuted *after* truncation and padding, so the control
    cannot change which tokens survive ``max_len`` and cannot touch a padding
    position. ``arm.target_token_shuffle`` is ``None`` on every arm a loader
    builds, so this branch is dead unless a caller asked for the control by name.
    """
    if not texts:
        raise ValueError(f"{arm.name}: cannot tokenise an empty batch")
    if max_len < 1:
        raise ValueError("max_len must be positive")
    if arm.spec.architecture == "progen3":
        raise ValueError(
            f"{arm.name}: ProGen3 must be tokenised by ProGen3BatchPreparer, not "
            "tokenize_batch; a generic tokenizer call drops sequence_ids"
        )
    if arm.spec.input_format == INPUT_FORMAT_GMASK_SOP_EOS:
        rows = [
            _proteinglm.encode_budget_text(arm.tokenizer, text, max_len=max_len)
            for text in texts
        ]
    else:
        rows = [arm.tokenizer(t, return_tensors=None)["input_ids"][:max_len] for t in texts]
    empty = [index for index, row in enumerate(rows) if not row]
    if empty:
        # A zero-token row would contribute a fully masked line to the batch and
        # therefore to every mean computed over it, without appearing anywhere
        # as a dropped record.
        raise ValueError(f"{arm.name}: rows {empty} of the batch tokenise to no tokens")
    width = max(len(r) for r in rows)
    pad = arm.tokenizer.pad_token_id
    if pad is None:
        raise ValueError(f"{arm.name}: tokenizer has no pad token")
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, : len(row)] = 1
    if arm.target_token_shuffle is not None:
        ids = arm.target_token_shuffle.apply(ids, mask)
    return ids, mask


def symbols_per_token(arm: Arm, texts: list[str], max_len: int) -> float:
    """Measured tokenizer expansion over exactly the scored window.

    Protein arms count residues; the text arm counts characters. Counting before
    truncation inflates the ratio for long sequences, so both are counted after.
    ProteinGLM budget serving encodes through :func:`proteinglm.encode_budget_text`
    so the continuation is not silently truncated and does not grow a tail EOS.
    """
    tokens = 0
    symbols = 0
    for text in texts:
        if arm.spec.architecture == "rita":
            from ..models.rita_fitness import native_encode_for_budget

            ids = native_encode_for_budget(arm.tokenizer, text)[:max_len]
        elif arm.spec.architecture == "progen3":
            from ..models.progen3 import require_progen3_handle

            batch = require_progen3_handle(arm).batch([text], reverse=False)
            ids = batch["input_ids"][0].detach().cpu().tolist()[:max_len]
        elif arm.spec.input_format == INPUT_FORMAT_GMASK_SOP_EOS:
            ids = _proteinglm.encode_budget_text(
                arm.tokenizer, text, max_len=max_len
            )
        else:
            ids = arm.tokenizer(text, return_tensors=None)["input_ids"][:max_len]
        decoded = arm.tokenizer.decode(ids)
        tokens += len(ids)
        symbols += (
            sum(1 for c in decoded if c in AA20) if arm.modality == "protein" else len(decoded)
        )
    if tokens == 0:
        raise RuntimeError(f"{arm.name}: empty cohort")
    return symbols / tokens
