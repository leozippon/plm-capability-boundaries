"""ProGen3, loaded so that it cannot come back silently random.

**Why this module exists.** The entry point everyone calls,
``ProGen3ForCausalLM.from_pretrained(path, moe_implementation="eager")``,
*succeeds* on the released checkpoints. It emits a "newly initialized" warning,
returns a model whose embeddings, attention and norms are loaded correctly, and
whose every expert and every router is freshly random -- because the released
weights are in **megablocks packing** and carry no key the eager
``SparseMoeBlock`` recognises. Nothing raises. The model generates. It scores
Swiss-Prot at **17.15 nats/token** against the **1.983** the same checkpoint
reaches once the experts are actually loaded (EXP: ``progen3_eager_probe``,
steps 08-09, one L20). A measurement taken on that object is a measurement of
noise wearing a checkpoint's name, and no exception separates the two.

:func:`load_progen3` is therefore the only supported way into these checkpoints
in this repository. It converts the packed tensors, loads them with
``strict=True``, and refuses on any missing or unexpected key.

**The conversion**, verified against megablocks 0.7.0's ``LearnedRouter`` and
``MemoryOptimizedGroupedGLU`` and against the eager block's own forward, with
``9216 = 8 experts x 1152`` on the 112M and ``30720 = 8 x 3840`` on the 3B:

===========================================  ===========================================
released (megablocks packing)                eager ``SparseMoeBlock``
===========================================  ===========================================
``...block_sparse_moe.experts.mlp.w1``       ``...experts.{e}.w1.weight``  split
``...block_sparse_moe.experts.mlp.v1``       ``...experts.{e}.w3.weight``  split
``...block_sparse_moe.experts.mlp.w2``       ``...experts.{e}.w2.weight``  split + transpose
``...block_sparse_moe.router.layer.weight``  ``...block_sparse_moe.gate.weight``  rename
``mlm_head.weight``                          dropped: ``ProGen3ForCausalLM`` has no MLM head
===========================================  ===========================================

**The mapping is load-bearing and the evidence says a wrong one is unmissable.**
Under the mapping above, on the 112M: UniRef50 2.588 (the paper reports ~2.50),
Swiss-Prot 1.983, residue-shuffled Swiss-Prot 2.940, uniform over 20 residues
2.996. Corrupting the mapping while keeping ``strict=True`` clean: ``w1``/``v1``
swapped 3.201, gate rows rolled by one expert 3.173. Both are *worse than
shuffled protein*, so a mapping error cannot hide inside a plausible number --
provided something looks. :func:`self_check` is the thing that looks.

**Two checkpoints, one packing, two bands.** 112M and 3B ship the same
megablocks packing under the same tensor names, so
:func:`convert_megablocks_state_dict` serves both unchanged. What differs is the
file layout -- the 3B arrives in two safetensors shards named by a
``model.safetensors.index.json`` -- the placement of the attention, which the
3B's ``fused_attention_norm`` moves under ``norm_attn_norm``, and above all the
*number*. A correctly loaded 3B scores far below a correctly loaded 112M, so a
single shared band would either reject the larger checkpoint or stop
discriminating on the smaller one. :data:`SELF_CHECK_REFERENCES` therefore
declares each checkpoint's own measured value and its own corruption controls,
and a config matching no declared entry is refused rather than gated against
somebody else's number.

**What this module deliberately does not do.** It imports the patched
third-party ``progen3`` package by path and nothing else from that tree; the
tree is CC BY-NC-ND, git-ignored, and must not be vendored. It needs neither
megablocks (the eager block replaces it) nor flash-attn (a pure-PyTorch fp32
RMSNorm replaces the fused kernel); both substitutions are local patches to the
ignored copy, not to this repository.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from ..core.arms import MODEL_BASE_DIR, MODEL_ROOT, REPO, env_path

#: The released ProGen3-112M checkpoint directory.
PROGEN3_CHECKPOINT = env_path("PROGEN3_DIR", MODEL_BASE_DIR / "progen3-112m")

#: Import root of the patched third-party ``progen3`` package. Resolved at call
#: time rather than at import, so this module can be imported -- and its pure
#: functions tested -- on a host that carries neither the tree nor the weights.
PROGEN3_SOURCE = Path(
    os.environ.get(
        "TRANSFER_PROGEN3_SRC",
        REPO / "external/code/ProGenMech/external/progen3/src",
    )
)

#: Where a packed expert tensor's rows go, keyed by the released tensor's name.
#: ``v1`` is the *up* projection and lands on ``w3``; ``w1`` is the gate. Getting
#: those two the wrong way round is the 3.201-nat control above.
EXPERT_TARGETS = {"w1": "w1", "v1": "w3", "w2": "w2"}

#: Released tensors with no counterpart in ``ProGen3ForCausalLM``.
DROPPED_KEYS = ("mlm_head.weight",)

#: Eight Swiss-Prot sequences, 85-138 residues, taken at a fixed stride from
#: ``uniprot_sprot.fasta.gz`` and frozen here as literals. Literals rather than a
#: cohort draw because the self-check has to work in an environment that has the
#: weights and no corpus, and because a self-check whose input can move is not a
#: check. Accessions, in order: Q8YYH3, B2TK01, Q9CQM1, Q8SPI2, A7ZV06, B7JXS8,
#: P12104, Q4QMV0.
SELF_CHECK_SEQUENCES: tuple[str, ...] = (
    "MPNSTPQSQLIRAHVFVTGRVQGVGFRYSTVDTASQLGLTGWVRNLPDGRVEAVFEGVRDIVEDMVRWCHAGPPAAVVQDVAVEYEEPEGLRGFEVKRLVK",
    "MADTFLLKIVTPDKDIFNGNIKRIFLKNSVGRLEILANHANMVTSTISSIVEFTDADGKDRKLFISKGIASIFNNEMTIFSESAEFSDNIDLNRAEKAKERAEKRLLEGNKYDKERAELALLRSIERINLKKMN",
    "MIGGNTTIISGAINASTEAPGLGTGGRAWPVLVGVVLGAVVLSILIALAAKCHLCRRYHASYRHRPLSSAGGGNRPPVGEDEDDDGFIEDNYIQPGAGEMETTGSRDHFSL",
    "MPVVTGRLRDPDINPCLSESDASTRCLDENNYDKERCSTYFLKYKNCRKFWHSIMMQRRRNGVKPCMPTAAERDEILRAMGKMPY",
    "MLDEKSSNTASVVVLCTAPDEATAQDLAAKVLAEKLAACATLIPGATSLYYWEGKLEQEYEVQMILKTTVSHQQALLECLKSHHPYQTPELLVLPVTHGDTDYLSWLNASLR",
    "MNDYTRTIRLSDTDAAGVVYFASLLSICHEAYEASLEASGIDLKSFFRDSEVVIPIVHAEIDFFRPLYSGDRIIITLTTLQLKDTEFEITYQVGLVAPQSSLIAKAKTRHVAINPQTRQRTPLSESLMQWLKSTENSE",
    "MAFDSTWKVDRSENYDKFMEKMGVNIVKRKLAAHDNLKLTITQEGNKFTVKESSTFRNIEVVFELGVTFNYNLADGTELRGTWSLEGNKLIGKFKRTDNGNELNTVREIIGDELVQTYVYEGVEAKRIFKKD",
    "MQALLFISYGAILGASLRWAIGLLFNPLFSSFAFGTLIANLLGCLIIGVLLGFFWQFPQISSEWRLFLITGFLGSLTTFSSFSSEVVELFFNDKWLNGFCVLMMHLFGCLAMTVLGIWIYKICSQLLS",
)

#: Half-width of every checkpoint's self-check band, in nats/token.
#:
#: Sized from two measurements rather than from taste. The **spread of a
#: correctly loaded model** across everything an environment can plausibly change
#: is 0.004 nats: on the 112M, under EXP-R2-242's residue-only span, bfloat16 and
#: float16 at batch sizes 8, 4 and 1 give 2.3265-2.3271, scoring N->C only instead
#: of both directions gives 2.3236, and CPU bfloat16 gives 2.3265. The **distance
#: to the nearest corruption** that ``strict=True`` cannot see is 0.71 nats on the
#: 112M and 2.12 nats on the 3B, so a half-width of 0.3 nats is ~75x the observed
#: spread and still leaves more than 0.40 nats of clearance below the nearest
#: corruption on both checkpoints: it cannot fail on hardware and cannot pass on a
#: broken mapping. One shared half-width rather than one per checkpoint because it
#: is a
#: statement about measurement noise, which does not scale with the model; what
#: does scale is the value it is centred on, and that is declared per checkpoint.
#:
#: The lower end is not a numerical tolerance. A value materially *below* the
#: measured one means the scored-target convention moved -- a mask that stopped
#: scoring the hard positions, say -- which corrupts every downstream number just
#: as thoroughly as a wrong expert mapping and would otherwise look like an
#: improvement.
SELF_CHECK_HALF_WIDTH = 0.30


@dataclass(frozen=True)
class SelfCheckReference:
    """One checkpoint's measured self-check evidence and the band it licenses.

    ``correct_mapping`` is that checkpoint's own mean per-token NLL of
    :data:`SELF_CHECK_SEQUENCES` under this module's scoring convention (both
    directions, the residue targets :func:`residue_target_mask` selects, bfloat16);
    ``corruptions`` are what the mappings that survive ``strict=True`` score on the
    same eight sequences.
    Both are measured on the checkpoint they are filed under, never carried over
    from another one: the panel-wide figures in the module docstring were taken
    on 64 Swiss-Prot records at 60-400 residues and do not transfer to this
    eight-record set, and the 112M's figures do not transfer to the 3B.

    A reference with no corruption beside it is a band nobody has shown to
    discriminate, so :func:`check_nll` publishes the corruptions next to the
    verdict and the tests hold every declared checkpoint to clearing its own
    nearest one.
    """

    name: str
    correct_mapping: float
    corruptions: dict[str, float]

    @property
    def band(self) -> tuple[float, float]:
        """The interval :func:`check_nll` accepts, in nats/token."""

        return (
            self.correct_mapping - SELF_CHECK_HALF_WIDTH,
            self.correct_mapping + SELF_CHECK_HALF_WIDTH,
        )

    @property
    def measured(self) -> dict[str, float]:
        """Every value measured on this checkpoint, correct mapping first."""

        return {"correct_mapping": self.correct_mapping, **self.corruptions}


#: The declared checkpoints, keyed by the architecture fingerprint
#: ``(num_hidden_layers, hidden_size, intermediate_size)`` of the config that was
#: actually loaded. Keyed by the config rather than by the directory name because
#: ``PROGEN3_DIR`` relocates the weights and a mirror is free to rename
#: the directory, while the shape the state dict has to match is not free to
#: move.
SELF_CHECK_REFERENCES: dict[tuple[int, int, int], SelfCheckReference] = {
    # Re-measured under EXP-R2-242's residue-only scored span, on one H200,
    # bfloat16, both directions, batch 8: that campaign's stage-20 loader
    # self-check measured 2.3270, an independent probe 2.3271, and CPU bfloat16
    # 2.3265.
    (10, 384, 1152): SelfCheckReference(
        name="progen3-112m",
        correct_mapping=2.3270,
        corruptions={
            "w1_v1_swapped": 3.0355,
            "gate_rows_rolled_by_one": 3.0662,
            "from_pretrained_eager_random_moe": 18.5247,
        },
    ),
    # Re-measured under EXP-R2-242's residue-only scored span, on one H200,
    # bfloat16, both directions, batch 8: that campaign's stage-20 loader
    # self-check measured 1.5301 and an independent probe on another card of the
    # same pod measured 1.5315. The nearest corruption here stands 1.82 nats
    # above the band, so the band does not rest on the difference.
    (24, 1280, 3840): SelfCheckReference(
        name="progen3-3b",
        correct_mapping=1.5301,
        corruptions={
            "w1_v1_swapped": 4.4700,
            "gate_rows_rolled_by_one": 3.6496,
            "from_pretrained_eager_random_moe": 11.3407,
        },
    ),
}

#: ProGen3's attention calls ``F.scaled_dot_product_attention`` inside
#: ``sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])``, and that backend has
#: no float32 kernel: a float32 model raises ``RuntimeError: No available
#: kernel`` from inside the first layer. Half precision is not a performance
#: choice here, it is the only precision this checkpoint runs in, so
#: :func:`load_progen3` refuses the others up front rather than letting an
#: unrelated-looking kernel error surface ten frames down.
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


# ------------------------------------------------------------------ conversion


def convert_megablocks_state_dict(
    raw: dict[str, torch.Tensor], *, num_experts: int, intermediate_size: int
) -> dict[str, torch.Tensor]:
    """Rewrite released megablocks-packed weights for the eager ``SparseMoeBlock``.

    Pure and checkpoint-free, so the mapping can be tested without the weights.
    ``num_experts`` and ``intermediate_size`` come from the model config and are
    checked against every packed tensor: a packed row count that is not
    ``num_experts * intermediate_size`` would otherwise be split into
    wrong-sized slices and reported by ``load_state_dict`` as a shape mismatch
    on a key whose name says nothing about which of the two numbers was wrong.
    """

    expected_rows = num_experts * intermediate_size
    converted: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if key in DROPPED_KEYS:
            continue
        if ".block_sparse_moe.router.layer.weight" in key:
            converted[key.replace(".router.layer.weight", ".gate.weight")] = value
            continue
        if ".block_sparse_moe.experts.mlp." not in key:
            converted[key] = value
            continue
        which = key.rsplit(".", 1)[1]
        if which not in EXPERT_TARGETS:
            raise ValueError(
                f"{key}: packed expert tensor {which!r} has no declared target; "
                f"this module knows {sorted(EXPERT_TARGETS)}"
            )
        if value.shape[0] != expected_rows:
            raise ValueError(
                f"{key}: packed row count {value.shape[0]} is not num_experts * "
                f"intermediate_size = {num_experts} * {intermediate_size} = "
                f"{expected_rows}; the expert split would be silently wrong"
            )
        target = EXPERT_TARGETS[which]
        for expert in range(num_experts):
            block = value[expert * intermediate_size : (expert + 1) * intermediate_size, :]
            # w2 is the DOWN projection: packed as (ffn, hidden), wanted as
            # (hidden, ffn). The other two are (ffn, hidden) either way.
            payload = block.T.contiguous() if which == "w2" else block
            converted[
                key.replace(
                    f".experts.mlp.{which}", f".experts.{expert}.{target}.weight"
                )
            ] = payload
    return converted


def release_shards(checkpoint: Path | str = PROGEN3_CHECKPOINT) -> list[Path]:
    """The safetensors files a release consists of, single-file or sharded.

    ProGen3-112M ships one ``model.safetensors``; ProGen3-3B ships two shards and
    a ``model.safetensors.index.json`` naming them. Where the index exists it is
    the authority on which files the release consists of, so a shard it names and
    the directory does not hold is refused here, by name and as an incomplete
    release. That is the state an interrupted download leaves behind -- the index
    is small and arrives first, so it describes more than the directory holds --
    and safetensors' own error for it names a file that could not be opened
    without saying that the checkpoint is half there.
    """

    checkpoint = Path(checkpoint)
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = [checkpoint / name for name in sorted(set(weight_map.values()))]
        absent = [shard.name for shard in shards if not shard.is_file()]
        if absent:
            raise FileNotFoundError(
                f"{index} names {len(shards)} shards and {checkpoint} is missing "
                f"{absent}; the release is incomplete"
            )
        return shards
    single = checkpoint / "model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(
        f"{checkpoint} holds neither model.safetensors nor "
        "model.safetensors.index.json; set PROGEN3_DIR"
    )


def released_state_dict(
    checkpoint: Path | str = PROGEN3_CHECKPOINT,
) -> dict[str, torch.Tensor]:
    """The released tensors exactly as shipped, in megablocks packing.

    Exposed because the audit has to compare them against the backbone embedded
    in a replacement checkpoint, and the file layout of the release is this
    module's business rather than every caller's -- including whether it arrives
    in one file or several.
    """

    from safetensors.torch import load_file  # noqa: PLC0415

    state: dict[str, torch.Tensor] = {}
    for shard in release_shards(checkpoint):
        state.update(load_file(str(shard)))
    return state


def strict_load(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load ``state`` into ``model``, refusing any missing or unexpected key.

    ``load_state_dict(strict=True)`` would also raise, and two things make this
    worth writing anyway. The refusal names the module's own failure mode rather
    than listing keys, because the whole point is that a *partial* load of this
    checkpoint is silent and plausible. And the key check happens **before**
    anything is assigned, so a refused load leaves the model untouched instead of
    leaving the caller holding the half-loaded object this module exists to
    prevent.
    """

    expected = set(model.state_dict())
    missing = sorted(expected - set(state))
    unexpected = sorted(set(state) - expected)
    if missing or unexpected:
        raise RuntimeError(
            "ProGen3 state dict does not match the model exactly, so part of the "
            "model would keep its random initialisation and the model would still "
            "run and still score. Refusing.\n"
            f"  missing ({len(missing)}): {missing}\n"
            f"  unexpected ({len(unexpected)}): {unexpected}"
        )
    model.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------- loader


@dataclass(frozen=True)
class ProGen3:
    """A loaded ProGen3 checkpoint and the handful of handles the audit needs."""

    model: Any
    config: Any
    preparer: Any
    device: torch.device
    checkpoint: Path

    @property
    def tokenizer(self) -> Any:
        return self.preparer.tokenizer

    @property
    def n_layers(self) -> int:
        return int(self.config.num_hidden_layers)

    @property
    def n_heads(self) -> int:
        return int(self.config.num_attention_heads)

    @property
    def moe_blocks(self) -> list[torch.nn.Module]:
        """The per-layer MoE blocks, in layer order.

        This is the module a replacement model replaces, so it is the one place
        the evaluation needs to reach into. See :func:`moe_intercept` for reading
        and substituting its input and output.
        """

        return [layer.block_sparse_moe for layer in self.model.model.layers]

    @property
    def attention_blocks(self) -> list[torch.nn.Module]:
        """The per-layer attention modules, in layer order.

        The counterpart of :attr:`moe_blocks`: one declaration of where a layer's
        attention lives, so :func:`ablated` addresses ``o_proj`` through it rather
        than by walking the module tree itself. ``fused_attention_norm`` decides
        which of two places that is -- directly on the decoder layer, as on the
        112M, or under ``norm_attn_norm``, as on the 3B -- and it is read from the
        config rather than guessed from the module tree, so a layout that is
        neither raises ``AttributeError`` rather than silently addressing the
        wrong module. Both layouts run the same ``_sdpa_attn``, so the
        head-to-column correspondence :func:`ablated` relies on is the same one.
        """

        if self.config.fused_attention_norm:
            return [layer.norm_attn_norm.self_attn for layer in self.model.model.layers]
        return [layer.self_attn for layer in self.model.model.layers]

    def batch(self, sequences: list[str], *, reverse: bool = False) -> dict[str, torch.Tensor]:
        """Model kwargs for a batch of raw residue strings, on this model's device."""

        return self.preparer.get_batch_kwargs(
            sequences, device=self.device, reverse=reverse
        )


def load_progen3(
    checkpoint: Path | str = PROGEN3_CHECKPOINT,
    *,
    device: str | torch.device = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    source: Path | str = PROGEN3_SOURCE,
) -> ProGen3:
    """Load a released ProGen3 checkpoint with every expert and router loaded.

    Every step that could fail silently is made to fail loudly instead: the
    third-party package must be present at ``source``, the release must be
    present and complete, the packed shapes must match the config, and the
    converted state dict must match the model exactly. What is left is checked
    numerically by :func:`self_check`.
    """

    checkpoint = Path(checkpoint)
    source = Path(source)
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"ProGen3 cannot run in {dtype}: its attention is pinned to the "
            f"flash SDPA backend, which has no kernel for it. Supported: "
            f"{list(SUPPORTED_DTYPES)}"
        )
    if not (source / "progen3").is_dir():
        raise FileNotFoundError(
            f"no progen3 package under {source}; set TRANSFER_PROGEN3_SRC to the "
            "src/ directory of the patched third-party copy"
        )
    # Before the third-party import and before the model is built, so that an
    # absent or half-downloaded release is named as such rather than surfacing
    # later as a missing config or a missing key.
    raw = released_state_dict(checkpoint)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    from progen3.batch_preparer import ProGen3BatchPreparer  # noqa: PLC0415
    from progen3.config import ProGen3Config  # noqa: PLC0415
    from progen3.modeling import ProGen3ForCausalLM  # noqa: PLC0415

    settings = json.loads((checkpoint / "config.json").read_text())
    settings["moe_implementation"] = "eager"
    config = ProGen3Config(**settings)
    converted = convert_megablocks_state_dict(
        raw,
        num_experts=config.num_experts,
        intermediate_size=config.intermediate_size,
    )
    model = ProGen3ForCausalLM(config)
    strict_load(model, converted)
    model = model.to(device=device, dtype=dtype).eval()
    return ProGen3(
        model=model,
        config=config,
        preparer=ProGen3BatchPreparer(),
        device=torch.device(device),
        checkpoint=checkpoint,
    )


# --------------------------------------------------------------------- scoring


def forward(pg: ProGen3, batch: dict[str, torch.Tensor]) -> Any:
    """One forward pass on a prepared batch: the single spelling of these kwargs.

    ``sequence_ids`` and ``position_ids`` are not optional on this model and are
    not what a generic decoder call passes, so every place that ran the model
    used to restate them -- scoring here, and two capture sweeps in
    ``replacement_faithfulness.py``. Three copies of one call is three places
    a keyword can be dropped, and a dropped ``sequence_ids`` is a plausible
    number rather than an error.
    """

    return pg.model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        sequence_ids=batch["sequence_ids"],
        use_cache=False,
        return_dict=True,
    )


def scored_logits(
    pg: ProGen3, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(logits, targets, mask)`` aligned for next-token scoring, logits in fp32.

    The shift and the non-residue mask are declared once here because every
    downstream number -- NLL, KL, every ablation effect -- is taken through them,
    and an off-by-one on either produces a plausible number rather than an error.
    ``targets`` is every label after the first position and ``mask`` selects the
    residues among them: ``prepare_clm`` places its ``"1"`` marker, the ``"2"``
    terminator and ``<eos>`` inside the batch's labels, so without the mask those
    three positions are scored on every record of the panel's only arm that
    carries them, and no comparable arm scores any of its markers (ProGen2's
    direction marker is position 0 and therefore never a target; ProtGPT2's,
    ZymCTRL's and ProteinGLM's renderings carry no terminal marker at all).
    """

    output = forward(pg, batch)
    labels = batch["labels"]
    targets = labels[..., 1:]
    mask = residue_target_mask(pg, labels)
    return output.logits[..., :-1, :].float(), targets, mask


def token_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-position negative log-likelihood of ``targets`` under ``logits``."""

    flat = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    )
    return flat.view(targets.shape)


@torch.no_grad()
def sequence_nll(
    pg: ProGen3,
    sequences: list[str],
    *,
    batch_size: int = 8,
    directions: tuple[bool, ...] = (False, True),
) -> torch.Tensor:
    """Per-sequence mean NLL, one row per (sequence, direction).

    ``directions`` holds the ``reverse`` flags to score. ProGen3 is trained in
    both N->C and C->N, and its published perplexities are the mean of the two;
    a stage that scores one direction is measuring something else and has to say
    so, which is why this is an argument and not a constant.
    """

    rows: list[torch.Tensor] = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        for reverse in directions:
            batch = pg.batch(chunk, reverse=reverse)
            logits, targets, mask = scored_logits(pg, batch)
            nll = token_nll(logits, targets)
            rows.append(((nll * mask).sum(1) / mask.sum(1)).float().cpu())
    return torch.cat(rows)


def mean_nll(pg: ProGen3, sequences: list[str], **kwargs: Any) -> float:
    """Mean per-token NLL over :func:`sequence_nll`'s rows."""

    return float(sequence_nll(pg, sequences, **kwargs).mean())


# ------------------------------------------------------------------ self-check


def self_check_reference(config: Any) -> SelfCheckReference:
    """The declared self-check evidence for the checkpoint ``config`` describes.

    An undeclared architecture is refused rather than gated against another
    checkpoint's number. A band is a tripwire only where the value it brackets
    and the corruptions it clears were measured on the checkpoint being scored;
    borrowed, it is either a band that rejects a correct load or one no wrong
    mapping can fall outside, and both are worse than no gate because both come
    with a verdict.
    """

    fingerprint = (
        int(config.num_hidden_layers),
        int(config.hidden_size),
        int(config.intermediate_size),
    )
    reference = SELF_CHECK_REFERENCES.get(fingerprint)
    if reference is None:
        raise KeyError(
            "no self-check reference has been measured for a ProGen3 with "
            f"(layers, hidden, intermediate) = {fingerprint}, so its loader gate "
            "would degenerate to 'did it load' -- which this checkpoint's eager "
            "path passes with every expert random. Measure the correct mapping "
            "and at least one strict-clean corruption on SELF_CHECK_SEQUENCES, "
            "then declare them in SELF_CHECK_REFERENCES. Declared: "
            f"{[declared.name for declared in SELF_CHECK_REFERENCES.values()]}"
        )
    return reference


def check_nll(reference: SelfCheckReference, value: float) -> dict[str, Any]:
    """Refuse an NLL outside the band this checkpoint's own measurement declares.

    Separated from :func:`self_check` so that each declared band can be tested
    against its own recorded corruption values without a checkpoint or a GPU.
    """

    low, high = reference.band
    inside = low <= value <= high
    record = {
        "checkpoint": reference.name,
        "nll": float(value),
        "band": [float(low), float(high)],
        "n_sequences": len(SELF_CHECK_SEQUENCES),
        "reference": dict(reference.measured),
        "verdict": "PASS" if inside else "FAIL",
    }
    if not inside:
        raise RuntimeError(
            f"ProGen3 self-check NLL {value:.4f} nats/token is outside the band "
            f"[{low:.4f}, {high:.4f}] declared for {reference.name}. Values "
            f"measured on that checkpoint over the same eight sequences: "
            f"{reference.measured}. Above the band the most likely cause is a "
            "wrong expert or router mapping, which load_state_dict(strict=True) "
            "cannot see; below it, a change to the scored-target convention. "
            "Either way the model must not be measured until this is resolved."
        )
    return record


def self_check(pg: ProGen3, *, batch_size: int = 8) -> dict[str, Any]:
    """Score the frozen sequences and refuse unless the NLL is in band.

    The reference is resolved before the model is scored, so a checkpoint nobody
    has measured costs a refusal rather than a forward pass and a number.
    """

    reference = self_check_reference(pg.config)
    return check_nll(
        reference, mean_nll(pg, list(SELF_CHECK_SEQUENCES), batch_size=batch_size)
    )


# ------------------------------------------------------- interventions and taps


#: Tokens that are not residues. Padding and the two terminus markers are what a
#: CLM batch actually contains; ``<mask>``, ``<bos_glm>``, ``<eos_span>`` and the
#: hundred span tokens belong to the infilling objective and never appear in one,
#: so the list is a subset of ProGen3's special tokens that coincides with it for
#: every batch this repository builds.
NON_RESIDUE_TOKENS = ("<pad>", "<bos>", "<eos>", "<mask>", "1", "2")


def non_residue_token_ids(pg: ProGen3) -> tuple[int, ...]:
    """The ids :data:`NON_RESIDUE_TOKENS` names, in that declaration's order.

    The single resolution of that declaration, for the two spans that have to
    agree about which positions of a batch are not residues: the one
    reconstruction statistics are read on, :func:`content_mask`, and the one a
    likelihood is scored on, :func:`residue_target_mask`.
    """

    vocabulary = pg.tokenizer.get_vocab()
    absent = [name for name in NON_RESIDUE_TOKENS if name not in vocabulary]
    if absent:
        raise RuntimeError(
            f"the ProGen3 tokenizer carries no {absent}; the residue mask cannot "
            "be built and the scored span would silently include non-residue "
            "positions"
        )
    return tuple(int(vocabulary[name]) for name in NON_RESIDUE_TOKENS)


def content_mask(pg: ProGen3, input_ids: torch.Tensor) -> torch.Tensor:
    """Residue positions: everything but padding and the sequence markers.

    The single copy of this decision. Both the replacement-faithfulness stage
    and the transcoder trainer score reconstructions, and a reconstruction
    scored on padding is scored on the easiest positions in the batch -- so the
    two must not be allowed to disagree about which positions those are
    (Appendix B rule 12).
    """

    mask = torch.ones_like(input_ids, dtype=torch.bool)
    for value in non_residue_token_ids(pg):
        mask &= input_ids != value
    return mask


def residue_target_mask(pg: ProGen3, labels: torch.Tensor) -> torch.Tensor:
    """Mask over next-token targets: ``True`` where the target is a residue.

    Column ``q`` governs the prediction of ``labels[q + 1]``, so the mask has one
    fewer column than ``labels``. Built from :func:`non_residue_token_ids`, so the
    scored span and :func:`content_mask`'s reconstruction span are the same span
    read at a different offset, rather than two rules that can drift apart.
    """

    targets = labels[..., 1:]
    mask = torch.ones_like(targets, dtype=torch.bool)
    for value in non_residue_token_ids(pg):
        mask &= targets != value
    return mask


@contextmanager
def moe_intercept(
    pg: ProGen3, fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor | None]
) -> Iterator[None]:
    """Read or replace every MoE block's output while the model runs.

    ``fn(layer, block_input, block_output)`` returns a tensor to substitute for
    the block's output, or ``None`` to leave it alone -- so the same primitive
    serves capture, transcoder replacement and mean ablation.

    Two facts about the block make a hand-written hook a trap, and they are the
    reason this is a function rather than a comment. The eager block returns
    ``(hidden_states, router_probabilities)``, so a hook that returns a bare
    tensor silently drops the router term; and a capture hook that returns the
    value it captured replaces the output with itself, which works until the
    captured tensor is detached or moved. Interceptors compose in entry order:
    the innermost ``with`` sees the output the outer one produced, which is what
    lets an ablation be applied on top of a replacement.
    """

    handles = []
    for layer, block in enumerate(pg.moe_blocks):

        def hook(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
            layer: int = layer,
        ) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            replacement = fn(layer, inputs[0], hidden)
            if replacement is None:
                return None
            if isinstance(output, tuple):
                return (replacement,) + tuple(output[1:])
            return replacement

        handles.append(block.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def router_probabilities(pg: ProGen3, layer: int, block_input: torch.Tensor) -> torch.Tensor:
    """The router distribution a MoE block **selects on**, recomputed in float32.

    The block returns a distribution alongside its hidden states, and that one is
    not this one: it is ``softmax`` in the hidden states' dtype, while the
    ``topk`` that picks the experts runs on a float32 ``softmax`` of the same
    logits. EXP-R2-130 measured the two disagreeing on the selected pair for 3.3%
    of tokens. Anything that labels a token by which experts ran must therefore
    recompute rather than read, which is what this does -- through the block's own
    ``gate``, so there is one declaration of where the router lives.

    Returns ``(..., num_experts)`` on ``block_input``'s leading shape.
    """

    gate = pg.moe_blocks[layer].gate
    return torch.softmax(gate(block_input).float(), dim=-1)


def router_probabilities_agree(pg: ProGen3, layer: int, block_input: torch.Tensor, returned: torch.Tensor) -> dict[str, Any]:
    """Whether the recomputed router matches the one the block handed back.

    Not a duplicate of the computation but a check on the *addressing*: if
    :attr:`ProGen3.moe_blocks` or the gate's name ever moved, the recomputation
    would silently describe a different module, and every routing cell derived
    from it would be a label for something else. The two are expected to differ
    only by the dtype of their softmax, so the tolerance is a dtype tolerance and
    the selected-set disagreement is reported rather than asserted away.
    """

    recomputed = router_probabilities(pg, layer, block_input).reshape(-1, returned.shape[-1])
    reference = returned.float().reshape(-1, returned.shape[-1])
    top_k = int(pg.config.num_experts_per_tok)
    chosen = recomputed.topk(top_k, dim=-1).indices.sort(dim=-1).values
    reference_choice = reference.topk(top_k, dim=-1).indices.sort(dim=-1).values
    return {
        "max_absolute_difference": float((recomputed - reference).abs().max()),
        "selected_set_disagreement": float((chosen != reference_choice).any(dim=-1).float().mean()),
        "n_tokens": int(reference.shape[0]),
    }


@dataclass(frozen=True)
class Component:
    """One ablatable unit, named the same way in the original and a replacement.

    Only units that exist unchanged in both models qualify, which is what makes
    a cross-model comparison of their effects meaningful. Experts do not
    qualify: a transcoder replacement has no experts, so an expert ablation has
    no counterpart to be compared against.
    """

    kind: str
    layer: int
    index: int | None = None

    @property
    def label(self) -> str:
        if self.index is None:
            return f"{self.kind}.L{self.layer}"
        return f"{self.kind}.L{self.layer}H{self.index}"


def component_grid(n_layers: int, n_heads: int, *, block_kind: str) -> list[Component]:
    """Every attention head, then every replaceable block, in layer order.

    Architecture-neutral because the grid is: what differs between a MoE decoder
    and a dense one is the *name* of the block being replaced, not the shape of
    the set. Declared once here, beside :class:`Component`, so that the dense
    adapter in :mod:`src.capability.replaceable` builds the same object rather than
    a parallel copy of it that could come to disagree about the ordering the
    saved effect matrices are indexed by.
    """

    heads = [
        Component("attention_head", layer, head)
        for layer in range(n_layers)
        for head in range(n_heads)
    ]
    blocks = [Component(block_kind, layer) for layer in range(n_layers)]
    return heads + blocks


def components(pg: ProGen3) -> list[Component]:
    """Every attention head, then every MoE block, in layer order."""

    return component_grid(pg.n_layers, pg.n_heads, block_kind="moe_block")


@contextmanager
def ablated(pg: ProGen3, component: Component) -> Iterator[None]:
    """Zero one component's contribution to the residual stream.

    An attention head is zeroed at the input of ``o_proj``, which is the only
    place its own slice still exists: ``_sdpa_attn`` reshapes ``(bsz, heads,
    len, head_dim)`` to ``(bsz, len, heads * head_dim)`` with heads contiguous,
    so head ``h`` owns columns ``h * head_dim`` to ``(h + 1) * head_dim``. A MoE
    block is zeroed at its output, which in a replacement model is the
    replacement's output -- the matched intervention.
    """

    if component.kind == "moe_block":
        target = component.layer

        def zero(layer: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | None:
            return torch.zeros_like(y) if layer == target else None

        with moe_intercept(pg, zero):
            yield
        return

    if component.kind != "attention_head":
        raise ValueError(f"no ablation is implemented for component kind {component.kind!r}")

    head_dim = pg.config.hidden_size // pg.n_heads
    low = component.index * head_dim
    high = low + head_dim
    projection = pg.attention_blocks[component.layer].o_proj

    def pre(module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        masked = inputs[0].clone()
        masked[..., low:high] = 0
        return (masked,) + tuple(inputs[1:])

    handle = projection.register_forward_pre_hook(pre)
    try:
        yield
    finally:
        handle.remove()


# ----------------------------------------------------- stage-01 N-to-C budget

#: Directory of the 3B rung. The 112M stays on :data:`PROGEN3_CHECKPOINT` so a
#: host that only relocates ``PROGEN3_DIR`` keeps the retrieval-bound dispatch.
PROGEN3_3B_CHECKPOINT = MODEL_ROOT / "progen3-3b"

PROGEN3_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def require_progen3_handle(arm: Any) -> ProGen3:
    """The packed-weight handle a budget score must use, or a loud refusal.

    Hugging Face eager MoE on these releases leaves every expert and router
    random and still runs. A generic ``model(input_ids=...)`` call is that
    failure mode, so the serving handle has to be present and has to be the
    object :func:`load_progen3` returned.
    """

    handle = (getattr(arm, "serving_provenance", None) or {}).get("progen3")
    if not isinstance(handle, ProGen3):
        raise ValueError(
            f"{getattr(arm, 'name', arm)}: architecture progen3 must be scored "
            "through src.capability.models.progen3.load_progen3; the HF eager path loads "
            "random experts and is not this estimand"
        )
    return handle


def n_to_c_target_rows(
    pg: ProGen3, sequences: Sequence[str], *, max_len: int
) -> list[np.ndarray]:
    """Per-record N-to-C residue ids from the native preparer, no forward pass.

    The same span :func:`scored_logits` scores, from the same
    :func:`residue_target_mask`, so a held-out reference counted here and a
    likelihood scored there are taken over one target set: the `"1"` marker,
    the `"2"` terminator and `<eos>` are context, not counts.
    """

    if max_len < 2:
        raise ValueError("max_len must admit at least one next-token target")
    if not sequences:
        raise ValueError("ProGen3 N-to-C scoring needs at least one sequence")
    rows: list[np.ndarray] = []
    for sequence in sequences:
        batch = pg.batch([sequence], reverse=False)
        width = int(batch["input_ids"].shape[-1])
        if width > max_len:
            raise ValueError(
                f"ProGen3 native rendering is {width} tokens and max_len is {max_len}"
            )
        labels = batch["labels"][0]
        targets = labels[1:]
        mask = residue_target_mask(pg, labels)
        ids = targets[mask].detach().cpu().numpy().astype(np.int64)
        if ids.size < 1:
            raise ValueError("a ProGen3 record produced no scored N-to-C targets")
        rows.append(ids)
    return rows


@torch.no_grad()
def n_to_c_scored_arrays(
    pg: ProGen3,
    sequences: Sequence[str],
    *,
    max_len: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(target_ids, nll_nats, sequence_index)`` for left-to-right scoring only.

    Bidirectional mean NLL is a different estimand (stage 20 DMS). Context
    information here is stage 01's left-to-right next-token NLL against a
    held-out unigram of the same N-to-C targets.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    target_blocks: list[np.ndarray] = []
    nll_blocks: list[np.ndarray] = []
    index_blocks: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        chunk = list(sequences[start : start + batch_size])
        batch = pg.batch(chunk, reverse=False)
        width = int(batch["input_ids"].shape[-1])
        if width > max_len:
            raise ValueError(
                f"ProGen3 native rendering is {width} tokens and max_len is {max_len}"
            )
        logits, targets, mask = scored_logits(pg, batch)
        nll = token_nll(logits, targets)
        keep = mask.bool()
        if not bool(torch.isfinite(nll[keep]).all()):
            raise FloatingPointError("ProGen3: non-finite N-to-C NLL")
        row = torch.arange(targets.shape[0], device=targets.device).unsqueeze(1).expand_as(
            targets
        )
        target_blocks.append(targets[keep].detach().cpu().numpy().astype(np.int64))
        nll_blocks.append(nll[keep].detach().float().cpu().numpy().astype(np.float64))
        index_blocks.append(
            (row[keep] + start).detach().cpu().numpy().astype(np.int64)
        )
    return (
        np.concatenate(target_blocks),
        np.concatenate(nll_blocks),
        np.concatenate(index_blocks),
    )


def load_progen3_as_arm(
    name: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    check: bool = True,
) -> Any:
    """Load one ProGen3 rung as an :class:`~src.capability.core.arms.Arm` for budget scoring.

    Not :func:`src.capability.core.arms.load_arm_spec`: that path is AutoModel and would
    score random experts. The self-check runs before the arm is returned.
    """

    from ..core.arms import Arm, arm_spec

    spec = arm_spec(name)
    if spec.architecture != "progen3":
        raise ValueError(f"{name}: load_progen3_as_arm is for architecture progen3")
    torch_dtype = PROGEN3_DTYPES.get(dtype)
    if torch_dtype is None:
        raise ValueError(
            f"ProGen3 cannot run in {dtype}: supported {sorted(PROGEN3_DTYPES)}"
        )
    pg = load_progen3(spec.path, device=device, dtype=torch_dtype)
    record = self_check(pg) if check else None
    return Arm(
        spec=spec,
        model=pg.model,
        tokenizer=pg.tokenizer,
        device=str(pg.device),
        dtype=dtype,
        serving_provenance={
            "progen3": pg,
            "self_check": record,
            "scoring_directions": ("n_to_c",),
            "not_bidirectional": (
                "stage 01 context information is left-to-right next-token NLL; "
                "the bidirectional mean used by stage 20 is a different estimand"
            ),
        },
    )
