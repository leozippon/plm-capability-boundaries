#!/usr/bin/env python3
"""Assemble the declared-property table of the readout panel, then relate it to outcomes.

Why two subcommands rather than one. The panel holds 33 unconditioned arms and
8 of them carry a representation residual over the strongest control this
programme has measured. With that many properties and that few survivors, a
table built *after* looking at which arms survived would find a separating
property whatever the truth is. So ``properties`` builds the table from declared
checkpoint metadata alone, reads no fitted increment, and prints the SHA-256 of
its canonical JSON form; ``separate`` refuses to run unless the table it is
handed still hashes to the digest that was recorded before any outcome was
attached.

Every value is either read from an artifact, read from ``src.capability.core.arms``, or
carried here as a literal transcribed from a repository record that is named in
:data:`PROSE_SOURCES`. Nothing is inferred from a model card this host cannot
read, and a property no record states is written ``not_recorded`` rather than
guessed: the whole point of the table is to say what the panel can and cannot
distinguish, and an invented cell would answer that question falsely.

CPU-only, seconds of work, no model load and no cohort draw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The admitted readout expansion archive. Its 34 per-arm manifests are the only
#: machine-readable per-checkpoint records this host holds: the checkpoints
#: themselves live in the pod's filesystem and are not reachable from here.
EXPANSION_ARCHIVE = (
    REPO_ROOT / "results/R2/readout_expansion_20260923/complete.tar.gz"
)

#: ZymCTRL is the one admitted arm outside the 33-arm anchor roster: its
#: EC-conditioned 40-assay panel is a different support, so no increment of its
#: is comparable with an anchor arm's. It is assembled and then excluded, so the
#: exclusion is visible rather than a name that never appears.
EC_CONDITIONED_ARM = "zymctrl"

PROSE_SOURCES = {
    "readout_widths": (
        "archive/docs/R2/readout-results.md, the two per-arm hidden-width tables in "
        "Limitations (20 batch-one arms and 14 batch-eight arms, 34 together)"
    ),
    "segmentation": (
        "archive/docs/R2/readout-depth-sweep.md, 'Position resolution is an interface "
        "property': the 19 arms whose rendering assigns one token per residue "
        "and the 15 whose tokens carry several, checked mechanically on the "
        "first eligible assay before extraction"
    ),
    "joint_renderings": "src/capability/models/joint_modes.py, JOINT_RENDERINGS",
    "prollama_lineage": "src/capability/models/joint_lineage.py, RUNGS",
    "instructprotein": "src/capability/models/instructprotein_fitness.py",
    "galactica": "src/capability/models/galactica_fitness.py",
    "arms": "src/capability/core/arms.py, PANEL / STAGED_ARMS / PROGEN3_ARMS",
    "readout_extraction": (
        "src/capability/readouts/readout_extraction.py, TEXT_EXTRA_ARM: the instruct arm "
        "reuses the qwen2.5-0.5b ArmSpec with a replaced name and path"
    ),
    "nominal_sizes": (
        "src/capability/core/arms.py load-check comments (progen2-small 151.1M, "
        "progen2-medium 765M, progen2-large 2779.4M, progen2-xlarge 6443.6M, "
        "protgpt2 774,030,080, rita-xl 1,208,655,872), "
        "../d2/archive/docs/prollama-capability-localization.md (Llama lineage "
        "6,738,417,664 stored elements), and each remaining checkpoint's own "
        "released size in its name"
    ),
}

#: Hidden width per arm, transcribed from the admitted readout report. Where
#: ``arms.py`` also declares ``d_model`` the two are required to agree, which is
#: what makes this transcription checkable rather than a second source of truth.
HIDDEN_WIDTH = {
    "bygpt5-base-en": 1536,
    "bygpt5-medium-en": 1536,
    "bygpt5-small-en": 1472,
    "dialogpt-small": 768,
    "galactica-1.3b": 2048,
    "galactica-125m": 768,
    "galactica-30b": 7168,
    "galactica-6.7b": 4096,
    "gpt2": 768,
    "gpt2-large": 1280,
    "gpt2-medium": 1024,
    "gpt2-xl": 1600,
    "instructprotein": 2048,
    "llama-2-7b": 4096,
    "llama-3.2-3b": 3072,
    "progen2-base": 1536,
    "progen2-large": 2560,
    "progen2-medium": 1536,
    "progen2-small": 1024,
    "progen2-xlarge": 4096,
    "progen3-112m": 384,
    "progen3-3b": 1280,
    "prollama": 4096,
    "prollama-stage-1": 4096,
    "proteinglm-7b-clm": 4096,
    "protgpt2": 1280,
    "protgpt3-1.3b": 1024,
    "qwen2.5-0.5b": 896,
    "qwen2.5-0.5b-instruct": 896,
    "qwen2.5-32b": 5120,
    "qwen2.5-7b": 3584,
    "qwen3-8b-base": 4096,
    "rita-xl": 2048,
    "zymctrl": 1280,
}

#: Nominal released parameter count, to the precision its source states. Exact
#: to the unit for the six checkpoints whose load checks recorded one; otherwise
#: the release's own stated size. Compared between arms with a declared 10%
#: tolerance, because two checkpoints of identical architecture can carry
#: nominal figures that differ only in rounding (ProGen2-base and -medium are
#: declared as the same parameter count and store byte-identical totals, yet
#: their stated sizes read 764M and 765M).
NOMINAL_PARAMETERS = {
    "bygpt5-base-en": 139_000_000,
    "bygpt5-medium-en": 289_000_000,
    "bygpt5-small-en": 73_000_000,
    "dialogpt-small": 124_000_000,
    "galactica-1.3b": 1_300_000_000,
    "galactica-125m": 125_000_000,
    "galactica-30b": 30_000_000_000,
    "galactica-6.7b": 6_700_000_000,
    "gpt2": 124_000_000,
    "gpt2-large": 774_000_000,
    "gpt2-medium": 355_000_000,
    "gpt2-xl": 1_558_000_000,
    "instructprotein": 1_300_000_000,
    "llama-2-7b": 6_738_417_664,
    "llama-3.2-3b": 3_200_000_000,
    "progen2-base": 764_000_000,
    "progen2-large": 2_779_400_000,
    "progen2-medium": 765_000_000,
    "progen2-small": 151_100_000,
    "progen2-xlarge": 6_443_600_000,
    "progen3-112m": 112_000_000,
    "progen3-3b": 3_000_000_000,
    "prollama": 6_738_417_664,
    "prollama-stage-1": 6_738_417_664,
    "proteinglm-7b-clm": 7_250_000_000,
    "protgpt2": 774_030_080,
    "protgpt3-1.3b": 1_300_000_000,
    "qwen2.5-0.5b": 494_000_000,
    "qwen2.5-0.5b-instruct": 494_000_000,
    "qwen2.5-32b": 32_500_000_000,
    "qwen2.5-7b": 7_600_000_000,
    "qwen3-8b-base": 8_200_000_000,
    "rita-xl": 1_208_655_872,
    "zymctrl": 738_000_000,
}

#: Released storage precision, transcribed only where a repository record states
#: it. Five arms do; the rest are ``not_recorded``, and the auxiliary
#: ``bytes_per_nominal_parameter`` below is what a reader has instead. That ratio
#: is not a substitute: a GPT-2-lineage checkpoint serialises a causal-mask
#: buffer of 1024x1024 entries per block alongside its weights, which inflates
#: the ratio by about 0.4 for GPT-2, DialoGPT, ProtGPT2 and ZymCTRL.
# The external note for these float16 assignments is not in this repository.
RELEASED_STORAGE_PRECISION = {
    "proteinglm-7b-clm": ("float32", "src/capability/core/arms.py: float32 weights on disk"),
    "rita-xl": ("float16", "src/capability/core/arms.py: float16 weights"),
    "llama-2-7b": (
        "float16",
        "every weight tensor stored float16 in both checkpoints, the 32 rotary buffers excepted",
    ),
    "prollama-stage-1": (
        "float16",
        "as the parent",
    ),
    "prollama": (
        "float16",
        "as the parent",
    ),
}

#: Scored-target support for arms whose ``ArmSpec`` leaves the field ``None``
#: while a repository record states the number. The five ProGen2 rungs share one
#: 31-symbol residue tokenizer behind a 32-column head; the GPT-2-lineage arms
#: and ProtGPT2 share the 50257-piece vocabulary the panel's controlled
#: comparison is declared on.
SCORED_ALPHABET_OVERRIDE = {
    "progen2-small": 32,
    "progen2-medium": 32,
    "progen2-base": 32,
    "gpt2": 50257,
    "gpt2-medium": 50257,
    "gpt2-large": 50257,
    "gpt2-xl": 50257,
    "dialogpt-small": 50257,
    "protgpt2": 50257,
    "qwen2.5-0.5b": 151936,
}

ONE_TOKEN_PER_RESIDUE = frozenset(
    {
        "progen2-small",
        "progen2-medium",
        "progen2-base",
        "progen2-large",
        "progen2-xlarge",
        "progen3-112m",
        "progen3-3b",
        "proteinglm-7b-clm",
        "protgpt3-1.3b",
        "rita-xl",
        "galactica-125m",
        "galactica-1.3b",
        "galactica-6.7b",
        "galactica-30b",
        "instructprotein",
        "bygpt5-small-en",
        "bygpt5-base-en",
        "bygpt5-medium-en",
        "zymctrl",
    }
)


@dataclass(frozen=True)
class Declared:
    """The properties no artifact and no ``ArmSpec`` on this host records."""

    architecture_family: str
    attention_structure: str
    mlp_structure: str
    tokenizer_family: str
    scored_target_alphabet: object
    pretraining_corpus: str
    protein_pretraining: str
    instruction_or_conditioning_stage: str
    protein_training_route: str


_DENSE = "dense_multi_head"
_GATED = "gated_dense"
_PLAIN = "dense_ungated"
_NR = "not_recorded"

#: Arms that ``arms.py`` does not declare, and the properties the repository's
#: own records state for them. Nine arms: the four Galactica rungs and
#: InstructProtein (reached by path through their own fitness doors), the three
#: ProLLaMA-lineage rungs (reached through ``joint_lineage.RUNGS``), and the
#: Qwen instruct arm (which reuses the base ArmSpec).
UNDECLARED_ARMS = {
    "galactica-125m": Declared(
        "opt", _DENSE, _PLAIN, "bpe", 50000,
        "scientific_text_papers_code_knowledge_bases", "no", "none", "none",
    ),
    "galactica-1.3b": Declared(
        "opt", _DENSE, _PLAIN, "bpe", 50000,
        "scientific_text_papers_code_knowledge_bases", "no", "none", "none",
    ),
    "galactica-6.7b": Declared(
        "opt", _DENSE, _PLAIN, "bpe", 50000,
        "scientific_text_papers_code_knowledge_bases", "no", "none", "none",
    ),
    "galactica-30b": Declared(
        "opt", _DENSE, _PLAIN, "bpe", 50000,
        "scientific_text_papers_code_knowledge_bases", "no", "none", "none",
    ),
    "instructprotein": Declared(
        "opt", _DENSE, _PLAIN, "bpe_with_dedicated_residue_ids", 50304,
        "uniref100_continued_pretraining_then_instruction_tuning", "yes",
        "instruction_tuned", "continued_pretraining_from_text_parent",
    ),
    "llama-2-7b": Declared(
        "llama", _NR, _GATED, "sentencepiece", 32000,
        "llama2_text_corpus", "no", "none", "none",
    ),
    "prollama-stage-1": Declared(
        "llama", _NR, _GATED, "sentencepiece", 32000,
        "llama2_text_then_protein_lora_continued_pretraining", "yes", "none",
        "continued_pretraining_from_text_parent",
    ),
    "prollama": Declared(
        "llama", _NR, _GATED, "sentencepiece", 32000,
        "llama2_text_then_protein_lora_continued_pretraining", "yes",
        "instruction_tuned", "continued_pretraining_from_text_parent",
    ),
    "qwen2.5-0.5b-instruct": Declared(
        "qwen2", "grouped_query", _GATED, "bpe", 151936,
        "qwen2.5_pretraining_mixture", "no", "instruction_tuned", "none",
    ),
}

#: Attention and MLP structure for the arms ``arms.py`` declares. Its
#: ``architecture`` field names a family, not a sublayer form, so these two are
#: carried here with the record that states them and left ``not_recorded``
#: wherever no record does.
STRUCTURE_BY_ARCHITECTURE = {
    "gpt2": (_DENSE, _PLAIN),
    "progen": (_DENSE, _PLAIN),
    "progen3": (_NR, "sparse_moe"),
    "proteinglm": (_NR, _NR),
    "rita": (_NR, _NR),
    "mixtral": ("grouped_query", "sparse_moe"),
    "qwen2": ("grouped_query", _GATED),
    "qwen3": ("grouped_query", _GATED),
    "llama": (_NR, _GATED),
    "t5_decoder": ("t5_relative_bias", "gated_gelu"),
    "opt": (_DENSE, _PLAIN),
}

PROTEIN_CORPORA = frozenset(
    {
        "uniref50",
        "uniref90_bfd30",
        "progen2_base_mixture",
        "uniprot_ec_annotated",
        "profluent_protein_atlas_v1",
        "uniref50s_uniref90_colabfolddb",
        "uniref100",
        "uniref100_continued_pretraining_then_instruction_tuning",
        "llama2_text_then_protein_lora_continued_pretraining",
    }
)

#: Arms whose protein pretraining is recorded but whose corpus name is the
#: ``undeclared`` sentinel. The corpus identity is genuinely unrecorded; the fact
#: that the checkpoint is a protein decoder is recorded by its own release.
PROTEIN_WITH_UNDECLARED_CORPUS = frozenset({"progen3-112m", "protgpt3-1.3b"})

#: The ordered property set. Group ``checkpoint`` holds declared properties of
#: the released weights; group ``execution`` holds two facts about how the
#: admitted extraction read them, which are not checkpoint properties and are
#: carried because both are material confounds across this panel.
PROPERTY_SET = (
    ("depth_blocks", "checkpoint", "transformer block count"),
    ("hidden_width", "checkpoint", "residual-stream width"),
    ("parameter_count_nominal", "checkpoint", "nominal released parameter count, matched within 10%"),
    ("released_storage_precision", "checkpoint", "storage dtype where a record states it"),
    ("architecture_family", "checkpoint", "declared model family"),
    ("attention_structure", "checkpoint", "dense multi-head, grouped-query or T5 relative bias"),
    ("mlp_structure", "checkpoint", "dense ungated, gated dense, gated GELU or sparse mixture of experts"),
    ("tokenizer_family", "checkpoint", "residue, multi-residue BPE, BPE, byte or sentencepiece"),
    ("scored_target_alphabet", "checkpoint", "declared support of scored next-token targets"),
    ("measured_segmentation", "checkpoint", "one token per residue or multi-residue, as measured"),
    ("pretraining_corpus", "checkpoint", "declared training corpus or the undeclared sentinel"),
    ("protein_pretraining", "checkpoint", "whether any declared training stage saw protein sequence"),
    ("training_objective", "checkpoint", "fitted training objective"),
    ("instruction_or_conditioning_stage", "checkpoint", "none, instruction tuned, or conditioned input"),
    ("protein_training_route", "checkpoint", "from scratch, continued from a text parent, or none"),
    ("extraction_load_precision", "execution", "dtype the admitted extraction loaded"),
    ("extraction_batch_size", "execution", "batch size the admitted extraction ran"),
)

PROPERTY_NAMES = tuple(name for name, _group, _definition in PROPERTY_SET)

#: Properties with a direction, reported as a range on each side of the boundary
#: rather than as a value count.
ORDINAL_PROPERTIES = frozenset(
    {
        "depth_blocks",
        "hidden_width",
        "parameter_count_nominal",
        "scored_target_alphabet",
        "extraction_batch_size",
    }
)
CHECKPOINT_PROPERTIES = tuple(
    name for name, group, _definition in PROPERTY_SET if group == "checkpoint"
)

# --------------------------------------------------------------------- outcomes
#
# Read only by ``separate``, and only after the property digest has been
# verified. Classes and counts are transcribed from the admitted 33-arm
# generalizability check in archive/docs/R1/gate-local-context.md; that document reports
# the 14 arms with at least one seed resolved below zero as a count rather than
# as a list, so those arms carry one class and are not split further here.

REPRESENTATION_OUTCOME = {
    "resolved_positive_all_three": (
        "progen2-small", "progen2-medium", "progen2-base", "progen2-large",
        "progen2-xlarge", "progen3-112m", "progen3-3b", "proteinglm-7b-clm",
    ),
    "resolved_positive_two_of_three": ("protgpt2", "rita-xl"),
    "unresolved_all_three": (
        "galactica-125m", "galactica-1.3b", "galactica-6.7b", "galactica-30b",
        "instructprotein", "protgpt3-1.3b", "prollama-stage-1", "prollama",
        "qwen2.5-32b",
    ),
}

LIKELIHOOD_OUTCOME = {
    "resolved_positive_all_three": (
        "progen2-small", "progen2-medium", "progen2-base", "progen2-large",
        "progen2-xlarge", "progen3-112m", "progen3-3b", "proteinglm-7b-clm",
        "protgpt2", "protgpt3-1.3b", "rita-xl", "instructprotein",
        "prollama-stage-1", "prollama",
    ),
    "one_resolved_seed": ("galactica-30b", "qwen2.5-32b"),
}


_MANIFEST_CACHE: dict[str, dict] = {}


def _manifests() -> dict[str, dict]:
    """Per-arm identity records of the admitted expansion, from its archive.

    Memoised because the archive is 323 MB of gzip and a caller that validates
    the table's refusals rebuilds it several times.
    """

    if _MANIFEST_CACHE:
        return _MANIFEST_CACHE
    if not EXPANSION_ARCHIVE.is_file():
        raise FileNotFoundError(
            f"{EXPANSION_ARCHIVE} is absent; this table is built from the admitted "
            "expansion archive and has no second source on this host"
        )
    found: dict[str, dict] = {}
    with tarfile.open(EXPANSION_ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            base = Path(member.name).name
            if not (base.startswith("manifest_") and base.endswith(".json")):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"{member.name} is not a readable archive member")
            payload = json.loads(handle.read().decode("utf-8"))
            arm = payload["identity"]["arm"]
            if arm in found:
                raise ValueError(f"{arm} has two manifests in the archive")
            found[arm] = payload
    if len(found) != 34:
        raise ValueError(f"expected 34 manifests in the archive, found {len(found)}")
    _MANIFEST_CACHE.update(found)
    return _MANIFEST_CACHE


#: Parameter counts are nominal figures at different roundings, so two arms match
#: on that property when their counts agree to within this fraction. Every other
#: property is compared for exact equality.
PARAMETER_TOLERANCE = 0.10


def compare_property(left: dict, right: dict, name: str) -> str:
    """Classify one property of one pair as matching, differing or undecidable.

    The third class is not a nicety. Attention and MLP structure are unrecorded
    for four of the panel's protein arms, and released storage precision for
    twenty-nine of the thirty-four, so treating an absent record as a difference
    would rank a pair by how little the repository happens to know about it.
    """

    a, b = left[name], right[name]
    if a == _NR or b == _NR:
        return "undecidable"
    if name == "parameter_count_nominal":
        return "differ" if abs(a - b) / max(a, b) > PARAMETER_TOLERANCE else "match"
    return "differ" if a != b else "match"


def build_properties() -> dict:
    """The declared-property table. Reads no fitted increment of any kind."""

    from src.capability.core import arms as arms_module

    spec_tables = (
        arms_module.PANEL,
        arms_module.STAGED_ARMS,
        arms_module.PROGEN3_ARMS,
    )
    specs = {}
    for table in spec_tables:
        for name, spec in table.items():
            specs.setdefault(name, spec)

    manifests = _manifests()
    rows: dict[str, dict] = {}
    for arm, payload in sorted(manifests.items()):
        identity = payload["identity"]
        blocks = payload["block_indices"]
        depth = int(blocks[-1]) + 1
        stored_bytes = sum(
            int(entry["bytes"]) for entry in identity["checkpoint_tensor_files"]
        )
        spec = specs.get(arm)

        if arm not in HIDDEN_WIDTH:
            raise ValueError(f"{arm} has no transcribed hidden width")
        width = HIDDEN_WIDTH[arm]

        if spec is not None:
            if int(spec.n_layer) != depth:
                raise ValueError(
                    f"{arm}: arms.py declares {spec.n_layer} blocks and the admitted "
                    f"manifest's block indices imply {depth}"
                )
            if int(spec.d_model) != width:
                raise ValueError(
                    f"{arm}: arms.py declares width {spec.d_model} and the admitted "
                    f"readout report records {width}"
                )
            architecture = spec.architecture
            tokenizer = spec.tokenisation
            alphabet = spec.scoring_target_alphabet_size
            if alphabet is None:
                alphabet = SCORED_ALPHABET_OVERRIDE.get(arm, _NR)
            corpus = spec.pretraining_corpus
            stage = "ec_conditioned_input" if arm == EC_CONDITIONED_ARM else "none"
            attention, mlp = STRUCTURE_BY_ARCHITECTURE[architecture]
            protein = (
                "yes"
                if corpus in PROTEIN_CORPORA or arm in PROTEIN_WITH_UNDECLARED_CORPUS
                else "no"
            )
            route = "from_scratch" if protein == "yes" else "none"
        else:
            declared = UNDECLARED_ARMS[arm]
            architecture = declared.architecture_family
            tokenizer = declared.tokenizer_family
            alphabet = declared.scored_target_alphabet
            corpus = declared.pretraining_corpus
            stage = declared.instruction_or_conditioning_stage
            attention = declared.attention_structure
            mlp = declared.mlp_structure
            protein = declared.protein_pretraining
            route = declared.protein_training_route

        nominal = NOMINAL_PARAMETERS[arm]
        precision, precision_source = RELEASED_STORAGE_PRECISION.get(arm, (_NR, _NR))
        segmentation = (
            "one_token_per_residue"
            if arm in ONE_TOKEN_PER_RESIDUE
            else "multi_residue"
        )

        rows[arm] = {
            "depth_blocks": depth,
            "hidden_width": width,
            "parameter_count_nominal": nominal,
            "released_storage_precision": precision,
            "architecture_family": architecture,
            "attention_structure": attention,
            "mlp_structure": mlp,
            "tokenizer_family": tokenizer,
            "scored_target_alphabet": alphabet if alphabet is not None else _NR,
            "measured_segmentation": segmentation,
            "pretraining_corpus": corpus,
            "protein_pretraining": protein,
            "training_objective": "next_token_likelihood",
            "instruction_or_conditioning_stage": stage,
            "protein_training_route": route,
            "extraction_load_precision": identity["dtype"],
            "extraction_batch_size": int(identity["batch_size"]),
            "stored_tensor_bytes": stored_bytes,
            "bytes_per_nominal_parameter": round(stored_bytes / nominal, 3),
            "released_storage_precision_source": precision_source,
            "admitted_assays": len(payload["assays"]),
            "in_anchor_roster": arm != EC_CONDITIONED_ARM,
        }

    for arm, row in rows.items():
        missing = [name for name in PROPERTY_NAMES if name not in row]
        if missing:
            raise ValueError(f"{arm} is missing {missing}")

    return {
        "schema_version": "matched_pair_properties_v1",
        "property_set": [
            {"name": name, "group": group, "definition": definition}
            for name, group, definition in PROPERTY_SET
        ],
        "prose_sources": PROSE_SOURCES,
        "anchor_roster_size": sum(1 for row in rows.values() if row["in_anchor_roster"]),
        "excluded_arm": EC_CONDITIONED_ARM,
        "arms": rows,
    }


def canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(payload: dict) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def _separation(arms: dict[str, dict], inside: set[str]) -> dict:
    """Per-property value counts on each side of one outcome boundary.

    A categorical property separates when no value appears on both sides; an
    ordinal one separates when the two ranges do not overlap. ``not_recorded``
    is counted as an ordinary value here, so a property that is unrecorded on
    both sides reads as non-separating rather than as agreeing.
    """

    outside = sorted(set(arms) - inside)
    report = {}
    for name in PROPERTY_NAMES:
        values_in = Counter(arms[arm][name] for arm in sorted(inside))
        values_out = Counter(arms[arm][name] for arm in outside)
        shared = sorted(set(values_in) & set(values_out), key=str)
        entry = {
            "inside": {str(k): v for k, v in sorted(values_in.items(), key=lambda kv: str(kv[0]))},
            "outside": {str(k): v for k, v in sorted(values_out.items(), key=lambda kv: str(kv[0]))},
            "values_on_both_sides": [str(value) for value in shared],
            "separates": not shared,
        }
        if name in ORDINAL_PROPERTIES:
            numeric_in = [v for v in values_in if isinstance(v, int)]
            numeric_out = [v for v in values_out if isinstance(v, int)]
            entry["inside_range"] = [min(numeric_in), max(numeric_in)]
            entry["outside_range"] = [min(numeric_out), max(numeric_out)]
            entry["ranges_overlap"] = (
                min(numeric_in) <= max(numeric_out)
                and min(numeric_out) <= max(numeric_in)
            )
            entry["separates"] = not entry["ranges_overlap"]
        report[name] = entry
    return report


def _outcome_class(arm: str, table: dict[str, tuple[str, ...]], fallback: str) -> str:
    for label, members in table.items():
        if arm in members:
            return label
    return fallback


def separate(payload: dict, expected_digest: str | None) -> dict:
    """Relate the frozen table to the two outcome classes. Never edits the table."""

    actual = digest(payload)
    if expected_digest is not None and actual != expected_digest:
        raise ValueError(
            f"property table digest {actual} does not match the declared "
            f"{expected_digest}; the table changed after it was frozen"
        )

    arms = {
        name: row for name, row in payload["arms"].items() if row["in_anchor_roster"]
    }
    if len(arms) != 33:
        raise ValueError(f"the anchor roster must hold 33 arms, holds {len(arms)}")

    representation = {
        arm: _outcome_class(arm, REPRESENTATION_OUTCOME, "some_seed_resolved_negative")
        for arm in arms
    }
    likelihood = {
        arm: _outcome_class(arm, LIKELIHOOD_OUTCOME, "no_resolved_seed")
        for arm in arms
    }
    survivors = {
        arm for arm, label in representation.items()
        if label == "resolved_positive_all_three"
    }
    if len(survivors) != 8:
        raise ValueError(f"expected 8 representation survivors, got {len(survivors)}")

    likelihood_positive = {
        arm for arm, label in likelihood.items()
        if label == "resolved_positive_all_three"
    }
    if len(likelihood_positive) != 14:
        raise ValueError(
            f"expected 14 likelihood-positive arms, got {len(likelihood_positive)}"
        )

    separation = _separation(arms, survivors)
    likelihood_separation = _separation(arms, likelihood_positive)

    pairs = []
    for left, right in combinations(sorted(arms), 2):
        rep_differs = representation[left] != representation[right]
        lik_differs = likelihood[left] != likelihood[right]
        if not (rep_differs or lik_differs):
            continue
        verdicts = {
            name: compare_property(arms[left], arms[right], name)
            for name in CHECKPOINT_PROPERTIES
        }
        differing = [n for n, v in verdicts.items() if v == "differ"]
        matching = [n for n, v in verdicts.items() if v == "match"]
        undecidable = [n for n, v in verdicts.items() if v == "undecidable"]
        pairs.append(
            {
                "arms": [left, right],
                "n_differing": len(differing),
                "n_undecidable": len(undecidable),
                "differing": differing,
                "matching": matching,
                "undecidable": undecidable,
                "representation_classes": [representation[left], representation[right]],
                "likelihood_classes": [likelihood[left], likelihood[right]],
                "representation_outcome_differs": rep_differs,
                "likelihood_outcome_differs": lik_differs,
                "position_resolved_both": (
                    arms[left]["measured_segmentation"] == "one_token_per_residue"
                    and arms[right]["measured_segmentation"] == "one_token_per_residue"
                ),
                "depth_matched": arms[left]["depth_blocks"] == arms[right]["depth_blocks"],
                "width_matched": arms[left]["hidden_width"] == arms[right]["hidden_width"],
            }
        )
    pairs.sort(
        key=lambda entry: (entry["n_differing"], entry["n_undecidable"], entry["arms"])
    )

    return {
        "property_table_sha256": actual,
        "representation_outcome": representation,
        "likelihood_outcome": likelihood,
        "representation_separation": separation,
        "likelihood_separation": likelihood_separation,
        "pairs": pairs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("properties", help="build and digest the declared-property table")
    build.add_argument("--out", type=Path, required=True)

    relate = sub.add_parser("separate", help="relate a frozen table to the outcome classes")
    relate.add_argument("--properties", type=Path, required=True)
    relate.add_argument("--expect-sha256", default=None)
    relate.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "properties":
        payload = build_properties()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(canonical(payload))
        print(f"arms={len(payload['arms'])} anchor={payload['anchor_roster_size']}")
        print(f"properties={len(PROPERTY_NAMES)} sha256={digest(payload)}")
        return 0

    payload = json.loads(args.properties.read_bytes().decode("utf-8"))
    report = separate(payload, args.expect_sha256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical(report))
    print(f"property_table_sha256={report['property_table_sha256']}")
    for label in ("representation_separation", "likelihood_separation"):
        separating = [
            name for name, entry in report[label].items() if entry["separates"]
        ]
        print(f"{label}: separating_properties={separating}")
    print(f"outcome_differing_pairs={len(report['pairs'])}")
    for entry in report["pairs"][:12]:
        print(
            f"  {entry['n_differing']:>2}d {entry['n_undecidable']:>2}u  "
            f"{entry['arms'][0]} | {entry['arms'][1]}"
            f"  rep={entry['representation_outcome_differs']}"
            f" lik={entry['likelihood_outcome_differs']}"
            f" posres={entry['position_resolved_both']}"
            f" depth={entry['depth_matched']} width={entry['width_matched']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
