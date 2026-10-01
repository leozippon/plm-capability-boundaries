"""Native unlabelled generation door for the seventeen missing arms (EXP-R2-246).

One frozen operating point per admitted checkpoint. There is no requested class,
so target-profile hits are undefined. ProGen3-3B is already complete and is
refused here rather than regenerated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from . import conditioned_generation as cg
from . import generation_evidence as ge
from ..core.arms import (
    BOS_DIRECTION_C_TO_N,
    BOS_DIRECTION_N_TO_C,
    EOS_BOUNDED_BOUNDARY,
    N_TO_C_MARKER,
    arm_spec,
)
from ..core.io import sha256_file, write_json
from ..models.joint_modes import rendering
from ..models.proteinglm import NATIVE_PREFIX as PROTEINGLM_PREFIX

CAMPAIGN = "EXP-R2-246"
EXPECT_GENERATE = "unconditional_generation.json"
EXPECT_INTERFACE = "native_generation_interface.json"
LEDGER_NAME = "attempts.jsonl"
STRUCTURE_SUBSET = "main_subset.jsonl"

POLICY = {
    "campaign": CAMPAIGN,
    "attempts": 800,
    "structure_samples": 128,
    "seed": 20260905,
    "temperature": 0.85,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.0,
    "min_new_tokens": 0,
    "max_new_tokens": 400,
    "batch_size": 8,
    "postselection": None,
    "length_strata": ge.POLICY["length_strata"],
    "structure_support": [16, 1024],
    "batch_seed_rule": "seed_plus_batch_index",
}

GenerateFn = Callable[..., list[str]]


@dataclass(frozen=True)
class UnconditionalArm:
    """One admitted unlabelled generation point and its native prompt."""

    name: str
    prompt: str
    close_token: str
    batch_size: int = 8
    use_cache: bool = True
    residue_escape: str | None = None
    official_progen3: bool = False
    dtype: str = "bfloat16"
    add_special_tokens: bool = True


def _galactica(name: str) -> UnconditionalArm:
    family = rendering("galactica")
    return UnconditionalArm(
        name=name,
        prompt=family.protein_start,
        close_token=family.protein_end,
        residue_escape=family.residue_escape,
        dtype="float32",
    )


def _progen2(name: str) -> UnconditionalArm:
    return UnconditionalArm(
        name=name,
        prompt=N_TO_C_MARKER,
        close_token=BOS_DIRECTION_C_TO_N,
        use_cache=False,
    )


def _prollama(name: str) -> UnconditionalArm:
    family = rendering("prollama")
    return UnconditionalArm(
        name=name,
        prompt=family.protein_start,
        close_token=family.protein_end,
    )


_INSTRUCT = rendering("instructprotein")

ADMITTED: dict[str, UnconditionalArm] = {
    "protgpt2": UnconditionalArm(
        name="protgpt2",
        prompt="<|endoftext|>\n",
        close_token="<|endoftext|>",
    ),
    "progen2-small": _progen2("progen2-small"),
    "progen2-base": _progen2("progen2-base"),
    "progen2-medium": _progen2("progen2-medium"),
    "progen2-large": _progen2("progen2-large"),
    "progen2-xlarge": _progen2("progen2-xlarge"),
    "progen3-112m": UnconditionalArm(
        name="progen3-112m",
        prompt=N_TO_C_MARKER,
        close_token="<eos>",
        official_progen3=True,
    ),
    "proteinglm-7b-clm": UnconditionalArm(
        name="proteinglm-7b-clm",
        prompt=PROTEINGLM_PREFIX,
        close_token="<eos>",
        dtype="float32",
    ),
    "protgpt3-1.3b": UnconditionalArm(
        name="protgpt3-1.3b",
        prompt=f"<|bos|>{BOS_DIRECTION_N_TO_C}",
        close_token="<|eos|>",
    ),
    "rita-xl": UnconditionalArm(
        name="rita-xl",
        prompt=EOS_BOUNDED_BOUNDARY,
        close_token=EOS_BOUNDED_BOUNDARY,
        batch_size=1,
        use_cache=False,
        dtype="float32",
        add_special_tokens=False,
    ),
    "galactica-125m": _galactica("galactica-125m"),
    "galactica-1.3b": _galactica("galactica-1.3b"),
    "galactica-6.7b": _galactica("galactica-6.7b"),
    "galactica-30b": _galactica("galactica-30b"),
    "instructprotein": UnconditionalArm(
        name="instructprotein",
        prompt=f"{_INSTRUCT.prefix_marker}{_INSTRUCT.protein_start}",
        close_token=_INSTRUCT.protein_end,
        residue_escape=_INSTRUCT.residue_escape,
        add_special_tokens=False,
    ),
    "prollama-stage-1": _prollama("prollama-stage-1"),
    "prollama": _prollama("prollama"),
}

ADMITTED_ARMS: tuple[str, ...] = tuple(ADMITTED)

#: Smaller checkpoints first; Galactica-30B is last and must run alone.
GENERATION_ORDER: tuple[str, ...] = (
    "progen3-112m",
    "galactica-125m",
    "progen2-small",
    "protgpt2",
    "progen2-base",
    "progen2-medium",
    "rita-xl",
    "protgpt3-1.3b",
    "galactica-1.3b",
    "instructprotein",
    "progen2-large",
    "progen2-xlarge",
    "galactica-6.7b",
    "proteinglm-7b-clm",
    "prollama-stage-1",
    "prollama",
    "galactica-30b",
)

REFUSED: dict[str, str] = {
    "zymctrl": "ZymCTRL's native input is an EC class tag, not an unlabelled protein start",
    "llama-2-7b": "Llama-2-7B Seq=< is a scoring wrap, not a declared unlabelled generator",
    "progen3-3b": "ProGen3-3B is already complete under EXP-R2-233; do not regenerate",
}

TEXT_ONLY: frozenset[str] = frozenset(
    {
        "gpt2-large",
        "dialogpt-small",
        "qwen2.5-0.5b",
        "llama-3.2-3b",
        "qwen2.5-7b",
        "qwen2.5-32b",
        "qwen3-8b-base",
    }
)

if set(GENERATION_ORDER) != set(ADMITTED):
    raise AssertionError("GENERATION_ORDER must list every admitted arm once")
if GENERATION_ORDER[-1] != "galactica-30b":
    raise AssertionError("galactica-30b must be last")
if "progen3-3b" in ADMITTED:
    raise AssertionError("progen3-3b is complete and must not be admitted here")


def require_admitted(arm: str) -> UnconditionalArm:
    """Refuse excluded checkpoints with the reason, then unknown names."""

    if arm in REFUSED:
        raise ValueError(REFUSED[arm])
    if arm in TEXT_ONLY:
        raise ValueError(f"{arm} is text-only and has no unlabelled protein generation point")
    if arm not in ADMITTED:
        raise KeyError(
            f"unknown unconditional-generation arm {arm!r}; admitted: {list(ADMITTED_ARMS)}"
        )
    return ADMITTED[arm]


def prompt_for(arm: str) -> str:
    return require_admitted(arm).prompt


def close_token_for(arm: str) -> str:
    return require_admitted(arm).close_token


def arm_batch_size(arm: str) -> int:
    return require_admitted(arm).batch_size


def extract_residues(arm: str, continuation: str) -> str:
    """Strict leading AA20 run of one continuation, empty strings retained."""

    spec = require_admitted(arm)
    text = continuation.split(spec.close_token, 1)[0]
    if spec.residue_escape:
        text = text.replace(spec.residue_escape, "")
    return cg.extract_protein(text, end_delimiter=spec.close_token)


def checkpoint_for_arm(arm: str) -> Path:
    """Resolve the staged weights the way the existing doors already name them."""

    require_admitted(arm)
    if arm.startswith("galactica-"):
        from ..models.galactica_fitness import CHECKPOINTS

        return CHECKPOINTS[arm]
    if arm == "instructprotein":
        from ..models.instructprotein_fitness import CHECKPOINTS

        return CHECKPOINTS[arm]
    if arm in {"prollama-stage-1", "prollama"}:
        from ..models.joint_lineage import rung

        return rung(arm).checkpoint
    return arm_spec(arm).path


def stop_status(arm: str, raw: str, residues: str) -> str:
    close = close_token_for(arm)
    if close and close in raw:
        return "native_terminal"
    if not residues:
        return "empty_or_noncanonical"
    return "budget_censored"


def attempt_row(arm: str, raw: str, index: int) -> dict[str, Any]:
    residues = extract_residues(arm, raw)
    row = ge._row(
        residues,
        arm=arm,
        class_key=None,
        condition="unconditioned",
        role="generation",
        source_label="unconditional_raw_attempts",
        source_key=CAMPAIGN,
        source_sample_index=index,
        primary_class=False,
    )
    row.update(
        {
            "campaign": CAMPAIGN,
            "raw_continuation": raw,
            "stop_status": stop_status(arm, raw, residues),
            "class_key": None,
            "target_profile_hit": None,
            "condition": "unconditioned",
        }
    )
    return row


def select_structure_parents(
    rows: list[dict],
    *,
    n: int | None = None,
    seed: int | None = None,
) -> tuple[list[dict], dict]:
    """Length-stratified 128-sample plus one composition shuffle; score-blind."""

    budget = POLICY["structure_samples"] if n is None else n
    draw_seed = POLICY["seed"] if seed is None else seed
    strata: dict[str, list[dict]] = {}
    for row in rows:
        if row["support_status"] == "eligible":
            strata.setdefault(row["stratum"], []).append(row)
        else:
            row["structure_exclusion_reason"] = row["support_status"]
    if not strata:
        return [], {}
    allocation = ge.allocate_strata({key: len(value) for key, value in strata.items()}, budget)
    selected: list[dict] = []
    blocks: dict[str, Any] = {}
    for key, count in allocation.items():
        population = strata[key]
        chosen = ge._uniform(
            population,
            count,
            seed=draw_seed,
            key=f"{CAMPAIGN}|structure|{key}",
        )
        for row in chosen:
            row.update(
                {
                    "phase": "main",
                    "inclusion_probability": count / len(population),
                    "selected_for_structure": True,
                }
            )
            selected.append(dict(row))
        blocks[key] = {
            "N": len(population),
            "n": count,
            "inclusion_probability": count / len(population),
        }
    for row in rows:
        if not row["selected_for_structure"] and row["structure_exclusion_reason"] is None:
            row["structure_exclusion_reason"] = "eligible_not_sampled"
    selected.extend(ge.composition_shuffle(row, seed=draw_seed) for row in list(selected))
    return sorted(selected, key=lambda row: row["id"]), blocks


def _decode_kwargs(spec: UnconditionalArm) -> dict[str, Any]:
    return {
        "max_new_tokens": POLICY["max_new_tokens"],
        "temperature": POLICY["temperature"],
        "top_p": POLICY["top_p"],
        "top_k": POLICY["top_k"],
        "use_cache": spec.use_cache,
        "add_special_tokens": spec.add_special_tokens,
    }


def _sample_next_id(logits: Any, *, temperature: float, top_k: int, top_p: float) -> Any:
    """One token from last-step logits. Does not call ``model.generate``."""

    import torch

    scores = logits / float(temperature)
    if top_k > 0:
        kept = min(int(top_k), int(scores.size(-1)))
        threshold = torch.topk(scores, kept).values[-1]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    if 0.0 < top_p < 1.0:
        ranked, order = torch.sort(scores, descending=True)
        cumulative = torch.cumsum(torch.softmax(ranked, dim=-1), dim=-1)
        drop = cumulative > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        ranked = ranked.masked_fill(drop, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(-1, order, ranked)
    return torch.multinomial(torch.softmax(scores, dim=-1), 1)


def _eos_id(spec: UnconditionalArm, tokenizer: Any) -> int:
    if spec.name == "rita-xl":
        from ..models.rita_fitness import NATIVE_EOS_ID

        return NATIVE_EOS_ID
    token_id = tokenizer.convert_tokens_to_ids(spec.close_token)
    if token_id is None or int(token_id) < 0:
        raise ValueError(f"{spec.name}: close token {spec.close_token!r} has no id")
    return int(token_id)


def _sample_by_forward(
    spec: UnconditionalArm, model: Any, tokenizer: Any, *, n: int, seed: int,
    token_traces: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Token-by-token decode through ``forward`` only.

    Transformers 4.50 ``generate`` passes ``cache_position`` and builds a
    ``DynamicCache`` from ``num_hidden_layers``. RITA and ProteinGLM accept
    neither. The operating point is still POLICY; only the decode driver changes.
    """

    import torch

    encoded = tokenizer(spec.prompt, return_tensors="pt", add_special_tokens=spec.add_special_tokens)
    prompt_ids = encoded["input_ids"]
    device = getattr(model, "device", None)
    if device is None:
        device = prompt_ids.device
    prompt_ids = prompt_ids.to(device)
    prompt_length = int(prompt_ids.shape[1])
    if prompt_length < 1:
        raise ValueError(f"{spec.name}: empty prompt")
    eos_id = _eos_id(spec, tokenizer)
    outputs: list[str] = []
    with torch.no_grad():
        for index in range(n):
            torch.manual_seed(seed + index)
            ids = prompt_ids
            for _ in range(POLICY["max_new_tokens"]):
                out = model(input_ids=ids)
                logits = out.logits if hasattr(out, "logits") else out[0]
                next_id = _sample_next_id(
                    logits[0, -1],
                    temperature=POLICY["temperature"],
                    top_k=POLICY["top_k"],
                    top_p=POLICY["top_p"],
                )
                ids = torch.cat([ids, next_id.view(1, 1).to(ids.device)], dim=1)
                if int(next_id.item()) == eos_id:
                    break
            outputs.append(tokenizer.decode(ids[0, prompt_length:], skip_special_tokens=False))
            if token_traces is not None:
                from .generation_replication import token_trace

                token_traces.append(token_trace(ids[0, prompt_length:].tolist(), eos_id,
                                                POLICY["max_new_tokens"]))
    return outputs


def _sample_official_progen3(model: Any, n: int, seed: int) -> list[str]:
    from .progen3_generation import make_generator

    import torch

    handle = getattr(model, "serving_provenance", {}).get("progen3")
    if handle is None:
        raise ValueError("ProGen3-112M generation needs the official loader handle")
    generator = make_generator(handle)
    outputs: list[str] = []
    batch = POLICY["batch_size"]
    batch_index = 0
    while len(outputs) < n:
        size = min(batch, n - len(outputs))
        torch.manual_seed(seed + batch_index)
        samples = list(
            generator.generate(
                prompt_for("progen3-112m"),
                size,
                POLICY["min_new_tokens"],
                POLICY["max_new_tokens"],
            )
        )
        if len(samples) != size:
            raise RuntimeError("official ProGen3 generator returned fewer attempts than requested")
        outputs.extend(item.generation for item in samples)
        batch_index += 1
    return outputs


def _sample_hf(spec: UnconditionalArm, model: Any, tokenizer: Any, n: int, seed: int) -> list[str]:
    if spec.name in {"rita-xl", "proteinglm-7b-clm"}:
        return _sample_by_forward(spec, model, tokenizer, n=n, seed=seed)
    return cg.sample_continuations(
        model,
        tokenizer,
        spec.prompt,
        n=n,
        seed=seed,
        batch_size=spec.batch_size,
        **_decode_kwargs(spec),
    )


def generate(
    arm: str,
    *,
    n: int | None = None,
    generate_fn: GenerateFn | None = None,
    model: Any = None,
    tokenizer: Any = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Sample retained attempts. A fake ``generate_fn`` keeps tests off the GPU."""

    spec = require_admitted(arm)
    count = POLICY["attempts"] if n is None else int(n)
    if count < 1:
        raise ValueError("generation needs a positive attempt count")
    prompt = spec.prompt
    if generate_fn is not None:
        raws = list(
            generate_fn(
                prompt=prompt,
                n=count,
                seed=POLICY["seed"],
                batch_size=spec.batch_size,
                **_decode_kwargs(spec),
            )
        )
    elif spec.official_progen3:
        raws = _sample_official_progen3(model, count, POLICY["seed"])
    else:
        if model is None or tokenizer is None:
            raise ValueError(f"{arm}: real generation needs a model and tokenizer, or a generate_fn")
        raws = _sample_hf(spec, model, tokenizer, count, POLICY["seed"])
    if len(raws) != count:
        raise RuntimeError(f"{arm}: generator returned {len(raws)} continuations, not {count}")
    rows = [attempt_row(arm, raw, index) for index, raw in enumerate(raws)]
    if output_dir is not None:
        write_attempt_ledger(Path(output_dir), rows, arm=arm)
    return rows


def write_attempt_ledger(output_dir: Path, rows: list[dict[str, Any]], *, arm: str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = output_dir / LEDGER_NAME
    ge.write_immutable(ledger, ge.jsonl_bytes(rows))
    summary = {
        "campaign": CAMPAIGN,
        "arm": arm,
        "stage": "generate",
        "attempts": len(rows),
        "policy": POLICY,
        "prompt": prompt_for(arm),
        "close_token": close_token_for(arm),
        "batch_size": arm_batch_size(arm),
        "attempts_sha256": sha256_file(ledger),
        "is_scientific_measurement": len(rows) == POLICY["attempts"],
    }
    write_json(output_dir / EXPECT_GENERATE, summary)
    return summary


def write_interface_record(output_dir: Path, arm: str, *, checkpoint: Path | None) -> dict[str, Any]:
    """Interface smoke receipt. Does not write the 800-attempt ledger."""

    spec = require_admitted(arm)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = output_dir / LEDGER_NAME
    if ledger.exists():
        raise ValueError(f"interface-only must not write or replace {ledger.name}")
    record = {
        "campaign": CAMPAIGN,
        "arm": arm,
        "interface_only": True,
        "is_scientific_measurement": False,
        "prompt": spec.prompt,
        "close_token": spec.close_token,
        "batch_size": spec.batch_size,
        "use_cache": spec.use_cache,
        "dtype": spec.dtype,
        "policy": POLICY,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
    }
    write_json(output_dir / EXPECT_INTERFACE, record)
    if ledger.exists():
        raise RuntimeError("interface-only wrote a scientific ledger")
    return record


def write_structure_subset(output_dir: Path, rows: list[dict[str, Any]], *, arm: str) -> dict[str, Any]:
    subset, selection = select_structure_parents(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ge.write_immutable(output_dir / STRUCTURE_SUBSET, ge.jsonl_bytes(subset))
    summary = {
        "campaign": CAMPAIGN,
        "arm": arm,
        "stage": "sample-structure",
        "structure_records_including_shuffles": len(subset),
        "structure_selection": selection,
        "main_subset_sha256": sha256_file(output_dir / STRUCTURE_SUBSET),
    }
    write_json(output_dir / "structure_selection.json", summary)
    return summary


def read_attempts(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def load_generator(arm: str, *, device: str, checkpoint: Path | None = None) -> tuple[Any, Any]:
    """Load weights for the real generate path. Tests inject a generate callable instead."""

    spec = require_admitted(arm)
    path = Path(checkpoint) if checkpoint is not None else checkpoint_for_arm(arm)
    dtype = spec.dtype
    if spec.official_progen3:
        import torch

        from ..models.progen3 import load_progen3

        pg = load_progen3(path, device=device, dtype=torch.bfloat16)
        pg.model.serving_provenance = {"progen3": pg}
        return pg.model, pg.tokenizer
    if arm in {"protgpt2", "progen2-small", "progen2-base", "progen2-medium"}:
        from ..core.arms import load_arm

        handle = load_arm(arm, device=device, dtype=dtype)
        return handle.model, handle.tokenizer
    if arm in {"progen2-large", "progen2-xlarge", "protgpt3-1.3b"}:
        from ..core.arms import load_arm_spec

        handle = load_arm_spec(arm_spec(arm), device=device, dtype=dtype)
        return handle.model, handle.tokenizer
    if arm == "proteinglm-7b-clm":
        from ..models import proteinglm as pglm

        loaded = pglm.load_pretrained(path, device=device, dtype=dtype)
        return loaded["model"], loaded["tokenizer"]
    if arm == "rita-xl":
        from ..models import rita_fitness as rita

        scorer = rita.RitaFitnessScorer(checkpoint=path, device=device, dtype=dtype, batch_size=1)
        return scorer.model, scorer.tokenizer
    if arm.startswith("galactica-"):
        from ..models import galactica_fitness as galactica

        loaded = galactica.load_galactica(arm, device=device, dtype=dtype)
        return loaded.model, loaded.tokenizer
    if arm == "instructprotein":
        from ..models import instructprotein_fitness as instruct

        loaded = instruct.load_instructprotein(arm, device=device, dtype=dtype)
        return loaded.model, loaded.tokenizer
    if arm in {"prollama-stage-1", "prollama"}:
        from ..models import joint_lineage as lineage

        loaded = lineage.load_rung(arm, device=device, dtype=dtype)
        return loaded.model, loaded.tokenizer
    raise KeyError(f"{arm}: no generator loader")


def interface_payload(arm: str, *, checkpoint: Path | None = None) -> Mapping[str, Any]:
    spec = require_admitted(arm)
    return {
        "arm": arm,
        "prompt": spec.prompt,
        "close_token": spec.close_token,
        "batch_size": spec.batch_size,
        "checkpoint": str(checkpoint if checkpoint is not None else checkpoint_for_arm(arm)),
    }
