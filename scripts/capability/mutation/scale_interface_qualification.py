#!/usr/bin/env python3
"""ProGen2 scale-interface qualification. Not a panel stage and not a capability claim.

This run asks only whether the medium/large/xlarge loaders expose a usable
direction-controlled scoring interface: native ``\"1\" + sequence`` versus a
wrong direction marker ``\"2\" + sequence`` over the same residue targets,
scored with full-width log_softmax/gather.

A PASS means that interface and direction control are available. It is not
panel admission, not a causal scale result, and not a knowledge result.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.arms import (  # noqa: E402
    N_TO_C_MARKER,
    PANEL,
    REPO,
    Arm,
    arm_spec,
    load_arm_spec,
    output_logit_width,
    require_scoring_target_ids,
    scoring_target_alphabet,
)
from src.capability.core.io import write_json  # noqa: E402

SCHEMA_VERSION = "r2_transfer_scale_interface_qualification_v1"
DEFAULT_OUT = REPO / "results/R1/scale_interface_qualification"
SUCCESS_ARTEFACT = "scale_interface_qualification.json"

SCALE_INTERFACE_ARMS = ("progen2-medium", "progen2-large", "progen2-xlarge")
FIXED_SEQUENCE = (
    "MKTIIALSYIFCLVFADYKDDDDKACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYGGHPEPTIDE"
)
FIXED_SEQUENCE_SHA256 = (
    "36ff2a577d03461a6e995ef2a492604c8d510161058f4e575b12ddda968cc954"
)
NATIVE_MARKER = N_TO_C_MARKER
WRONG_MARKER = "2"
NATIVE_REPEAT_MAX_ABS = 1e-6
WRONG_MARKER_COST_MIN = 0.05
REQUIRED_SCORING_SUPPORT = 32
#: Live output width per rung, taken from the released heads rather than from a
#: config key the lineage spells three different ways. ``progen2-large`` carries
#: ``lm_head.weight [51200, 2560]`` and declares ``vocab_size`` 51200;
#: ``progen2-xlarge`` carries ``lm_head.weight [32, 4096]``, declares no
#: ``vocab_size`` at all, and builds its head from ``vocab_size_lm_head`` 32.
#: EXP-R2-068 already recorded that xlarge "loads and runs a forward pass
#: returning logits of width 32". Support and live width therefore coincide on
#: medium and xlarge and differ only on large; they are still checked separately,
#: because that they coincide is a fact about two checkpoints and not a rule.
REQUIRED_LIVE_WIDTH = {
    "progen2-medium": 32,
    "progen2-large": 51200,
    "progen2-xlarge": 32,
}

LOGITS_NOT_CROPPED_NOTE = (
    "full-width log_softmax/gather; no crop, slice, or 32-column renormalisation"
)
NOT_PANEL_ADMISSION = (
    "this run does not admit a checkpoint to PANEL; progen2-large and "
    "progen2-xlarge remain staged non-members"
)
DESCRIPTIVE_NOT_CAUSAL = (
    "a pass here is not a causal scale claim and is not a capability result"
)
NO_KNOWLEDGE = (
    "a pass here is not evidence that a model has learned biological knowledge"
)
PASS_MEANS = (
    "interface and direction control are usable; not a capability or knowledge result"
)


def sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def require_fixed_sequence(sequence: str) -> str:
    """Refuse any sequence other than the frozen 74-aa probe."""

    digest = sequence_digest(sequence)
    if sequence != FIXED_SEQUENCE or digest != FIXED_SEQUENCE_SHA256 or len(sequence) != 74:
        raise ValueError(
            "scale interface qualification is frozen to the 74-aa probe "
            f"with SHA256 {FIXED_SEQUENCE_SHA256}; got length {len(sequence)} "
            f"digest {digest}"
        )
    return digest


def require_qualification_arms(names: Sequence[str]) -> tuple[str, ...]:
    """Refuse any arm set or order other than medium, large, xlarge."""

    got = list(names)
    if got != list(SCALE_INTERFACE_ARMS):
        raise ValueError(
            "scale interface qualification arms are fixed as "
            f"{list(SCALE_INTERFACE_ARMS)}; got {got}"
        )
    leaked = [name for name in ("progen2-large", "progen2-xlarge") if name in PANEL]
    if leaked:
        raise ValueError(
            f"{leaked} are PANEL members; this stage is not panel admission"
        )
    return tuple(got)


def native_input(sequence: str) -> str:
    return NATIVE_MARKER + sequence


def wrong_marker_input(sequence: str) -> str:
    return WRONG_MARKER + sequence


def encode_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, return_tensors=None)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("tokenizer returned a batch, not a single sequence")
        ids = ids[0]
    return [int(value) for value in ids]


def require_single_marker_token(tokenizer: Any, marker: str, *, label: str) -> int:
    ids = encode_ids(tokenizer, marker)
    if len(ids) != 1:
        raise ValueError(
            f"{label} marker {marker!r} tokenized to {len(ids)} tokens, not one"
        )
    return ids[0]


def aligned_marker_targets(tokenizer: Any, *, sequence: str) -> dict[str, Any]:
    """Tokenize native and wrong-marker renderings and require aligned residue targets."""

    native_marker_id = require_single_marker_token(
        tokenizer, NATIVE_MARKER, label="native"
    )
    wrong_marker_id = require_single_marker_token(
        tokenizer, WRONG_MARKER, label="wrong"
    )
    if native_marker_id == wrong_marker_id:
        raise ValueError(
            f"native marker {NATIVE_MARKER!r} and wrong marker {WRONG_MARKER!r} "
            f"tokenized to the same id {native_marker_id}"
        )
    native_ids = encode_ids(tokenizer, native_input(sequence))
    wrong_ids = encode_ids(tokenizer, wrong_marker_input(sequence))
    if not native_ids or native_ids[0] != native_marker_id:
        raise ValueError(
            f"native rendering must start with marker id {native_marker_id}; "
            f"got {native_ids[:1]}"
        )
    if not wrong_ids or wrong_ids[0] != wrong_marker_id:
        raise ValueError(
            f"wrong rendering must start with marker id {wrong_marker_id}; "
            f"got {wrong_ids[:1]}"
        )
    native_targets = native_ids[1:]
    wrong_targets = wrong_ids[1:]
    if native_targets != wrong_targets:
        raise ValueError(
            "wrong-marker residue target ids must match native item-by-item "
            "(same residues, no reversal)"
        )
    if len(native_targets) != len(sequence):
        raise ValueError(
            f"expected {len(sequence)} residue targets, got {len(native_targets)}"
        )
    return {
        "native_marker_id": native_marker_id,
        "wrong_marker_id": wrong_marker_id,
        "target_ids": native_targets,
        "native_token_ids": native_ids,
        "wrong_token_ids": wrong_ids,
    }


def require_support_and_live_width(name: str, support: int, live_width: int) -> None:
    """Check scoring-target support and live logit width independently."""

    if name not in REQUIRED_LIVE_WIDTH:
        raise ValueError(f"no frozen live width for {name!r}")
    if support != REQUIRED_SCORING_SUPPORT:
        raise ValueError(
            f"{name}: scoring-target support must be {REQUIRED_SCORING_SUPPORT}, "
            f"got {support}"
        )
    expected_live = REQUIRED_LIVE_WIDTH[name]
    if live_width != expected_live:
        raise ValueError(
            f"{name}: live output width must be {expected_live}, got {live_width}"
        )


def require_uncropped_logits(
    logits: torch.Tensor, live_width: int, *, arm: str
) -> None:
    if logits.ndim != 3:
        raise ValueError(
            f"{arm}: logits must be [batch, time, width], got {tuple(logits.shape)}"
        )
    if int(logits.shape[-1]) != int(live_width):
        raise ValueError(
            f"{arm}: logits width {int(logits.shape[-1])} != live width "
            f"{live_width}; cropping is forbidden"
        )


def full_width_target_nll(
    logits: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    """Per-target NLL via full-dim log_softmax and gather. No column crop."""

    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits must be [batch, time, width] and input_ids [batch, time]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids time dimensions disagree")
    if logits.shape[1] < 2:
        raise ValueError("need at least one scored target")
    width = int(logits.shape[-1])
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    if int(logp.shape[-1]) != width:
        raise RuntimeError("log_softmax changed the output width")
    targets = input_ids[:, 1:]
    return -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def score_token_ids(
    arm: Arm, token_ids: Sequence[int], *, live_width: int
) -> list[float]:
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=arm.device)
    output = arm.model(input_ids=ids)
    if not hasattr(output, "logits"):
        raise TypeError(f"{arm.name}: model output has no logits")
    logits = output.logits
    require_uncropped_logits(logits, live_width, arm=arm.name)
    nll = full_width_target_nll(logits, ids)
    values = [float(value) for value in nll[0].detach().cpu().tolist()]
    if len(values) != len(token_ids) - 1:
        raise ValueError(
            f"{arm.name}: scored {len(values)} targets from {len(token_ids)} tokens"
        )
    return values


def require_finite(values: Sequence[float], *, label: str) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label}: non-finite NLL at target {index}")


def native_repeat_max_abs_diff(
    first: Sequence[float], second: Sequence[float]
) -> float:
    if len(first) != len(second):
        raise ValueError("native NLL repeats have different lengths")
    if not first:
        raise ValueError("native NLL is empty")
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def require_native_repeat(
    first: Sequence[float],
    second: Sequence[float],
    first_ids: Sequence[int],
    second_ids: Sequence[int],
) -> float:
    if [int(value) for value in first_ids] != [int(value) for value in second_ids]:
        raise ValueError("native repeat target ids disagree")
    require_finite(first, label="native run 1")
    require_finite(second, label="native run 2")
    delta = native_repeat_max_abs_diff(first, second)
    if delta > NATIVE_REPEAT_MAX_ABS:
        raise ValueError(
            f"native NLL repeat max abs diff {delta} exceeds {NATIVE_REPEAT_MAX_ABS}"
        )
    return delta


def mean_nll(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("no scored targets")
    return float(sum(float(value) for value in values) / len(values))


def wrong_marker_cost(mean_wrong: float, mean_native: float) -> float:
    return float(mean_wrong) - float(mean_native)


def require_wrong_marker_cost(cost: float) -> float:
    if not (cost > WRONG_MARKER_COST_MIN):
        raise ValueError(
            f"wrong-marker cost {cost} nats/target is not strictly > "
            f"{WRONG_MARKER_COST_MIN}"
        )
    return cost


def target_ids_digest(target_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in target_ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def qualify_loaded_arm(arm: Arm) -> dict[str, Any]:
    """Run the frozen interface checks on one already-loaded arm.

    The strict-load counts are read off the arm rather than reconstructed here.
    This function receives an already-built arm and cannot observe how it was
    loaded, so an arm that carries no counts is refused instead of being
    recorded as a clean strict load on no evidence.
    """

    name = arm.name
    if arm.strict_load is None:
        raise ValueError(
            f"{name}: this arm was not loaded strictly; refusing to record a "
            "strict-load block for a check that was never performed"
        )
    require_fixed_sequence(FIXED_SEQUENCE)
    alphabet = scoring_target_alphabet(arm.spec, getattr(arm.model, "config", None))
    live = output_logit_width(arm)
    require_support_and_live_width(name, int(alphabet["size"]), int(live["size"]))
    rendered = aligned_marker_targets(arm.tokenizer, sequence=FIXED_SEQUENCE)
    target_ids = [int(value) for value in rendered["target_ids"]]
    require_scoring_target_ids(target_ids, alphabet, arm=name)
    live_width = int(live["size"])
    native_nll_1 = score_token_ids(
        arm, rendered["native_token_ids"], live_width=live_width
    )
    native_nll_2 = score_token_ids(
        arm, rendered["native_token_ids"], live_width=live_width
    )
    repeat_delta = require_native_repeat(
        native_nll_1, native_nll_2, target_ids, target_ids
    )
    wrong_nll = score_token_ids(
        arm, rendered["wrong_token_ids"], live_width=live_width
    )
    require_finite(wrong_nll, label="wrong marker")
    if len(wrong_nll) != len(target_ids):
        raise ValueError(
            f"{name}: wrong-marker NLL count {len(wrong_nll)} != "
            f"target count {len(target_ids)}"
        )
    mean_native = mean_nll(native_nll_1)
    mean_wrong = mean_nll(wrong_nll)
    cost = require_wrong_marker_cost(wrong_marker_cost(mean_wrong, mean_native))
    return {
        "name": name,
        "verdict": "PASS",
        "shape": {"n_layer": int(arm.spec.n_layer), "d_model": int(arm.spec.d_model)},
        "dtype": arm.dtype,
        "strict_load": dict(arm.strict_load),
        "scoring_target_support": {
            "size": int(alphabet["size"]),
            "source": alphabet["source"],
        },
        "live_output_width": {
            "size": int(live["size"]),
            "source": live["source"],
        },
        "native_marker_id": int(rendered["native_marker_id"]),
        "wrong_marker_id": int(rendered["wrong_marker_id"]),
        "target_ids_sha256": target_ids_digest(target_ids),
        "target_count": len(target_ids),
        "native_nll_run_1": native_nll_1,
        "native_nll_run_2": native_nll_2,
        "native_mean_nll_run_1": mean_native,
        "native_mean_nll_run_2": mean_nll(native_nll_2),
        "native_repeat_max_abs_diff": float(repeat_delta),
        "wrong_nll": wrong_nll,
        "wrong_mean_nll": mean_wrong,
        "wrong_marker_cost_nats_per_target": float(cost),
    }


def release_arm(arm: Arm | None) -> None:
    if arm is None:
        return
    if getattr(arm, "model", None) is not None:
        arm.model = None  # type: ignore[assignment]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_qualification_arm(name: str, *, device: str, dtype: str) -> Arm:
    return load_arm_spec(arm_spec(name), device=device, dtype=dtype, strict=True)


def build_success_payload(
    *,
    device: str,
    dtype: str,
    sequence_digest_value: str,
    per_arm: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    names = [str(row["name"]) for row in per_arm]
    require_qualification_arms(names)
    if any(row.get("verdict") != "PASS" for row in per_arm):
        raise ValueError("refusing to build a success payload from a non-PASS arm")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "dtype": dtype,
        "fixed_sequence": FIXED_SEQUENCE,
        "fixed_sequence_sha256": sequence_digest_value,
        "fixed_sequence_length": len(FIXED_SEQUENCE),
        "logits_not_cropped": True,
        "logits_not_cropped_note": LOGITS_NOT_CROPPED_NOTE,
        "arms": list(SCALE_INTERFACE_ARMS),
        "not_panel_admission": True,
        "not_panel_admission_note": NOT_PANEL_ADMISSION,
        "descriptive_not_causal": True,
        "descriptive_not_causal_note": DESCRIPTIVE_NOT_CAUSAL,
        "no_knowledge": True,
        "no_knowledge_note": NO_KNOWLEDGE,
        "pass_means": PASS_MEANS,
        "verdict": "PASS",
        "per_arm": {str(row["name"]): dict(row) for row in per_arm},
    }


def qualify_all_arms(
    *,
    device: str,
    dtype: str,
    load_fn: Callable[..., Arm] | None = None,
    score_fn: Callable[[Arm], dict[str, Any]] | None = None,
    release_fn: Callable[[Arm | None], None] | None = None,
) -> dict[str, Any]:
    """Score every frozen arm in order and stop at the first failure."""

    names = require_qualification_arms(SCALE_INTERFACE_ARMS)
    digest = require_fixed_sequence(FIXED_SEQUENCE)
    load = load_fn or load_qualification_arm
    score = score_fn or qualify_loaded_arm
    release = release_fn or release_arm
    reports: list[dict[str, Any]] = []
    for name in names:
        arm = None
        try:
            arm = load(name, device=device, dtype=dtype)
            reports.append(score(arm))
        finally:
            release(arm)
    return build_success_payload(
        device=device,
        dtype=dtype,
        sequence_digest_value=digest,
        per_arm=reports,
    )


def write_success_artefact(directory: Path, payload: Mapping[str, Any]) -> Path:
    if payload.get("verdict") != "PASS":
        raise ValueError("refusing to write a non-PASS artefact")
    destination = Path(directory) / SUCCESS_ARTEFACT
    write_json(destination, dict(payload))
    return destination


def run_qualification(
    *,
    device: str,
    dtype: str,
    out: Path,
    load_fn: Callable[..., Arm] | None = None,
    score_fn: Callable[[Arm], dict[str, Any]] | None = None,
    release_fn: Callable[[Arm | None], None] | None = None,
) -> Path:
    destination_dir = Path(out)
    destination = destination_dir / SUCCESS_ARTEFACT
    if destination.exists():
        destination.unlink()
    destination_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = qualify_all_arms(
            device=device,
            dtype=dtype,
            load_fn=load_fn,
            score_fn=score_fn,
            release_fn=release_fn,
        )
        return write_success_artefact(destination_dir, payload)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path = run_qualification(device=args.device, dtype=args.dtype, out=args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
