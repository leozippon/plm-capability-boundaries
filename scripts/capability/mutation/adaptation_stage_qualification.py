#!/usr/bin/env python3
"""EXP-R2-226's all-or-stop qualification gate for one rung of the ProLLaMA lineage.

**What this decides.** A rung of ``Llama-2-7b-hf`` -> ``ProLLaMA_Stage_1`` ->
``ProLLaMA`` is scored on the two frozen protein-capability queues only after
every clause of EXP-R2-226's qualification holds. Failure of any clause stops
that rung and every later rung that would pair with it. Nothing here is replaced
to recover a rung: not the probes, not the floor, not the rendering, not the
checkpoint.

Three of the four clauses need a card and are evaluated here:

1. **Strict load** at the campaign precision, with Transformers' loading
   diagnostics exposed. ``missing_keys``, ``unexpected_keys``,
   ``mismatched_keys`` and ``error_msgs`` must all be empty; a newly initialised
   head is unavailable, never a pass (L24). The refusal lives in
   :func:`src.capability.models.joint_lineage.load_rung`, which is also what the two
   fitness stages load through, so a qualified rung and a scored rung cannot
   have been built differently.
2. **Fixed-sequence NLL self-check** under the bare ``Seq=<...>`` rendering, run
   twice at the campaign precision. The target ids must match exactly between
   the repeats, every value must be finite, and the maximum per-target absolute
   difference must be at most :data:`REPEAT_MAX_ABS` nats.
3. **Directional-reversal control** under the same bare rendering, on the
   **adapted rungs only**. Reversing the residue order must cost strictly more
   than :data:`REVERSAL_COST_MIN` nats per scored token.

The fourth clause -- context information identified by the EXP-R2-221
displacement-corrected rule -- is **reused, not redrawn**, from the stage-21
re-analysis this repository already holds. It needs no card and no model, so it
is not evaluated here; ``adaptation_stage_capability.py`` reads it off that
artefact and refuses a ladder whose rungs do not carry it.

**The base rung enters as a declared floor, not as a qualified protein arm.**
Clause 3 is deliberately not applied to ``llama-2-7b``: its measured reversal
cost is -0.0013 nats per scored token, which is what an unadapted text decoder
on protein should look like -- no directional reading of sequence at all. It is
qualified so the ladder has a pre-adaptation reference, and a correlation it
returns is what these queues yield without a directional reading. It is never a
protein capability.

One rung per invocation, one artefact per rung, because this gate stops the rung
rather than the campaign. What each rung's failure costs the campaign is
EXP-R2-226's declaration and is applied where the ladder is assembled: Stage 2
failing leaves the ladder as base -> Stage 1, and Stage 1 failing stops the
campaign entirely because the base rung alone is not a ladder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.models import joint_lineage as L  # noqa: E402
from src.capability.core.arms import REPO  # noqa: E402
from src.capability.core.io import write_json  # noqa: E402

SCHEMA_VERSION = "r2_adaptation_stage_qualification_v1"
DEFAULT_OUT = REPO / "results/R1/adaptation_stage_qualification"

#: The campaign's precision, on every rung and at both fitness endpoints. The
#: released checkpoints declare float16, so this is a declared cast -- identical
#: on all three rungs, which is what the paired difference requires.
CAMPAIGN_DTYPE = "bfloat16"

#: Numerical determinism of one forward pass, in nats per target. EXP-R2-226's
#: number. Two runs of the same tokens through the same weights on the same
#: device must agree to this; anything larger is non-determinism the scoring
#: cannot be read through.
REPEAT_MAX_ABS = 1e-6

#: How much worse the reversed residue order must read, in nats per SCORED
#: TOKEN. EXP-R2-226's number, frozen before any score on this ladder existed.
#:
#: **Attainability is shown rather than assumed** (Appendix B rule 2). EXP-R2-152
#: measured this lineage's reversal cost at +0.1442 on Stage 1 and +0.1465 on
#: Stage 2 -- about three times this bar -- and -0.0013 on the unadapted base,
#: which is why the base rung is exempt rather than expected to clear it. The
#: unit is the token and not the residue because this family's trained protein
#: format has no per-residue alphabet to reach; see
#: :mod:`src.capability.models.joint_modes`.
REVERSAL_COST_MIN = 0.05


class ClauseFailure(Exception):
    """One named qualification clause did not hold.

    Its own exception type rather than a bare ``ValueError`` so that a refusal
    can be attributed to the clause that fired and written into a durable FAIL
    artefact -- EXP-R2-226 requires the failure to be reported with its clause --
    while a genuine defect anywhere else still propagates as itself.
    """

    def __init__(self, clause: str, reason: str) -> None:
        super().__init__(f"{clause}: {reason}")
        self.clause = clause
        self.reason = reason


def artefact_name(rung: str) -> str:
    return f"adaptation_stage_qualification_{rung}.json"


def sequence_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Probe:
    """The frozen probe, pinned by the digest of its own literal.

    Pinned so that an edit to the literal is a refusal rather than a quietly
    different measurement: the self-check's whole value is that it is the same
    sequence every time it is run.
    """

    label: str
    sequence: str
    sha256: str
    note: str


#: The one fixed sequence. EXP-R2-226 froze a *fixed-sequence* self-check and a
#: reversal control of that same rendering, so this is one probe and not a
#: cohort.
#:
#: **Natural, and deliberately not famous.** The first 80 residues of avGFP: a
#: natural sequence rather than an alphabet run, so the reading is a reading on
#: protein rather than on a pattern, and unremarkable enough that the reversal
#: control measures a directional reading rather than a retrieval. That second
#: property is not assumed. Measured on this repository's own staged
#: checkpoints at float32 on CPU before this file was frozen, the 76-residue
#: ubiquitin monomer -- the obvious alternative -- reads 0.855 nats per scored
#: token on ``ProLLaMA_Stage_1`` against 4.622 on the unadapted base, and its
#: reversal costs 3.65 nats. That is a sequence the adaptation's corpus
#: contains verbatim, and a reversal control taken on it would report
#: memorisation with the sign of a directional reading. avGFP's own reversal
#: cost on the same rungs is +0.135 and +0.094 against -0.025 on the base,
#: which is the shape EXP-R2-152 measured over a 128-record Swiss-Prot draw
#: (+0.1442, +0.1465, -0.0013) and is therefore a representative probe rather
#: than a retrieved one.
#:
#: The literal is the same 80 residues ``second_stage_interface_qualification.py``
#: uses. The literal is shared because it is a good natural probe; the
#: declaration is not, because each campaign's freeze stays where it was written.
PROBE = Probe(
    label="avgfp_n80",
    sequence=(
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQ"
    ),
    sha256="3a6033d2eb88fa724c2aab48cb19d903201b1c2efefda1b82b8edb83a03ebbcf",
    note="the first 80 residues of avGFP",
)


def require_probe() -> Probe:
    """Refuse a probe whose literal has drifted from its frozen digest.

    Not a :class:`ClauseFailure`: a drifted literal is a defect in this file, not
    a property of a checkpoint, and must not be recorded as a rung failing a gate.
    """

    measured = sequence_digest(PROBE.sequence)
    if measured != PROBE.sha256:
        raise ValueError(
            f"probe {PROBE.label} hashes to {measured}, not the frozen "
            f"{PROBE.sha256}; the self-check is only a check because the sequence "
            "is the same one every time"
        )
    return PROBE


def require_finite(values: Sequence[float], *, label: str) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise ClauseFailure(
                "nll_self_check", f"{label}: non-finite NLL at scored token {index}"
            )


def target_ids_digest(target_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in target_ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def repeat_max_abs_diff(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("the two NLL repeats have different lengths")
    if not first:
        raise ValueError("the NLL repeat is empty")
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def require_repeat(first: Sequence[float], second: Sequence[float]) -> float:
    require_finite(first, label="run 1")
    require_finite(second, label="run 2")
    delta = repeat_max_abs_diff(first, second)
    if delta > REPEAT_MAX_ABS:
        raise ClauseFailure(
            "nll_self_check",
            f"the NLL repeat's maximum absolute difference is {delta} nats, above "
            f"{REPEAT_MAX_ABS}; the scoring cannot be read through that much "
            "non-determinism",
        )
    return delta


def require_reversal_cost(cost: float) -> float:
    if not (cost > REVERSAL_COST_MIN):
        raise ClauseFailure(
            "directional_reversal",
            f"the directional-reversal cost is {cost} nats per scored token, which "
            f"is not strictly > {REVERSAL_COST_MIN}. This rung does not read the "
            "residue order under the rendering it is about to be scored under, so "
            "a correlation from it would not be a directional reading of sequence",
        )
    return cost


# ------------------------------------------------------------------- scoring


def scored_token_nll(loaded: L.LoadedRung, sequence: str) -> tuple[list[float], list[int]]:
    """Per-scored-token NLL of one sequence under the bare block, and its targets.

    Whole-string log-softmax and gather, then a restriction to the positions the
    rendering declares. The four wrapper tokens are not among them: the scored
    span is the one contiguous token run whose spellings are exactly the
    sequence, which is also what refuses a rendering where a residue merged into
    a delimiter.
    """

    import torch

    record = loaded.tokenisation.render(sequence)
    ids = torch.tensor([list(record.token_ids)], dtype=torch.long, device=loaded.model.device)
    with torch.no_grad():
        logits = loaded.model(input_ids=ids).logits
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
    positions = [int(position) for position in record.scored_positions]
    values = [float(nll[position - 1]) for position in positions]
    targets = [int(record.token_ids[position]) for position in positions]
    return values, targets


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("no scored token")
    return float(sum(float(value) for value in values) / len(values))


def qualify_loaded_rung(loaded: L.LoadedRung) -> dict[str, Any]:
    """Clauses 2 and 3 on a rung that has already cleared clause 1 by loading."""

    probe = require_probe()
    first, first_targets = scored_token_nll(loaded, probe.sequence)
    second, second_targets = scored_token_nll(loaded, probe.sequence)
    if first_targets != second_targets:
        raise ClauseFailure(
            "nll_self_check",
            f"{probe.label}: the two repeats scored different target ids, so they "
            "are not two readings of one measurement",
        )
    delta = require_repeat(first, second)
    native = mean(first)
    record: dict[str, Any] = {
        "nll_self_check": {
            "verdict": "PASS",
            "probe": probe.label,
            "probe_sha256": probe.sha256,
            "probe_note": probe.note,
            "n_residues": len(probe.sequence),
            "n_scored_tokens": len(first),
            "residues_per_scored_token": len(probe.sequence) / len(first),
            "target_ids_sha256": target_ids_digest(first_targets),
            "mean_nll_nats_per_scored_token": native,
            "repeat_max_abs_diff_nats": delta,
            "repeat_tolerance_nats": REPEAT_MAX_ABS,
            "repeats": 2,
            "note": (
                "run twice at the campaign precision on the same device. The target "
                "ids matched exactly, every value is finite, and the maximum "
                "per-target absolute difference is within the frozen tolerance"
            ),
        }
    }
    if not loaded.rung.reversal_control_applies:
        record["directional_reversal"] = {
            "verdict": "NOT_APPLICABLE",
            "reason": (
                "the base rung enters this ladder as a declared floor and not as a "
                "qualified protein arm. Its measured reversal cost is -0.0013 nats "
                "per scored token -- no directional reading of sequence at all -- "
                "which is what an unadapted text decoder on protein should look "
                "like, so this clause is deliberately not applied to it and a "
                "correlation it returns is never a protein capability"
            ),
            "floor_nats_per_scored_token": REVERSAL_COST_MIN,
        }
        return record
    values, _ = scored_token_nll(loaded, probe.sequence[::-1])
    require_finite(values, label=f"{probe.label} reversed")
    reversed_mean = mean(values)
    cost = require_reversal_cost(reversed_mean - native)
    record["directional_reversal"] = {
        "verdict": "PASS",
        "cost_nats_per_scored_token": cost,
        "floor_nats_per_scored_token": REVERSAL_COST_MIN,
        "native_mean_nats_per_scored_token": native,
        "reversed_mean_nats_per_scored_token": reversed_mean,
        "reversed_n_scored_tokens": len(values),
        "attainability_note": (
            "EXP-R2-152 measured +0.1442 on Stage 1 and +0.1465 on Stage 2 under "
            "this same bare rendering over a 128-record Swiss-Prot draw, about "
            "three times this floor, so this is not a gate its own positive "
            "controls cannot pass (Appendix B rule 2)"
        ),
    }
    return record


def _failure_payload(rung: str, failure: ClauseFailure) -> dict[str, Any]:
    """The durable record of a rung that did not qualify, and of what stopped it."""

    declaration = L.rung(rung)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "adaptation_stage_qualification",
        "campaign": "EXP-R2-226",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rung": rung,
        "training_stage": declaration.training_stage,
        "verdict": "FAIL",
        "failed_clause": failure.clause,
        "reason": failure.reason,
        "all_or_stop_note": (
            "this rung is not scored, no other checkpoint takes its slot, and the "
            "seed, the queue, the floor, the rendering and the checkpoint are not "
            "replaced to recover it. Stage 2 failing leaves the ladder as base -> "
            "Stage 1; Stage 1 failing stops the campaign, because the base rung "
            "alone is not a ladder"
        ),
    }


def qualify(rung: str, *, device: str, dtype: str) -> dict[str, Any]:
    """Every clause this gate can evaluate, on one rung, or a refusal."""

    declaration = L.rung(rung)
    try:
        loaded = L.load_rung(rung, device=device, dtype=dtype)
    except ValueError as failure:
        # The only ValueErrors load_rung raises are its strict-loading refusal
        # and its observed-dtype refusal, both of which are clause 1.
        raise ClauseFailure("strict_load", str(failure)) from failure
    clauses = qualify_loaded_rung(loaded)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "adaptation_stage_qualification",
        "campaign": "EXP-R2-226",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rung": rung,
        "training_stage": declaration.training_stage,
        "rung_note": declaration.note,
        "verdict": "PASS",
        "rendering": {
            "family": L.RENDERING_FAMILY,
            "mode": L.PROTEIN_MODE,
            "protein_context": None,
            "note": (
                "the bare Seq=<...> block. Stage 2's declared instruction form is "
                "deliberately not used on any rung: a true superfamily would be an "
                "a conditioning leak on the quantity being measured and a "
                "fabricated one would put a false fact inside the measurement"
            ),
        },
        "strict_load": {
            "verdict": "PASS",
            "diagnostics": loaded.facts["strict_loading_diagnostics"],
            "note": (
                "missing_keys, unexpected_keys, mismatched_keys and error_msgs are "
                "all empty; a newly initialised head is unavailable, never a pass (L24)"
            ),
        },
        "checkpoint_facts": loaded.facts,
        **clauses,
        "context_information_clause": {
            "verdict": "NOT_EVALUATED_HERE",
            "reason": (
                "EXP-R2-226 reuses rather than redraws this clause: the stage-21 "
                "readings at corrected seed 20260728, re-analysed under the "
                "EXP-R2-221 displacement-corrected rule, are the qualification. It "
                "needs no card and no model, and adaptation_stage_capability.py "
                "reads it off that artefact and refuses a ladder without it"
            ),
        },
        "all_or_stop_note": (
            "failure of any clause stops this rung and every later rung that would "
            "pair with it. The seed, the queue, the floor, the rendering and the "
            "checkpoint are not replaced to recover a rung"
        ),
    }


def read_verdict(directory: Path, rung: str, *, dtype: str) -> dict[str, Any]:
    """The qualification a fitness stage must find before it scores a rung.

    Imported by ``retrieval_bound.py`` and ``designed_referent.py`` so that
    "an arm is scored only after every clause holds" is executable rather than a
    sentence in a log entry. An absent artefact is a refusal and not a pass: a
    rung that was never qualified is exactly the state the all-or-stop rule
    exists to keep out of a scored run.

    ``dtype`` is checked because the clauses are properties of a *build*, not of
    a checkpoint: the self-check's determinism tolerance and the reversal cost
    were read at one precision, and a run scoring at another would be reading a
    build nothing qualified.
    """

    path = Path(directory) / artefact_name(rung)
    if not path.is_file():
        raise FileNotFoundError(
            f"{rung} carries no EXP-R2-226 qualification at {path}. Run "
            "adaptation_stage_qualification.py on this rung first; an unqualified "
            "rung is not scored"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact") != "adaptation_stage_qualification":
        raise ValueError(f"{path} is not an EXP-R2-226 qualification artefact")
    if payload.get("rung") != rung:
        raise ValueError(f"{path} qualifies {payload.get('rung')!r}, not {rung!r}")
    if payload.get("verdict") != "PASS":
        raise ValueError(
            f"{rung} did not qualify: {payload.get('verdict')!r}. A rung that "
            "failed a clause is not scored and no other checkpoint takes its slot"
        )
    qualified_at = payload["checkpoint_facts"]["dtype_requested"]
    if qualified_at != dtype:
        raise ValueError(
            f"{rung} was qualified at {qualified_at} and this run scores at "
            f"{dtype}. The clauses are properties of a build, so scoring at "
            "another precision would read a build nothing qualified"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", required=True, choices=list(L.LINEAGE_RUNGS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default=CAMPAIGN_DTYPE,
        choices=("bfloat16", "float16", "float32"),
        help="the campaign precision. bfloat16 is EXP-R2-226's declaration on "
        "every rung and at both fitness endpoints; another value qualifies a "
        "build the scoring will not use",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / artefact_name(args.rung)
    try:
        payload = qualify(args.rung, device=args.device, dtype=args.dtype)
    except ClauseFailure as failure:
        # Written before the nonzero exit, so the clause that fired survives in
        # the artefact and not only in a runtime log. The exit is still nonzero:
        # a rung that did not qualify is a defect an operator must see.
        write_json(destination, _failure_payload(args.rung, failure))
        print(f"[qualification] {args.rung} FAIL ({failure.clause}) -> {destination}")
        print(f"[qualification] {failure.reason}", file=sys.stderr)
        raise SystemExit(1)
    write_json(destination, payload)
    print(f"[qualification] {args.rung} {payload['verdict']} -> {destination}")
    reversal = payload["directional_reversal"]
    print(
        f"[qualification] reversal {reversal['verdict']} "
        f"{reversal.get('cost_nats_per_scored_token')}"
    )


if __name__ == "__main__":
    main()
