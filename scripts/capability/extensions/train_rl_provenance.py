#!/usr/bin/env python3
"""Train one arm of the reward-provenance experiment with GRPO.

Declared in ``docs/rl-environment-provenance.md`` and frozen in
``src/capability/extensions/rl_environment_provenance.py``. One invocation is
one arm: the policy is ProtGPT3-112M, the algorithm is a group-relative policy
gradient with a KL penalty to the frozen base, and the only thing that differs
between invocations is which reward is paid.

The environment arm's reward needs no fold at all; the memory-derived arm's is
the published DPO objective's own structural statistic under this experiment's
declared low-complexity mask, which is not the published pipeline's -- see the
registration under *What arm M does and does not share with the published
objective*.

Two of the rewards need a protein fold per completion, so the process
holds the policy, the frozen reference and ESMFold2 at once. That is why it runs
under the project's isolated folding runtime (``runtimes/esmfold2/bin/python``,
Transformers 5.x) rather than under ``ct``: the folding instrument is only
importable there, and the policy -- a Mixtral-style MoE -- loads and generates
there as well, which was verified before anything was trained. The environment
arm needs no fold and runs under either interpreter.

Resumption. The run checkpoints every declared interval into ``latest/``, which
is a directory the project's own generation lane can be pointed at, and the
rollout seed of step ``s`` is the declared base seed plus ``s``, so a resumed run
reproduces the rollout of the step it resumes at and no RNG blob is carried.
``--resume`` refuses a checkpoint whose recorded hyperparameters are not the
declared ones.

Usage:
    runtimes/esmfold2/bin/python scripts/capability/extensions/train_rl_provenance.py \
        --arm memory --out <run dir>/memory --device cuda:0 \
        --esmfold-model "$MODEL_ROOT/ESMFold2-hf"
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.extensions import grpo  # noqa: E402
from src.capability.extensions import rl_environment_provenance as rp  # noqa: E402
from src.capability.extensions import sequence_reward as sr  # noqa: E402
from src.capability.generation import objective_tuned_arms as ot  # noqa: E402
from src.capability.generation import unconditional_generation as ug  # noqa: E402

RUN_RECORD = "training_run.json"


def load_policy(checkpoint: Path, *, device: str, trainable: bool):
    """Load one copy of the policy in float32, under the declared prompt contract.

    Float32 rather than the checkpoint's stored bfloat16, for both the policy and
    the reference. The loss differentiates a ratio of log-probabilities and
    penalises a KL between two of them; bfloat16 rounding on those quantities is
    larger than the per-step movement being measured, and 109M parameters in
    float32 is under half a gigabyte.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32)
    model = model.to(device)
    if trainable:
        model.train()
        model.config.use_cache = True
    else:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, tokenizer


def interface_record(tokenizer) -> dict[str, Any]:
    """Refuse to train behind a prompt that is not the policy's own rendering."""

    prompt = ug.prompt_for(rp.POLICY_ARM)
    contract = ot.prompt_contract(rp.POLICY_ARM, tokenizer, prompt)
    alphabet = ot.residue_alphabet_contract(rp.POLICY_ARM, tokenizer)
    close = ug.close_token_for(rp.POLICY_ARM)
    eos_id = tokenizer.convert_tokens_to_ids(close)
    if eos_id is None or int(eos_id) < 0:
        raise SystemExit(f"close token {close!r} has no id in this tokenizer")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise SystemExit("the policy's tokenizer declares no pad token; generation fills on it")
    return {
        "prompt": prompt,
        "prompt_ids": contract["prompt_ids"],
        "declared_marker_ids": contract["declared_marker_ids"],
        "input_format": contract["input_format"],
        "close_token": close,
        "eos_id": int(eos_id),
        "pad_id": int(pad_id),
        "residue_ids": alphabet["residue_ids"],
    }


def decode_completions(tokenizer, completions, action_mask) -> list[str]:
    """The residue string the project's declared extractor reads out of each completion.

    The action span is decoded, not the padded row: everything after the first
    stop was never sampled. The extractor is the lane's own, so a completion that
    leaves the canonical alphabet contributes the shorter string it extracted to,
    exactly as it would in the generation ledger.
    """

    out = []
    for row, mask in zip(completions.tolist(), action_mask.tolist()):
        ids = [token for token, keep in zip(row, mask) if keep]
        text = tokenizer.decode(ids, skip_special_tokens=False)
        out.append(ug.extract_residues(rp.POLICY_ARM, text))
    return out


def reward_batch(arm: str, sequences: list[str], scorer) -> tuple[list[float], list[dict[str, Any]]]:
    """Rewards and per-completion records for one group.

    A completion that extracted to nothing is paid the declared empty-completion
    reward and is not folded. Every other completion that the arm's reward needs a
    structure for is folded, and a fold that fails is a failure of the run rather
    than a zero reward: paying zero for an unmeasurable completion teaches the
    policy to produce unmeasurable completions.
    """

    stats = [sr.statistics(sequence) for sequence in sequences]
    readings = (scorer.read_many(sequences) if scorer is not None
                else [None] * len(sequences))
    values: list[float] = []
    records: list[dict[str, Any]] = []
    for sequence, stat, reading in zip(sequences, stats, readings):
        plddt = None if reading is None else reading.mean_ca_plddt
        value = sr.reward(arm, stat, mean_ca_plddt=plddt)
        values.append(value)
        record = {"reward": value, **stat.as_record(),
                  "terms": sr.terms(stat, mean_ca_plddt=plddt),
                  "hits": sr.meets_declared_objective(stat, mean_ca_plddt=plddt)}
        if reading is not None:
            record["fold"] = reading.as_record()
        records.append(record)
    return values, records


def group_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """What a step keeps about its group, beyond the reward summary."""

    import numpy as np

    def mean(key: str) -> float:
        return float(np.mean([record[key] for record in records]))

    hit_keys = sorted({key for record in records for key in record["hits"]})
    term_keys = sorted({key for record in records for key in record["terms"]})
    out = {
        "length_mean": mean("length"),
        "net_charge_mean": mean("net_charge_ph7"),
        "low_complexity_fraction_mean": mean("low_complexity_fraction"),
        "distinct_5mer_fraction_mean": mean("distinct_5mer_fraction"),
        "longest_single_residue_run_mean": mean("longest_single_residue_run"),
        "composition_entropy_bits_mean": mean("composition_entropy_bits"),
        "empty_completions": int(sum(1 for record in records if record["length"] == 0)),
        # The means hide the tail, and the tail is where a structural reward can be
        # paid on something that is not a structure. These three make the thing the
        # structural-support rule exists for visible per step instead of inferable
        # only from a crash.
        "length_min": int(min(record["length"] for record in records)),
        "length_max": int(max(record["length"] for record in records)),
        "below_structure_support": int(sum(
            1 for record in records
            if 0 < record["length"] < sr.STRUCTURE_SUPPORT[0])),
        "term_means": {key: float(np.mean([record["terms"][key] for record in records
                                           if key in record["terms"]]))
                       for key in term_keys},
        "hit_rates": {key: float(np.mean([bool(record["hits"].get(key, False))
                                          for record in records]))
                      for key in hit_keys},
    }
    folded = [record["fold"] for record in records if "fold" in record]
    if folded:
        out["mean_ca_plddt_mean"] = float(np.mean([item["mean_ca_plddt"] for item in folded]))
        out["fold_seconds_total"] = float(sum(item["seconds"] for item in folded))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", required=True, choices=list(rp.TRAINED_ARMS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-checkpoint", type=Path, default=None,
                        help="None resolves the declared staged base")
    parser.add_argument("--esmfold-model", type=Path, default=None,
                        help="None resolves $MODEL_ROOT/ESMFold2-hf")
    parser.add_argument("--steps", type=int, default=rp.STEPS)
    parser.add_argument("--group-size", type=int, default=rp.GROUP_SIZE)
    # Calibration overrides. A run that uses one is recorded
    # ``is_scientific_measurement: false``: the declared values are the ones the
    # registration froze, and a run under other values is a pilot whatever it
    # produced.
    parser.add_argument("--learning-rate", type=float, default=rp.LEARNING_RATE)
    parser.add_argument("--kl-coefficient", type=float, default=rp.KL_COEFFICIENT)
    # Schedule, not experiment: the checkpoint intervals decide nothing about what
    # is learned, so they are overridable and leave the run scientific.
    parser.add_argument("--checkpoint-every", type=int, default=rp.CHECKPOINT_EVERY)
    parser.add_argument("--milestone-every", type=int, default=rp.MILESTONE_EVERY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-fold", action="store_true",
                        help="validation only: pay the closed-form terms alone and refuse "
                             "to write a scientific record")
    args = parser.parse_args(argv)

    import torch

    import dataclasses

    arm = rp.arm(args.arm)
    settings = dataclasses.replace(
        rp.settings(steps=args.steps, group_size=args.group_size),
        learning_rate=args.learning_rate,
        kl_coefficient=args.kl_coefficient,
        checkpoint_every=args.checkpoint_every,
        milestone_every=args.milestone_every,
    )
    scientific = (
        args.steps == rp.STEPS
        and args.group_size == rp.GROUP_SIZE
        and args.learning_rate == rp.LEARNING_RATE
        and args.kl_coefficient == rp.KL_COEFFICIENT
        and not args.no_fold
    )
    base = Path(args.base_checkpoint) if args.base_checkpoint else rp.policy_checkpoint()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latest = out / grpo.LATEST_DIR
    trajectory_path = out / grpo.TRAJECTORY

    resume_from = latest if (args.resume and (latest / grpo.TRAINER_STATE).exists()) else None
    weights = resume_from if resume_from is not None else base
    model, tokenizer = load_policy(weights, device=args.device, trainable=True)
    reference, _ = load_policy(base, device=args.device, trainable=False)
    interface = interface_record(tokenizer)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, betas=rp.ADAM_BETAS,
        weight_decay=rp.WEIGHT_DECAY,
    )
    start_step = 0
    if resume_from is not None:
        state = grpo.load_checkpoint(resume_from, model=model, optimiser=optimiser,
                                     map_location=args.device)
        grpo.require_same_settings(state["settings"], settings)
        start_step = int(state["step"]) + 1
        print(json.dumps({"resumed_at_step": start_step, "from": str(resume_from)}), flush=True)

    scorer = None
    if arm.needs_structure and not args.no_fold:
        from src.capability.extensions.plddt_scorer import PlddtScorer

        model_dir = Path(args.esmfold_model) if args.esmfold_model else rp.esmfold2_checkpoint()
        scorer = PlddtScorer(model_dir, device=args.device, realisation="A")
        print(json.dumps({"esmfold2": scorer.measurement_record()}), flush=True)
    elif arm.needs_structure:
        print("--no-fold: the memory term is absent, so this is a validation run only",
              flush=True)

    record = {
        "schema": rp.SCHEMA,
        "campaign": rp.CAMPAIGN,
        "stage": "train",
        "arm": arm.name,
        "reward": arm.reward,
        "reward_provenance": arm.provenance,
        "base_checkpoint": str(base),
        "resumed_from": None if resume_from is None else str(resume_from),
        "device": args.device,
        "gpu": torch.cuda.get_device_properties(args.device).name
        if args.device.startswith("cuda") else None,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings.as_record(),
        "interface": {key: value for key, value in interface.items() if key != "residue_ids"},
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "contract": rp.contract(),
        "folding": None if scorer is None else scorer.measurement_record(),
        "no_fold": bool(args.no_fold),
        "is_scientific_measurement": scientific,
    }
    if record["parameters"] != rp.POLICY_PARAMETERS:
        raise SystemExit(
            f"loaded {record['parameters']} parameters against the declared "
            f"{rp.POLICY_PARAMETERS}; refusing to train a checkpoint that is not the policy"
        )
    write_json(out / RUN_RECORD, record)

    prompt_ids = interface["prompt_ids"]
    for step in range(start_step, settings.steps):
        step_started = time.monotonic()
        sampled = grpo.rollout(
            model,
            prompt_ids=prompt_ids,
            group_size=settings.group_size,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.rollout_temperature,
            eos_id=interface["eos_id"],
            pad_id=interface["pad_id"],
            seed=settings.seed + step,
        )
        sequences = decode_completions(tokenizer, sampled["completions"], sampled["action_mask"])
        rewards, records = reward_batch(arm.reward, sequences, scorer)
        advantages = torch.tensor(grpo.group_advantages(rewards), device=args.device)
        input_ids = sampled["input_ids"]
        prompt_length = sampled["prompt_length"]
        action_mask = sampled["action_mask"]
        # Everything past the first stop is pad that ``generate`` wrote and the
        # policy never sampled. Causal attention means right-hand padding cannot
        # reach an earlier position, so masking it changes no log-probability in
        # the action span; it is masked anyway, because an attention mask that
        # claims a token the model never saw is a claim that will mislead the
        # next reader of this code.
        attention_mask = torch.cat(
            [torch.ones((input_ids.shape[0], prompt_length), dtype=action_mask.dtype,
                        device=action_mask.device), action_mask],
            dim=1,
        ).to(input_ids.dtype)
        with torch.no_grad():
            logp_old = grpo.token_log_probs(model, input_ids, attention_mask,
                                            prompt_length=prompt_length)
            logp_ref = grpo.token_log_probs(reference, input_ids, attention_mask,
                                            prompt_length=prompt_length)
        entropy = grpo.sequence_entropy(model, input_ids, attention_mask, action_mask,
                                        prompt_length=prompt_length)
        epochs = []
        for epoch in range(settings.inner_epochs):
            logp = grpo.token_log_probs(model, input_ids, attention_mask,
                                        prompt_length=prompt_length)
            loss, diagnostics = grpo.total_loss(
                logp, logp_old, logp_ref, advantages, action_mask,
                clip_epsilon=settings.clip_epsilon,
                kl_coefficient=settings.kl_coefficient,
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            diagnostics["gradient_norm"] = grpo.gradient_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimiser.step()
            diagnostics.update(epoch=epoch, loss=float(loss.item()))
            epochs.append(diagnostics)
        step_record = {
            "step": step,
            "rollout_seed": settings.seed + step,
            **grpo.group_summary(rewards),
            **group_record(records),
            "action_tokens": int(action_mask.sum().item()),
            "terminated": int(sum(sampled["terminated"])),
            "entropy_nats_per_token": entropy,
            "epochs": epochs,
            "seconds": time.monotonic() - step_started,
        }
        grpo.append_trajectory(trajectory_path, step_record)
        print(json.dumps({key: step_record[key] for key in
                          ("step", "reward_mean", "reward_sd", "reward_max",
                           "length_mean", "terminated", "entropy_nats_per_token",
                           "seconds")}), flush=True)
        if (step + 1) % settings.checkpoint_every == 0 or step + 1 == settings.steps:
            grpo.save_checkpoint(latest, model=model, tokenizer_source=base,
                                 optimiser=optimiser, step=step, settings=settings,
                                 extra={"arm": arm.name, "campaign": rp.CAMPAIGN})
        if (step + 1) % settings.milestone_every == 0 or step + 1 == settings.steps:
            grpo.save_checkpoint(out / f"step-{step + 1:05d}", model=model,
                                 tokenizer_source=base, optimiser=optimiser, step=step,
                                 settings=settings,
                                 extra={"arm": arm.name, "campaign": rp.CAMPAIGN})

    trajectory = grpo.read_trajectory(trajectory_path)
    record.update(
        finished_utc=datetime.now(timezone.utc).isoformat(),
        steps_recorded=len(trajectory),
        movement=grpo.optimisation_moved(trajectory, window=rp.TRAJECTORY_WINDOW),
        folding_usage=None if scorer is None else scorer.usage(),
    )
    write_json(out / RUN_RECORD, record)
    print(json.dumps({"arm": arm.name, "steps": len(trajectory),
                      "movement": record["movement"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
