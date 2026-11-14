"""Group-relative policy optimisation for one unconditional protein policy.

Written directly against PyTorch and ``transformers`` because a group-relative
policy gradient is a short, self-contained thing: a group of completions from one
prompt, their rewards centred within the group, a token-level policy-gradient
loss, and a KL penalty back to the frozen reference. There is no value network,
no reward model and no second policy, so there is nothing for an RL framework to
supply that is not spelled out below. ``trl``, ``peft`` and ``deepspeed`` are not
installed on either host and none of them is needed.

The experiment this serves is declared in ``docs/rl-environment-provenance.md``
and its hyperparameters are frozen in
:mod:`src.capability.extensions.rl_environment_provenance`. This module holds
the algorithm and the checkpoint format; it declares no reward, no arm and no
sample size.

One prompt, one group
---------------------
The policy is unconditional: its prompt is two tokens, the beginning-of-sequence
marker and the N-to-C direction token, and there is exactly one of them. So the
"group" GRPO averages its baseline over is the step's own set of completions, and
the group-relative baseline reduces to the step's mean reward. That reduction is
stated here rather than disguised by splitting one prompt's completions into
several nominal groups, which would only add noise to the baseline.

Why the advantage is centred and not standardised
-------------------------------------------------
:func:`group_advantages` subtracts the group mean and divides by nothing. A
division by the group's standard deviation rescales the step size by a quantity
that depends on the group, is undefined when every reward in the group is equal,
and gives the largest updates to the groups that discriminate least. Centring
alone is what makes the estimator group-relative; a group whose rewards are all
equal then contributes exactly zero, which is the right answer and is what
:func:`group_advantages` returns.

Why the loss is a token mean over the whole group
-------------------------------------------------
Length is a reward-relevant variable in this experiment -- one declared term is a
window on it -- so a loss that normalises each completion by its own token count
would weight short completions more per token and confound the length endpoint
with the optimiser. The loss therefore sums over every action token of every
completion in the group and divides by the total number of action tokens.

What an action is
-----------------
Every token the policy sampled, the terminating ``<|eos|>`` included where the
completion ended on one. The reward is a function of the residue string the
project's declared extractor reads out of the completion, so a completion that
wandered out of the residue alphabet earns the reward of the shorter string it
extracted to, and every token of that trajectory shares the advantage. That is
ordinary trajectory-level credit assignment, and it is what lets the policy learn
both what to emit and when to stop; excluding the stop token would leave the
length term unlearnable.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: Name of the resumable checkpoint a run overwrites, and of the directory it is
#: assembled in before the rename. A preemption during the assembly leaves the
#: previous ``latest`` untouched.
LATEST_DIR = "latest"
PENDING_DIR = "latest.pending"
#: The optimiser, step and history that ``save_pretrained`` does not carry.
TRAINER_STATE = "trainer_state.pt"
#: One JSON line per optimisation step, appended as the run goes.
TRAJECTORY = "trajectory.jsonl"

#: Settings a resume may change, because none of them decides what is learned:
#: the step budget and the two checkpoint intervals are a schedule.
RESUME_EXEMPT: frozenset[str] = frozenset({"steps", "checkpoint_every", "milestone_every"})


@dataclass(frozen=True)
class GRPOSettings:
    """Every hyperparameter of the algorithm, declared by the caller.

    Defaults are deliberately absent on the fields that decide what is learned:
    a run records these values, and a default that silently stood in for a
    declaration would be a hyperparameter nobody declared.
    """

    group_size: int
    steps: int
    inner_epochs: int
    clip_epsilon: float
    kl_coefficient: float
    learning_rate: float
    grad_clip: float
    max_new_tokens: int
    rollout_temperature: float
    seed: int
    checkpoint_every: int
    milestone_every: int

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("a group-relative baseline needs at least two completions")
        if self.steps < 1 or self.inner_epochs < 1:
            raise ValueError("steps and inner epochs are positive")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("the clipping range is a fraction strictly inside (0, 1)")
        if self.kl_coefficient < 0.0:
            raise ValueError("a KL penalty coefficient is non-negative")
        if not self.learning_rate > 0.0 or not self.grad_clip > 0.0:
            raise ValueError("the learning rate and gradient clip are positive")
        if self.max_new_tokens < 1:
            raise ValueError("a completion budget is positive")
        if not self.rollout_temperature > 0.0:
            raise ValueError("a rollout temperature is positive")
        if self.checkpoint_every < 1 or self.milestone_every < 1:
            raise ValueError("checkpoint intervals are positive")

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------- advantages

def group_advantages(rewards: Sequence[float]) -> np.ndarray:
    """Group-relative advantage: each reward minus the group's mean reward.

    A group in which every reward is equal -- which includes every group whose
    completions all missed a hard window, and every group of identical
    completions -- returns exactly zero for every member, so it contributes no
    policy gradient at all rather than an arbitrary direction scaled by a
    vanishing standard deviation.
    """

    values = np.asarray(rewards, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("a group advantage needs at least two rewards in one vector")
    if not np.isfinite(values).all():
        raise ValueError("a group advantage needs a finite reward on every completion")
    return values - values.mean()


def group_summary(rewards: Sequence[float]) -> dict[str, Any]:
    """What a step records about its group's rewards, degeneracy included."""

    values = np.asarray(rewards, dtype=float)
    spread = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return {
        "group_size": int(values.size),
        "reward_mean": float(values.mean()),
        "reward_sd": spread,
        "reward_min": float(values.min()),
        "reward_max": float(values.max()),
        "degenerate_group": bool(spread == 0.0),
    }


# ----------------------------------------------------------------- the loss

def _check_shapes(logp: Any, logp_old: Any, advantages: Any, action_mask: Any) -> None:
    if logp.shape != logp_old.shape or logp.shape != action_mask.shape:
        raise ValueError(
            f"token log-probabilities, their sampling-time values and the action "
            f"mask share one shape; got {tuple(logp.shape)}, "
            f"{tuple(logp_old.shape)} and {tuple(action_mask.shape)}"
        )
    if advantages.shape != logp.shape[:1]:
        raise ValueError(
            f"one advantage per completion; got {tuple(advantages.shape)} for "
            f"{logp.shape[0]} completions"
        )


def policy_gradient_loss(logp, logp_old, advantages, action_mask, *, clip_epsilon: float):
    """Token-level clipped surrogate, summed over the group and divided by its tokens.

    ``logp`` carries gradient; ``logp_old`` is the same quantity under the
    weights that sampled the group and is detached. On the first inner epoch the
    two are equal, the ratio is exactly one, the clip is inactive and this is the
    plain group-relative policy gradient. Later inner epochs reuse the same
    completions, which is why the ratio and the clip are here at all: the
    memory-derived reward costs a protein fold per completion, and spending that
    fold on one gradient step would waste almost all of it.

    Returns the loss and the diagnostics a step records: the mean ratio and the
    share of tokens the clip was active on.
    """

    import torch

    _check_shapes(logp, logp_old, advantages, action_mask)
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("the clipping range is a fraction strictly inside (0, 1)")
    mask = action_mask.to(logp.dtype)
    tokens = mask.sum()
    if float(tokens) <= 0:
        raise ValueError("a policy-gradient step needs at least one action token")
    ratio = torch.exp(logp - logp_old)
    advantage = advantages.to(logp.dtype).unsqueeze(-1)
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    loss = -(surrogate * mask).sum() / tokens
    with torch.no_grad():
        active = ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).to(logp.dtype)
        diagnostics = {
            "action_tokens": int(tokens.item()),
            "ratio_mean": float((ratio * mask).sum().item() / tokens.item()),
            "clip_fraction": float((active * mask).sum().item() / tokens.item()),
        }
    return loss, diagnostics


def reference_kl(logp, logp_ref, action_mask):
    """Token-mean KL from the policy to the frozen reference, k3 estimator.

    ``exp(d) - d - 1`` with ``d = log pi_ref - log pi`` is non-negative for every
    token, is zero exactly where the two agree, and has the KL as its expectation
    under the policy. Added to the loss with a positive coefficient it penalises
    divergence in either direction, which a plain ``log pi_ref - log pi`` -- whose
    per-token value is signed and whose mean can be driven negative -- does not.
    """

    import torch

    if logp.shape != logp_ref.shape or logp.shape != action_mask.shape:
        raise ValueError("a KL term needs matching log-probability and mask shapes")
    mask = action_mask.to(logp.dtype)
    tokens = mask.sum()
    if float(tokens) <= 0:
        raise ValueError("a KL term needs at least one action token")
    delta = logp_ref - logp
    per_token = torch.exp(delta) - delta - 1.0
    return (per_token * mask).sum() / tokens


def total_loss(logp, logp_old, logp_ref, advantages, action_mask, *,
               clip_epsilon: float, kl_coefficient: float):
    """The declared objective: clipped surrogate plus the KL penalty."""

    gradient, diagnostics = policy_gradient_loss(
        logp, logp_old, advantages, action_mask, clip_epsilon=clip_epsilon
    )
    kl = reference_kl(logp, logp_ref, action_mask)
    diagnostics.update(policy_gradient_loss_value=float(gradient.item()),
                       reference_kl=float(kl.item()))
    return gradient + kl_coefficient * kl, diagnostics


# ------------------------------------------------------- rollouts and scoring

def rollout(model, *, prompt_ids: Sequence[int], group_size: int, max_new_tokens: int,
            temperature: float, eos_id: int, pad_id: int, seed: int) -> dict[str, Any]:
    """Sample one group of completions from the policy, with their action mask.

    The rollout is drawn from the policy itself: temperature as declared, and no
    top-k or top-p truncation. A truncated decoder is a different distribution
    from the one whose log-probabilities the loss differentiates, and a policy
    gradient taken under one distribution and scored under another is biased. The
    project's frozen generation operating point, which does truncate, is what the
    *evaluation* pools use, where comparability with the existing base pools is
    what matters rather than gradient correctness.

    The step's seed is ``base seed + step index``, set on the global generator
    the way the project's own forward-decoder does, so a resumed run reproduces
    the rollout of the step it resumes at without carrying an RNG blob.
    """

    import torch

    if group_size < 2:
        raise ValueError("a group-relative baseline needs at least two completions")
    device = next(model.parameters()).device
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    prompt = prompt.expand(group_size, -1)
    prompt_length = prompt.shape[1]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=True,
            temperature=float(temperature),
            top_k=0,
            top_p=1.0,
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=1,
            eos_token_id=int(eos_id),
            pad_token_id=int(pad_id),
            use_cache=True,
        )
    if was_training:
        model.train()
    completions = generated[:, prompt_length:]
    mask = _action_mask(completions, eos_id=eos_id)
    return {
        "input_ids": generated,
        "completions": completions,
        "action_mask": mask,
        "prompt_length": prompt_length,
        "terminated": [bool(row.any()) for row in (completions == eos_id)],
        "action_lengths": [int(value) for value in mask.sum(dim=1).tolist()],
    }


def _action_mask(completions, *, eos_id: int):
    """Tokens the policy actually chose: up to and including the first stop.

    ``generate`` fills the tail of a terminated row with the pad token. Those
    positions were never sampled and must not enter the loss, while the stop
    token itself was sampled and must.

    Every row therefore carries at least one action, including a row that stopped
    immediately: emitting the stop as the first token is a choice the policy made,
    it earns the declared empty-completion reward, and the gradient has to reach
    it or the policy could never unlearn stopping at once.
    """

    import torch

    is_eos = completions == eos_id
    # cumulative count of stops strictly before each position
    before = torch.cumsum(is_eos.to(torch.long), dim=1) - is_eos.to(torch.long)
    return before == 0


def token_log_probs(model, input_ids, attention_mask, *, prompt_length: int):
    """Log-probability the model assigns to each generated token actually taken.

    One forward pass over prompt and completion. The logit at position ``t``
    predicts token ``t + 1``, so the span scoring the completion starts at
    ``prompt_length - 1``. The log-softmax is taken in float32 whatever the
    forward dtype, because a ratio of two log-probabilities is the quantity the
    loss differentiates and bf16 rounding there is larger than the updates being
    measured.
    """

    import torch

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits if hasattr(output, "logits") else output[0]
    span = logits[:, prompt_length - 1 : -1, :].float()
    targets = input_ids[:, prompt_length:]
    if span.shape[1] != targets.shape[1]:
        raise ValueError(
            f"scoring span of {span.shape[1]} positions against {targets.shape[1]} "
            "generated tokens"
        )
    return torch.log_softmax(span, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def token_entropies(model, input_ids, attention_mask, *, prompt_length: int):
    """Predictive entropy of each generated position, in nats, as a (batch, tokens) tensor.

    The entropy of the policy's own next-token distribution at each step of the
    completion, which is a property of the model and not of the token it happened
    to sample. One forward pass, log-softmax in float32 for the same reason
    :func:`token_log_probs` takes it there.
    """

    import torch

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits if hasattr(output, "logits") else output[0]
        span = logits[:, prompt_length - 1 : -1, :].float()
        logp = torch.log_softmax(span, dim=-1)
        return -(logp.exp() * logp).sum(dim=-1)


def sequence_entropy(model, input_ids, attention_mask, action_mask, *, prompt_length: int) -> float:
    """Token-mean predictive entropy over a group's whole action span, in nats.

    Reported rather than optimised. The entropy change between base and policy is
    one of this experiment's declared endpoints, so an entropy bonus in the loss
    would be optimising an endpoint and there is none.
    """

    per_token = token_entropies(model, input_ids, attention_mask, prompt_length=prompt_length)
    mask = action_mask.to(per_token.dtype)
    return float((per_token * mask).sum().item() / max(float(mask.sum().item()), 1.0))


def per_sequence_entropy(model, input_ids, attention_mask, action_mask, *,
                         prompt_length: int) -> list[float]:
    """Each completion's own token-mean predictive entropy, in nats.

    The group-level mean cannot serve a resampling family: the declared
    resampling unit is the sequence, so the entropy endpoint needs one value per
    sequence rather than one per group.
    """

    per_token = token_entropies(model, input_ids, attention_mask, prompt_length=prompt_length)
    mask = action_mask.to(per_token.dtype)
    totals = (per_token * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return [float(value) for value in (totals / counts)]


# ------------------------------------------------------------- checkpointing

#: Files a checkpoint carries verbatim from the base rather than re-serialised.
#: Training changes weights; it does not change what model this is or how its
#: text is tokenised, so the base's own declaration of both is what a trained
#: checkpoint must state.
#:
#: This is a correctness requirement and not tidiness. The two interpreters this
#: experiment spans do not write those declarations the same way. The folding
#: runtime's Transformers 5.x records a tokenizer backend class name the
#: validated 4.x interpreter refuses to load, and it moves the rotary base out
#: of ``rope_theta`` into a ``rope_parameters`` block that 4.x does not read --
#: so a 5.x-written config loaded under 4.x would silently fall back to the
#: default rotary base and sample from a different model than the one that was
#: trained. The evaluation lane runs under 4.x. Copying the base's files keeps a
#: trained checkpoint architecture- and tokenizer-identical to the base it came
#: from, which is what a paired contrast needs anyway, and keeps it loadable by
#: both.
BASE_DECLARATION_FILES: tuple[str, ...] = (
    "config.json", "generation_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json",
)

#: The precision field the copied config has to be corrected on, because it is
#: the one thing about the checkpoint that training *does* change: the base is
#: serialised bfloat16 and a trained checkpoint carries the float32 parameters
#: the loss was differentiated in. Written under the 4.x spelling, which 5.x
#: also reads.
STORED_PRECISION_FIELD = "torch_dtype"


def save_checkpoint(directory: Path, *, model, tokenizer_source: Path, optimiser,
                    step: int, settings: GRPOSettings,
                    extra: dict[str, Any] | None = None) -> Path:
    """Write a resumable checkpoint that is also a loadable checkpoint directory.

    The weights go through ``save_pretrained`` and the tokenizer is copied from
    ``tokenizer_source``, so the result is a directory the project's own
    generation lane can be pointed at with ``--checkpoint`` and load under its
    declared shape check. The optimiser state and the step index go beside them.
    There is no RNG blob: the rollout seed is a declared function of the step, so
    resuming at a step reproduces that step's rollout.

    The write is assembled in a sibling directory and renamed into place, because
    a preemption in the middle of a 400 MB write would otherwise leave the only
    resumable checkpoint truncated.
    """

    import torch

    directory = Path(directory)
    source = Path(tokenizer_source)
    pending = directory.parent / f"{directory.name}.pending"
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    model.save_pretrained(pending, safe_serialization=True)
    copied = [name for name in BASE_DECLARATION_FILES if (source / name).is_file()]
    if "config.json" not in copied:
        raise ValueError(
            f"{source} carries no config.json; a checkpoint whose architecture is "
            "not declared by the base it was trained from cannot be read as that base"
        )
    if "tokenizer.json" not in copied and "vocab.json" not in copied:
        raise ValueError(
            f"{source} carries no tokenizer file to copy into the checkpoint; a "
            "checkpoint without one cannot be read by the generation lane"
        )
    for name in copied:
        shutil.copy2(source / name, pending / name)
    declaration = json.loads((pending / "config.json").read_text(encoding="utf-8"))
    declaration[STORED_PRECISION_FIELD] = str(
        next(model.parameters()).dtype
    ).removeprefix("torch.")
    (pending / "config.json").write_text(
        json.dumps(declaration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "step": int(step),
            "optimiser": optimiser.state_dict(),
            "settings": settings.as_record(),
            "tokenizer_source": str(source),
            "tokenizer_files": copied,
            "extra": dict(extra or {}),
        },
        pending / TRAINER_STATE,
    )
    if directory.exists():
        previous = directory.parent / f"{directory.name}.previous"
        if previous.exists():
            shutil.rmtree(previous)
        os.replace(directory, previous)
        os.replace(pending, directory)
        shutil.rmtree(previous)
    else:
        os.replace(pending, directory)
    return directory


def load_checkpoint(directory: Path, *, model, optimiser, map_location: str = "cpu") -> dict[str, Any]:
    """Restore a checkpoint's optimiser state and step into a loaded model.

    The weights are expected to have been loaded from this same directory by the
    caller, which is what makes the checkpoint one object rather than two: a
    state file whose weights came from somewhere else would resume a different
    run. The settings the checkpoint recorded are returned so the caller can
    refuse a resume under changed hyperparameters rather than continuing a run
    that is no longer the declared one.
    """

    import torch

    state = torch.load(Path(directory) / TRAINER_STATE, map_location=map_location,
                       weights_only=False)
    optimiser.load_state_dict(state["optimiser"])
    return state


def require_same_settings(recorded: dict[str, Any], settings: GRPOSettings) -> None:
    """Refuse a resume whose hyperparameters differ from the checkpoint's.

    A run that changes its learning rate, group size or KL coefficient halfway is
    not the run that was declared, and a trajectory that silently spans two
    settings cannot be read as one. :data:`RESUME_EXEMPT` names the three fields
    that decide nothing about what is learned -- the step budget and the two
    checkpoint intervals -- so changing one of them is a change to the schedule
    and not to the experiment.
    """

    declared = settings.as_record()
    differing = {
        key: (recorded.get(key), declared[key])
        for key in declared
        if key not in RESUME_EXEMPT and recorded.get(key) != declared[key]
    }
    if differing:
        raise ValueError(
            "refusing to resume under changed hyperparameters: "
            + ", ".join(f"{key} was {was!r}, now {now!r}" for key, (was, now) in differing.items())
        )


def append_trajectory(path: Path, record: dict[str, Any]) -> None:
    """Append one step's record, flushed, so a preempted run keeps its history."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_trajectory(path: Path) -> list[dict[str, Any]]:
    """Every step record a run has written, in order, duplicates of a resumed step removed.

    A preemption between a checkpoint and the next checkpoint leaves steps in the
    trajectory whose weights were never saved; those steps run again on resume and
    append a second record. The later record is the one that belongs to the run
    that continued, so it wins.
    """

    path = Path(path)
    if not path.exists():
        return []
    by_step: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            by_step[int(record["step"])] = record
    return [by_step[step] for step in sorted(by_step)]


def optimisation_moved(trajectory: Sequence[dict[str, Any]], *, window: int) -> dict[str, Any]:
    """Whether the mean reward of the last ``window`` steps exceeds the first ``window``.

    A descriptive readout for the run record, not the experiment's decision rule:
    whether optimisation happened is decided on the evaluation pools under the
    declared band, not on the training trajectory, which is a sequence of
    different sample sizes from a moving distribution.
    """

    rewards = [float(record["reward_mean"]) for record in trajectory]
    if len(rewards) < 2 * int(window):
        return {"measurable": False,
                "reason": f"a {window}-step comparison needs {2 * window} steps, has {len(rewards)}"}
    first = float(np.mean(rewards[:window]))
    last = float(np.mean(rewards[-window:]))
    return {"measurable": True, "window": int(window), "first_window_mean": first,
            "last_window_mean": last, "change": last - first,
            "steps": len(rewards)}


def gradient_norm(model) -> float:
    """Total gradient norm over the parameters that carry one, before clipping."""

    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm().item()) ** 2
    return math.sqrt(total)
