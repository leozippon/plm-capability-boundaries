"""The reward-provenance experiment: arms, sizes, families, seeds, decision rules.

Declared in ``docs/rl-environment-provenance.md``, which is the registration;
this module is the machine-readable half of it and the single source for every
constant a producer or an analysis reads. Nothing here fits, samples or scores
anything. The reward functions are in
:mod:`src.capability.extensions.sequence_reward`, the algorithm in
:mod:`src.capability.extensions.grpo`, the folding instrument in
:mod:`src.capability.extensions.plddt_scorer`, and the estimators in
:mod:`src.capability.extensions.rl_provenance_analysis`.

What the experiment decides
---------------------------
The project's thesis holds that pre-training stores an average -- evolutionary
memory -- and that a bias toward an objective is installed only by post-training
against an environment. Protein design has no simulated environment: every
reward available to a protein policy is a learned proxy trained on the same
evolutionary corpus. The thesis therefore predicts that reinforcement learning
on a protein language model cannot install a genuine design bias, and that any
apparent bias is pre-training memory redistributed into a narrower region.

That prediction has never been tested, because no public checkpoint was trained
with online RL and this project had trained nothing. The evidence so far is
evaluative: released DPO checkpoints whose gains sit in the high-identity strata
and are reproduced 0.78 to 1.18 times over by an explicit filter on the base
pool. An evaluation of someone else's optimisation cannot separate "RL cannot
install a bias" from "that recipe did not", and it cannot vary the one thing the
thesis is about, which is where the reward came from.

So this experiment trains policies from one base, under rewards that differ only
in provenance, and asks of each whether the gain was reachable by rejection
sampling from the base distribution. The decisive quantity is ``k*``:
the smallest number of base samples that, with selection under the same reward,
matches the trained policy's mean reward. A small ``k*`` means the optimisation
redistributed memory the base already held. A large one means it did not.

The arms, and the one difference between them
---------------------------------------------
Identical base, prompt, algorithm, hyperparameters, seeds and budgets. The
reward is the only thing that changes, and what changes about it is where it
comes from: a learned predictor trained on the policy's own corpus, a closed-form
target no learned model touches, or both. The low-complexity constraint is in
every arm -- a constraint at its limit is part of the published objective, and it
is what forbids a poly-lysine solution to a charge window -- so it cannot explain
a difference between them. Being shared is the only property of it this
experiment depends on: the mask itself is this project's own operationalisation
and not the published pipeline's, which
:data:`~.sequence_reward.LOW_COMPLEXITY_PROVENANCE` states.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..core.arms import MODEL_ROOT
from ..core.statistics import (
    BOUNDED_NEGATIVE,
    FINDING_STATES,
    MINIMUM_BOOTSTRAP_UNITS,
    NON_IDENTIFIABLE,
    SUPPORTED,
    UNRESOLVED,
)
from ..generation import objective_tuned_arms as ot
from . import sequence_reward as sr

#: The run identity every artefact carries. The date of the declaration, not a
#: new E-number: ``docs/rl-environment-provenance.md`` is the registration.
CAMPAIGN = "rl_environment_provenance_20261114"

#: The stable schema label of every record this experiment writes.
SCHEMA = "rl_environment_provenance"

# ------------------------------------------------------------------ the policy

#: The base policy, and the arm name its prompt contract, extractor, loader and
#: attempt ledger are already declared under. The policy is not a new arm: it is
#: this checkpoint's weights, and every trained policy is a *checkpoint* of it,
#: reached through the generation lane's declared ``--checkpoint`` override under
#: that lane's own shape check. Declaring new arm names for the trained weights
#: would duplicate a prompt contract that is identical by construction, and
#: would put this experiment's checkpoints inside the objective-alignment
#: campaign's frozen pair table, which is that campaign's own single source.
POLICY_ARM = "protgpt3-112m"

#: The released DPO sibling of the same base, aligned against the structural
#: statistic arm ``memory`` is paid on. It is evaluated, never trained.
#:
#: The comparison it supports is a comparison of **pLDDT objectives with
#: differing low-complexity constraints**, not of identical objectives: the
#: published alignment's mask is not stated anywhere this project can read, arm
#: M's mask is this experiment's own, and only the 0.25 limit and the pLDDT
#: statistic are common to both. It is still the closest available comparison --
#: one base, one structural statistic, one released checkpoint aligned against it
#: -- and it belongs in every pool comparison for that reason, described as what
#: it is.
PUBLISHED_DPO_ARM = "protgpt3-112m-dpo"

#: Parameter count of the base, measured on the staged checkpoint. Recorded
#: because it is the fact that makes this experiment affordable: an MoE of this
#: size trains and folds inside one card's allocation.
POLICY_PARAMETERS = 109_125_120


def policy_checkpoint() -> Path:
    """The staged base weights, resolved through the declaration that owns them."""

    return ot.spec_for(POLICY_ARM).path


def published_dpo_checkpoint() -> Path:
    return ot.spec_for(PUBLISHED_DPO_ARM).path


#: The staged ESMFold2 checkpoint, under the same model root every other staged
#: instrument resolves from.
ESMFOLD2_DIRECTORY_NAME = "ESMFold2-hf"


def esmfold2_checkpoint(model_root: Path | None = None) -> Path:
    return Path(model_root or MODEL_ROOT) / ESMFOLD2_DIRECTORY_NAME


def prediction_spec(pool: str, checkpoint: Path):
    """The arm declaration the prediction side-effect scores one pool's weights under.

    This is the hook into the existing off-roster route. That route resolves an
    arm by name through ``objective_tuned_arms.spec_for`` and loads it with
    ``load_arm_spec``, which verifies the declared depth and width against the
    live config of the directory it is given -- so a trained checkpoint of this
    policy is scoreable there as the policy's own declaration with its path
    replaced, and nothing about the rendering, the packing family or the marker
    ids changes, because training changed none of them.

    What it does *not* do is add a name to that campaign's declarations. Its
    admitted list is frozen in ``configs/objective_tuned_prediction.json`` and
    its pair table is the single source of what that campaign measures; adding
    this experiment's weights to either is an amendment to its registration and
    not something this one may make. The route's producer takes an arm *name*,
    so scoring these weights through it needs that producer to accept a
    checkpoint override alongside the name -- honoured or refused, never
    ignored, which is the rule the generation lane already applies for exactly
    this failure. That change is reported rather than made here.
    """

    import dataclasses

    if pool not in EVALUATED_POOLS:
        raise KeyError(f"unknown pool {pool!r}; declared: {list(EVALUATED_POOLS)}")
    directory = Path(checkpoint)
    if not (directory / "config.json").is_file():
        raise FileNotFoundError(
            f"{directory} declares no config.json, so its architecture cannot be "
            "verified against the policy's declaration"
        )
    return dataclasses.replace(ot.spec_for(POLICY_ARM), path=directory)


# -------------------------------------------------------------------- the arms

ARM_MEMORY = sr.REWARD_MEMORY
ARM_ENVIRONMENT = sr.REWARD_ENVIRONMENT
ARM_BOTH = sr.REWARD_BOTH
ARM_ENVIRONMENT_HARD = sr.REWARD_ENVIRONMENT_HARD

#: Trained arms, in declared order. One arm per reward, and the reward's name is
#: the arm's name: the reward *is* the arm.
#:
#: ``environment_hard`` is the positive control, added by the amendment of
#: 2026-11-14 after the first three arms were launched and before any evaluation
#: pool was read. It widens every declared family from four columns to five, and
#: the best-of-k families from forty to fifty; a wider band makes every positive
#: reading harder to reach, including the ones this project prefers, which is the
#: direction an added arm should push.
TRAINED_ARMS: tuple[str, ...] = (ARM_MEMORY, ARM_ENVIRONMENT, ARM_BOTH,
                                 ARM_ENVIRONMENT_HARD)

#: Every pool the analysis reads, in declared order. ``base`` is the untrained
#: policy and is the control for all of them; ``published_dpo`` is the released
#: checkpoint. The trained arms carry the name of the reward they were trained
#: against.
POOL_BASE = "base"
POOL_PUBLISHED_DPO = "published_dpo"
EVALUATED_POOLS: tuple[str, ...] = (POOL_BASE, *TRAINED_ARMS, POOL_PUBLISHED_DPO)

#: The pools a contrast is formed for: everything except the control itself.
CONTRAST_POOLS: tuple[str, ...] = (*TRAINED_ARMS, POOL_PUBLISHED_DPO)


@dataclass(frozen=True)
class Arm:
    """One trained arm: its reward, that reward's provenance, and what it tests."""

    name: str
    reward: str
    needs_structure: bool
    provenance: str
    question: str


ARMS: tuple[Arm, ...] = (
    Arm(
        name=ARM_MEMORY,
        reward=ARM_MEMORY,
        needs_structure=True,
        provenance=sr.REWARD_PROVENANCE[sr.REWARD_MEMORY],
        question=(
            "the published DPO objective's own pLDDT statistic, trained online "
            "as a continuous reward instead of as a binary preference over a "
            "frozen sample, and under this experiment's own low-complexity mask "
            "at that objective's 0.25 limit rather than under the published "
            "pipeline's mask, which is not stated anywhere this project can "
            "read. What is shared is the statistic and the limit; what is not is "
            "the mask. Its reward is computed by a model trained on the same "
            "evolutionary record as the policy, so the thesis predicts the gain "
            "is reachable by selection over the base pool and is therefore a "
            "redistribution"
        ),
    ),
    Arm(
        name=ARM_ENVIRONMENT,
        reward=ARM_ENVIRONMENT,
        needs_structure=False,
        provenance=sr.REWARD_PROVENANCE[sr.REWARD_ENVIRONMENT],
        question=(
            "a miniature simulated environment: the reward is ground truth "
            "rather than a proxy, and no learned model enters it. If a bias can "
            "be installed at all by post-training, it can be installed here, and "
            "if it cannot be installed here then the obstacle is the policy "
            "rather than the reward's provenance"
        ),
    ),
    Arm(
        name=ARM_ENVIRONMENT_HARD,
        reward=ARM_ENVIRONMENT_HARD,
        needs_structure=False,
        provenance=sr.REWARD_PROVENANCE[sr.REWARD_ENVIRONMENT_HARD],
        question=(
            "the positive control. A small k* on the declared environment target "
            "admits two readings that predict the same number -- reinforcement "
            "learning installs nothing at this scale even against a closed-form "
            "reward, or that target was reachable by selection from the base all "
            "along -- and only a target selection demonstrably cannot reach "
            "separates them. Its window is chosen by one declared rule applied "
            "once to the base pool's measured conjunction rates, and written down "
            "before this arm trains"
        ),
    ),
    Arm(
        name=ARM_BOTH,
        reward=ARM_BOTH,
        needs_structure=True,
        provenance=sr.REWARD_PROVENANCE[sr.REWARD_BOTH],
        question=(
            "whether an orthogonal bias can be installed without destroying the "
            "memory-derived score, which is the question a design campaign "
            "actually faces"
        ),
    ),
)

ARM_BY_NAME: Mapping[str, Arm] = {arm.name: arm for arm in ARMS}


def arm(name: str) -> Arm:
    if name not in ARM_BY_NAME:
        raise KeyError(f"unknown arm {name!r}; declared: {list(TRAINED_ARMS)}")
    return ARM_BY_NAME[name]


# --------------------------------------------------------- the frozen settings

#: Rollout length budget, in new tokens. It is the cap every pool -- base,
#: trained and published -- is generated under, so the arms' length
#: distributions are comparable and the declared length window is reachable
#: from below and exceedable from above. Prompt plus completion stays well
#: inside the base's declared ``max_position_embeddings`` of 1025.
MAX_NEW_TOKENS = 200

#: Completions per optimisation step. The policy is unconditional, so there is
#: exactly one prompt and the group is the step's own sample set; this is the
#: group size of the group-relative baseline, stated as such rather than
#: disguised as several nominal groups of one prompt.
GROUP_SIZE = 48

#: Optimisation steps per arm. Equal across arms by declaration, so the arms
#: differ in their reward and not in how long they were trained.
STEPS = 320

#: Gradient passes over each sampled group. Above one, the clipped surrogate's
#: importance ratio is what makes the reuse a correct policy gradient. It is
#: here because the memory-derived reward costs a protein fold per completion,
#: and a single gradient step per fold would spend the experiment's whole
#: compute budget on 320 updates.
INNER_EPOCHS = 4

CLIP_EPSILON = 0.2
KL_COEFFICIENT = 0.02
LEARNING_RATE = 2e-6
ADAM_BETAS: tuple[float, float] = (0.9, 0.95)
WEIGHT_DECAY = 0.0
GRADIENT_CLIP = 1.0

#: Rollout temperature. One, with no top-k or top-p truncation: the gradient is
#: differentiated under the distribution the sample came from, and a truncated
#: decoder is a different distribution. The evaluation pools use the project's
#: frozen operating point instead, where comparability with the existing base
#: pools is what matters.
ROLLOUT_TEMPERATURE = 1.0

#: Base rollout seed. The seed of step ``s`` is this plus ``s``, so a resumed run
#: reproduces the rollout of the step it resumes at and no RNG state is carried
#: in a checkpoint.
TRAINING_SEED = 20261114

CHECKPOINT_EVERY = 10
MILESTONE_EVERY = 80

#: Steps averaged at each end of a trajectory for the descriptive movement
#: readout. The decision is taken on the evaluation pools, never on this.
TRAJECTORY_WINDOW = 20

HYPERPARAMETER_NOTE = (
    "the learning rate, KL coefficient, group size and step budget were fixed on "
    "a short local pilot of the environment arm, whose reward costs nothing to "
    "compute, before this declaration was written and before any scale run "
    "began. The pilot's only purpose was to establish that the optimiser moves "
    "at all: an experiment whose null reading could be 'the learning rate was "
    "too small' decides nothing, and calibrating against that failure runs "
    "against this project's own thesis rather than for it. The pilot is recorded "
    "as calibration and is not a result; every scale run starts fresh from the "
    "base checkpoint under exactly these values"
)


def settings(steps: int = STEPS, group_size: int = GROUP_SIZE):
    """The declared algorithm settings, as the trainer's own frozen object."""

    from .grpo import GRPOSettings

    return GRPOSettings(
        group_size=int(group_size),
        steps=int(steps),
        inner_epochs=INNER_EPOCHS,
        clip_epsilon=CLIP_EPSILON,
        kl_coefficient=KL_COEFFICIENT,
        learning_rate=LEARNING_RATE,
        grad_clip=GRADIENT_CLIP,
        max_new_tokens=MAX_NEW_TOKENS,
        rollout_temperature=ROLLOUT_TEMPERATURE,
        seed=TRAINING_SEED,
        checkpoint_every=CHECKPOINT_EVERY,
        milestone_every=MILESTONE_EVERY,
    )


def reward_calls(steps: int = STEPS, group_size: int = GROUP_SIZE) -> int:
    """Reward evaluations one arm's training consumes.

    Declared because it is the reference the compute-matched reading of ``k*``
    is taken against: a gain that selection reaches with three orders of
    magnitude fewer reward calls was not learned.
    """

    return int(steps) * int(group_size)


# ------------------------------------------------------------ evaluation pools

#: Independent generation streams per pool, with declared non-overlapping seeds.
#: Three, as the objective-alignment campaign's lane declares, so that the
#: resampling floor can be enforced per (pool, stream) and a stream-equal pool
#: mean is available.
STREAMS = 3

#: Attempts per stream of the base pool, and of every other pool. The base pool
#: is larger because the best-of-k control selects from it: the deepest declared
#: ``k`` is a sixth of one base stream, so the order statistic the control reads
#: is estimated from the pool rather than from its own tail alone.
BASE_ATTEMPTS_PER_STREAM = 1024
POOL_ATTEMPTS_PER_STREAM = 512

#: Declared stream seeds, non-overlapping across pools and streams. Each pool's
#: block is ``POOL_SEED_BASE + 1000 * pool index + stream index``, and the lane's
#: own ``seed_plus_batch_index`` rule then separates the batches inside a stream.
#: The block is a thousand wide so that a stream's batch seeds, which run from
#: the stream seed upward, cannot reach the next stream's.
POOL_SEED_BASE = 20261114_000


def pool_seeds(pool: str) -> tuple[int, ...]:
    """The declared generation seeds of one pool's streams."""

    if pool not in EVALUATED_POOLS:
        raise KeyError(f"unknown pool {pool!r}; declared: {list(EVALUATED_POOLS)}")
    index = EVALUATED_POOLS.index(pool)
    return tuple(POOL_SEED_BASE + 1000 * index + stream for stream in range(STREAMS))


def pool_attempts(pool: str) -> int:
    """Attempts per stream of one pool."""

    if pool not in EVALUATED_POOLS:
        raise KeyError(f"unknown pool {pool!r}; declared: {list(EVALUATED_POOLS)}")
    return BASE_ATTEMPTS_PER_STREAM if pool == POOL_BASE else POOL_ATTEMPTS_PER_STREAM


#: The resampling floor, which is the project's measured percentile-bootstrap
#: floor and nothing else, applied per (pool, stream) because the resample is
#: within stream.
MIN_UNITS_PER_POOL = MINIMUM_BOOTSTRAP_UNITS

#: Pools whose structures are folded twice, under the two declared realisations.
#: Only the base pool is: the best-of-k control selects on one realisation and
#: evaluates on the other, and a pool mean is unbiased under either.
DOUBLE_FOLDED_POOLS: tuple[str, ...] = (POOL_BASE,)

CROSS_FIT_NOTE = (
    "a reward containing the pLDDT term is measured with noise, and the expected "
    "maximum over k draws of a noisy score is inflated by that noise while a mean "
    "is not. Read naively, best-of-k would therefore be flattered and k* biased "
    "downward -- which is the direction that favours this project's own thesis. "
    "The control is cross-fitted instead: the base pool is folded under two "
    "declared independent realisations, the selection is made on realisation A "
    "and the selected sample's reward is read from realisation B, so the winner's "
    "curse is removed by construction rather than bounded by argument. The "
    "environment reward is exactly computable and carries no such noise, and its "
    "control is identical under either realisation"
)

# ---------------------------------------------------------------- the estimands

#: The declared k grid of the best-of-k control.
K_GRID: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

#: How many base-pool members the deepest declared ``k`` must have behind it.
#: R@k is read off the resampled pool's own order statistics, so at ``k`` equal
#: to the pool size almost all the weight would sit on one observation; a factor
#: of four keeps the deepest rung estimable while the band still widens there,
#: which is the honest consequence of asking about a deep tail.
DEEPEST_K_POOL_MULTIPLE = 4

#: The threshold at which a gain counts as reachable by rejection sampling. A
#: gain that 64 base samples plus selection match, with no weight update and
#: three orders of magnitude fewer reward evaluations than the training spent,
#: is not a bias the optimisation installed.
K_REDISTRIBUTION = 64

K_STAR_RULE = (
    f"k* is the smallest k of the declared grid at which the simultaneous "
    f"interval on R@k(base) minus the pool's mean reward lies at or above zero. "
    f"k* at or below {K_REDISTRIBUTION} is the redistribution reading; k* above "
    f"it is the reading that selection over the base pool does not reach the "
    f"gain at that budget. k* unreached anywhere on the grid is reported as "
    f"greater than {K_GRID[-1]} with the grid's own ceiling named, never as "
    "infinite"
)

#: The reachability ladder: charge windows further into the base's tail than the
#: declared one, at the same width and the same half-width, declared here before
#: any arm's evaluation pool existed.
#:
#: It exists because of an asymmetry the declared design cannot otherwise resolve.
#: If the environment arm's gain turns out reachable by selection from the base --
#: a small ``k*`` -- two readings fit: the method installs nothing at this scale
#: even against a closed-form reward, or the declared target was simply
#: base-reachable all along and the experiment had no real positive control. Those
#: predict the same ``k*``. What separates them is how ``k*`` behaves as the target
#: gets harder, and the base pool's half of that costs nothing: the charge of every
#: sampled sequence is already measured, so reading it against a harder window says
#: how much selection pressure that target would need. The ladder is therefore a
#: property of the *base* pool and is read as a diagnostic, never as a reward: no
#: policy is paid at any window but the declared one.
CHARGE_LADDER: tuple[tuple[float, float], ...] = (
    (8.0, 12.0), (14.0, 18.0), (20.0, 24.0), (26.0, 30.0), (32.0, 36.0),
)

#: The selection budget a rung has to defeat to serve as a positive control: the
#: deepest declared ``k``, so "unreachable" means unreachable by the best-of-k
#: control this experiment actually runs.
RUNG_BUDGET = K_GRID[-1]

#: A rung is unreachable at that budget when the upper confidence bound on the
#: probability that a ``RUNG_BUDGET``-sample rejection sampler hits it at all
#: falls below this. At a half, even the generous reading says such a sampler is
#: more likely to miss the target than to hit it.
RUNG_UNREACHABLE_BOUND = 0.5

#: One-sided confidence level of that bound.
RUNG_CONFIDENCE = 0.95

#: The first rung rule's own selection, kept on the record. It read the
#: conjunction indicator, which the base pool attains nowhere, so the exact bound
#: cleared its threshold at the first rung and it selected the declared target.
RUNG_RULE_FIRST_SELECTION: tuple[float, float] = (8.0, 12.0)

#: The second rule's quantity, level and budget: the first rung whose expected
#: best-of-k shaped reward over the base pool, at this budget, falls below this
#: level.
RUNG_SHAPED_BUDGET = K_GRID[-1]
RUNG_SHAPED_BOUND = 0.70

RUNG_RULE_VOIDED = (
    "the first rule is voided and the ground is narrow, because the ground is the "
    "only thing that makes voiding legitimate. A rule is *not* voidable because "
    "its answer is unfavourable to a hypothesis. It is voidable when its answer is "
    "**degenerate by construction**: the rung it selected is the declared target "
    "itself, which makes this arm a duplicate of arm E, and an arm that duplicates "
    "another arm answers no question at all. That is a defect in the instrument "
    "rather than a result about the world, and it is visible without reference to "
    "any trained pool -- so voiding on it costs the pre-registration nothing, "
    "where voiding because a number came out the wrong way would cost it "
    "everything.\n"
    "The premise that failed is worth stating as a finding. The rule was written "
    "for a base pool that reaches the declared target and needs a harder one. This "
    "pool attains the conjunction *never* in 3,072 draws and simultaneously "
    "reaches 0.9963 on the shaped reward, missing on two razor edges: charge +7.98 "
    "against a window opening at +8.0, low-complexity fraction 0.252 against a "
    "limit closing at 0.25. A shaped reward prices a constraint; a conjunction "
    "enforces it; and the two can disagree completely. The ladder is therefore "
    "uninformative on the conjunction, which is zero at every rung, and "
    "discriminating on the shaped reward, whose pool means run 0.3221 down to "
    "0.1666 and whose R@512 runs 0.9705 down to 0.6231.\n"
    "A rule has to be posed on a quantity that discriminates. That, and not a "
    "preference for a rung, is why the second rule is posed where it is."
)

RUNG_SHAPED_RULE = (
    f"the chosen rung is the first of the declared ladder whose expected "
    f"best-of-{RUNG_SHAPED_BUDGET} shaped environment reward over the base pool "
    f"falls below {RUNG_SHAPED_BOUND}. Declared from base-only data, before any "
    "trained pool had been read, and applied once.\n"
    "It selects [32, 36], whose R@512 is 0.6231 against [26, 30]'s 0.7394. That "
    "rung is a hard target and not an impossible one: the earlier degeneracy sweep "
    "established it is attainable at reward exactly 1.0 by a lysine-rich, "
    "non-repeating, full-diversity sequence at a K share of 0.28, and an "
    "independently constructed witness reaches 1.0 at K shares of 0.350, 0.318 and "
    "0.325 at lengths 100, 110 and 120 -- the share depending on length, since the "
    "charge window is absolute. Without that sweep already on the record the "
    "distinction between a hard rung and an unattainable one could not have been "
    "drawn at all."
)

RUNG_RULE = (
    f"the chosen rung is the first of the declared ladder whose one-sided "
    f"{RUNG_CONFIDENCE:.2f} upper bound on the probability that a "
    f"{RUNG_BUDGET}-sample rejection sampler hits its conjunction at all lies below "
    f"{RUNG_UNREACHABLE_BOUND}. The bound is exact-binomial (Clopper-Pearson) on the "
    "rung's observed conjunction count and not a bootstrap: a rung the base pool "
    "never hits has an all-zero indicator, whose nonparametric bootstrap is a "
    "zero-width interval at zero and therefore asserts unreachability with no "
    "uncertainty at all. That is the one place in this analysis where the "
    "resampling interval would be confidently wrong, and it is exactly the place "
    "the rung is chosen.\n"
    "If no rung of the declared ladder clears the bound -- if rejection sampling "
    "from the base reaches every charge target on it -- that is the finding and the "
    "ladder is not extended until something qualifies. It would mean the coordinate "
    "cannot support a positive control, which is a statement about the base "
    "distribution and not about reinforcement learning, and it is reported as such."
)

LADDER_NOTE = (
    "the base pool's expected best-of-k reward at each rung of the declared charge "
    "ladder, with the first rung the declared target itself. It answers one half of "
    "the ambiguity a small k* leaves: whether the declared target was base-reachable. "
    "It cannot answer the other half -- whether reinforcement learning installs a bias "
    "against a target selection cannot reach -- because the trained policies were "
    "optimised against the declared window and are off-objective at every higher rung. "
    "That half needs a policy trained against a higher rung, which is a design change "
    "and not an analysis"
)

# ------------------------------------------------- the length confound
#
# Mean CA-pLDDT depends on sequence length, and the dependence is **U-shaped** with
# a minimum somewhere around 72 to 100 residues. It is not monotone, which is how
# an earlier draft of this block had it. Priced from 93,049 of this project's own
# recorded ESMFold2 readings: 69.2 at 16-23 residues, 59.9 at 24-31, 56.5 at 32-39,
# 52.5 at 56-71, 51.3 at 72-95 -- and then back up, 52.6 at 96-143 and 56.3 at
# 144-223. A nested-prefix series over the same sequences shows the turn more
# sharply still, 35.1 at 120 residues, 51.1 at 160 and 60.6 at 192, and that
# within-sequence slope is the one that bears on a policy changing the length of
# its own output.
#
# So the risk is **movement away from the minimum in either direction**, not
# shortening. On the short side the best recorded readings at 16-30 residues are
# amphipathic helices at 80.8 to 84.3, a memory-derived reward of 0.81 to 0.84
# against the base pool's 0.468. On the long side the within-sequence slope between
# 160 and 192 residues is worth about nine pLDDT points on its own. Either
# direction can buy reward with nothing installed, and none of the four declared
# degeneracy bounds reads length at any point. The structural-support rule bounds
# the short channel at 16 residues and does nothing about the long one.
#
# The remedy is the device this project used to reject the depth confound against
# its clause (ii): read the contrast within strata of the confounding variable and
# withhold the verdict where the gain does not survive. A non-monotone dependence
# is the case where it is indispensable rather than merely convenient -- no linear
# control and no single covariate can remove a U -- and stratification is agnostic
# to the sign of the dependence, so it catches movement in either direction. It
# costs no compute, because the pool sampler already records every sequence's
# length.

#: Length stratum edges, lower inclusive and upper exclusive, declared explicitly
#: and fixed before any trained pool was read.
#:
#: They are fixed from the *base* policy's own length distribution -- measured at
#: this experiment's operating point and completion budget in the calibration
#: record cited in the registration -- and never from a trained pool's, so the
#: choice cannot be tuned to a result. The first edge is the structural support's
#: own lower bound, the last is one past the declared completion budget, and the
#: interior edges are closer together at the short end, where the fall from 16 to
#: 32 residues is about nine pLDDT points. They bracket the minimum of the U at
#: 64-95 and 96-127, which is the right shape, and it was arrived at for the wrong
#: reason: the rationale written here first described the dependence as monotone
#: and placed the fine edges where the fall was steepest, not where the turn is.
#:
#: The edge at 160 splits what was one 73-residue stratum on the rising limb.
#: Left whole it could hide up to about twenty pLDDT points -- the within-sequence
#: slope between 160 and 192 residues is about nine points on its own -- which is
#: larger than any contrast the stratum is meant to carry, so it would have
#: defeated the gate in exactly the place the gate matters. Nothing had been read
#: when it was split, so the correction cost the registration nothing; holding a
#: declared edge for consistency when it is wrong and nothing has been measured
#: would be false rigour rather than discipline.
#:
#: It is the *only* split, and the reason is the opposite failure. A stratum too
#: thin for either pool to populate clears the resampling floor on neither side
#: and yields no verdict at all, which is worse than a bounded residual: an
#: unresolved column withholds a reading, a wide one still gives a bounded one.
#: So the residual inside 128-159 and 160-200 is declared and kept rather than
#: subdivided away.
LENGTH_STRATUM_EDGES: tuple[int, ...] = (16, 24, 32, 48, 64, 96, 128, 160, 201)

LENGTH_STRATUM_NAMES: tuple[str, ...] = tuple(
    f"{low}-{high - 1}" for low, high in zip(LENGTH_STRATUM_EDGES[:-1],
                                             LENGTH_STRATUM_EDGES[1:])
)


def length_stratum(length: int) -> str | None:
    """The declared stratum one completion falls in, or ``None`` outside them all.

    ``None`` is every length the structural term is not defined on anyway -- the
    empty completion and anything below the structural support -- so a stratified
    structural contrast and the structural reward have the same domain by
    construction.
    """

    value = int(length)
    for index, name in enumerate(LENGTH_STRATUM_NAMES):
        if LENGTH_STRATUM_EDGES[index] <= value < LENGTH_STRATUM_EDGES[index + 1]:
            return name
    return None


CRITERION_RATE_AMPLIFICATION = (
    "the predicted-confidence rate is stratified because a threshold crossing "
    "amplifies a shift in a distribution's location when the threshold sits inside "
    "the shift, and here it does. The recorded mean CA-pLDDT at 16-23 residues is "
    "69.2, essentially on the project's 70-point criterion, so in that stratum "
    "about half of everything clears it from length alone; at 72-95 residues the "
    "mean is 51.3 and almost nothing does. The rate is therefore *more* "
    "length-driven than the mean it derives from, not less. It is also the "
    "quotable form -- 'this share of generations exceeds seventy' is a "
    "design-capability sentence in a way a mean is not -- and a caveat in a "
    "declaration does not travel into a later sentence the way a stratified number "
    "does. It stays a derived diagnostic and is not promoted to a verdict input"
)

STRATIFICATION_RULE = (
    "every pool contrast that involves the structural term is read within the "
    "declared length strata as well as over the pool, and an installed-bias "
    "verdict is withheld unless the arm's reward gain is bounded above zero in at "
    "least one stratum where both pools clear the resampling floor. The stratum "
    "that carries it is quoted with the verdict. "
    "Where no stratum admits the comparison -- which happens when a policy has "
    "moved its length distribution out of the base pool's altogether -- the "
    "verdict is withheld for want of a matched comparison rather than passed on "
    "the pooled gain. That is the conservative direction and it is the point: a "
    "gain measured across a length distribution the base pool does not share is "
    "not evidence that anything was installed. "
    "The best-of-k families are not stratified. k* asks whether rejection sampling "
    "from the base pool reaches the policy's mean reward, which is a question about "
    "the pools as drawn; whether that gain is a length artifact is what the "
    "stratified gain decides, and it decides it first. Where the stratified gain "
    "withholds the verdict, no k* reading is taken."
)

#: The quantile of the base pool's own base-model NLL above which a policy sample
#: counts as outside the base's observed support. Its null value is one minus the
#: quantile, which is what the containment reading is tested against.
SUPPORT_TAIL_QUANTILE = 0.99

SUPPORT_NOTE = (
    "the declared rewards are bounded above by one, and the base pool attains "
    "that bound, so 'the fraction of policy samples whose reward lies outside "
    "the base's observed reward range' is identically zero for a saturated "
    "reward and cannot carry the support reading on its own. It is reported for "
    "every pool, with its ceiling stated. The reading rests instead on the base "
    f"model's own likelihood: the share of a pool's samples whose per-residue "
    f"NLL under the base model exceeds the base pool's "
    f"{SUPPORT_TAIL_QUANTILE:.2f} quantile, whose null value is "
    f"{1 - SUPPORT_TAIL_QUANTILE:.2f} and which has no ceiling"
)

# ------------------------------------------------------------------- families
#
# One family per quantity and per unit. A maximum-statistic band in native units
# may not span two quantities measured in different units, which is the rule the
# objective-alignment declaration already applies to its identity endpoints, and
# the rewards are different quantities even though all of them are dimensionless.

FAMILY_REWARD_GAIN = "reward_gain"
FAMILY_BEST_OF_K = "best_of_k"
FAMILY_CROSS_NLL = "cross_nll"
FAMILY_ENTROPY = "entropy"
FAMILY_SUPPORT_TAIL = "support_tail_share"
FAMILY_IDENTITY_MEAN = "identity_mean_over_query"
FAMILY_IDENTITY_FRACTION = "identity_fraction_below_30"
FAMILY_HELD_OUT_NLL = "held_out_natural_nll"
FAMILY_PLDDT = "mean_ca_plddt"
FAMILY_DISTINCT_5MER = "distinct_5mer_fraction"
FAMILY_LONGEST_RUN = "longest_single_residue_run"
FAMILY_COMPOSITION_ENTROPY = "composition_entropy_bits"
FAMILY_LOW_COMPLEXITY = "low_complexity_fraction"
FAMILY_LENGTH = "length"
FAMILY_REWARD_GAIN_BY_LENGTH = "reward_gain_by_length"
FAMILY_PLDDT_BY_LENGTH = "mean_ca_plddt_by_length"
FAMILY_PLDDT_CRITERION_RATE_BY_LENGTH = "plddt_criterion_rate_by_length"

#: Families read within length strata that are diagnostics rather than verdict
#: inputs. The gate reads the reward gain alone; these say what the length shift
#: did to the quantities a reader would otherwise quote unstratified.
STRATIFIED_DIAGNOSTIC_FAMILIES: tuple[str, ...] = (
    FAMILY_PLDDT_BY_LENGTH, FAMILY_PLDDT_CRITERION_RATE_BY_LENGTH,
)
FAMILY_CHARGE_RATE = "charge_window_rate"
FAMILY_CONJUNCTION_RATE = "environment_conjunction_rate"
FAMILY_HARD_CONJUNCTION_RATE = "environment_hard_conjunction_rate"
FAMILY_MEMORY_CONJUNCTION_RATE = "memory_conjunction_rate"
FAMILY_PLDDT_CRITERION_RATE = "plddt_criterion_rate"

#: The family carrying each reward's *own* objective-attainment rate: the share of
#: a pool's completions that satisfy that reward's declared conjunction outright,
#: rather than approaching it.
#:
#: This is a declared reading for every arm and every pool, beside the shaped
#: reward, and it is here because of what the base pool showed. A pool can read
#: 0.9963 on the shaped environment reward while satisfying that objective zero
#: times in 3,072 draws, so **a reward gain is not evidence that the objective was
#: attained** -- and the whole inferential chain of this experiment runs through
#: reward gain. An arm whose shaped reward rises while its attainment rate stays
#: at zero has learned to approach constraints without meeting them, which is not
#: a failure of the measurement but very nearly this project's own claim about
#: redistribution inside a narrower domain. It has to be a quantity rather than
#: something a later reader reconstructs from two tables side by side.
OBJECTIVE_ATTAINMENT_FAMILY: Mapping[str, str] = {
    sr.REWARD_MEMORY: FAMILY_MEMORY_CONJUNCTION_RATE,
    sr.REWARD_ENVIRONMENT: FAMILY_CONJUNCTION_RATE,
    sr.REWARD_BOTH: FAMILY_MEMORY_CONJUNCTION_RATE,
    sr.REWARD_ENVIRONMENT_HARD: FAMILY_HARD_CONJUNCTION_RATE,
}

ATTAINMENT_NOTE = (
    "reported for every arm and every pool beside the shaped reward. The shaped "
    "reward prices a constraint and the conjunction enforces it, and the base pool "
    "shows how far apart the two can be: 0.9963 shaped against zero attainments in "
    "3,072 draws, missing on charge +7.98 against a window opening at +8.0 and a "
    "low-complexity fraction of 0.252 against a limit closing at 0.25. A rising "
    "shaped reward with a flat attainment rate is a policy approaching constraints "
    "without meeting them, and it is reported as that rather than as a gain"
)



def _reward_family(stem: str, reward: str) -> str:
    return f"{stem}__{reward}"


#: Declared size per family. Every family is the four contrast pools crossed
#: with whatever axis the family has; the reward-indexed families exist once per
#: reward because their units are that reward's.
def structural_rewards() -> tuple[str, ...]:
    """The declared rewards whose value depends on a folded structure.

    Only these are read within length strata, because only these carry the
    instrument whose confidence depends on length. The exactly computable rewards
    are untouched by the confound and by its remedy.
    """

    return tuple(name for name in sr.REWARD_NAMES if sr.REWARD_NEEDS_STRUCTURE[name])


FAMILY_SIZES: dict[str, int] = {
    **{_reward_family(FAMILY_REWARD_GAIN, reward): len(CONTRAST_POOLS)
       for reward in sr.REWARD_NAMES},
    **{_reward_family(FAMILY_REWARD_GAIN_BY_LENGTH, reward):
       len(CONTRAST_POOLS) * len(LENGTH_STRATUM_NAMES)
       for reward in structural_rewards()},
    FAMILY_PLDDT_BY_LENGTH: len(CONTRAST_POOLS) * len(LENGTH_STRATUM_NAMES),
    FAMILY_PLDDT_CRITERION_RATE_BY_LENGTH: (len(CONTRAST_POOLS)
                                            * len(LENGTH_STRATUM_NAMES)),
    FAMILY_LENGTH: len(CONTRAST_POOLS),
    **{_reward_family(FAMILY_BEST_OF_K, reward): len(CONTRAST_POOLS) * len(K_GRID)
       for reward in sr.REWARD_NAMES},
    FAMILY_CROSS_NLL: len(CONTRAST_POOLS) * 2,
    FAMILY_ENTROPY: len(CONTRAST_POOLS),
    FAMILY_SUPPORT_TAIL: len(CONTRAST_POOLS),
    FAMILY_IDENTITY_MEAN: len(CONTRAST_POOLS),
    FAMILY_IDENTITY_FRACTION: len(CONTRAST_POOLS),
    FAMILY_HELD_OUT_NLL: len(CONTRAST_POOLS),
    FAMILY_PLDDT: len(CONTRAST_POOLS),
    FAMILY_DISTINCT_5MER: len(CONTRAST_POOLS),
    FAMILY_LONGEST_RUN: len(CONTRAST_POOLS),
    FAMILY_COMPOSITION_ENTROPY: len(CONTRAST_POOLS),
    FAMILY_LOW_COMPLEXITY: len(CONTRAST_POOLS),
    FAMILY_CHARGE_RATE: len(CONTRAST_POOLS),
    FAMILY_CONJUNCTION_RATE: len(CONTRAST_POOLS),
    FAMILY_HARD_CONJUNCTION_RATE: len(CONTRAST_POOLS),
    FAMILY_MEMORY_CONJUNCTION_RATE: len(CONTRAST_POOLS),
    FAMILY_PLDDT_CRITERION_RATE: len(CONTRAST_POOLS),
}

#: One declared seed block per family, so no two families read the same draws.
#: The block base is this campaign's own date rather than the bidirectional
#: module's seed, so that no draw of this experiment can coincide with a draw of
#: the objective-alignment campaign's families.
BAND_SEED = 20261114
BAND_DRAWS = 10_000
BAND_CONFIDENCE = 0.95

FAMILY_SEED_OFFSETS: dict[str, int] = {
    name: 100_000 * index for index, name in enumerate(sorted(FAMILY_SIZES))
}

#: Columns a pool seed is declared for; the largest declared family is the
#: best-of-k family at forty columns.
MAX_FAMILY_COLUMNS = 100


def pool_draw_seed(family: str, column: int, side: int, stream_index: int) -> int:
    """The declared resampling seed of one pool of one column of one family."""

    if family not in FAMILY_SEED_OFFSETS:
        raise KeyError(f"no declared seed block for family {family!r}")
    if not 0 <= column < MAX_FAMILY_COLUMNS:
        raise ValueError(f"a pool seed is declared for at most {MAX_FAMILY_COLUMNS} columns")
    if side not in (0, 1) or not 0 <= stream_index < 100:
        raise ValueError("a pool seed is declared for two sides and at most 100 streams")
    return (BAND_SEED + FAMILY_SEED_OFFSETS[family] + 1000 * column
            + 100 * side + stream_index)


def family_name(stem: str, reward: str | None = None) -> str:
    """The declared name of one family, refusing an undeclared one."""

    name = stem if reward is None else _reward_family(stem, reward)
    if name not in FAMILY_SIZES:
        raise KeyError(f"undeclared family {name!r}")
    return name


# --------------------------------------------------------------- the readings

#: Finding states, from the project's single source.
STATES: tuple[str, ...] = FINDING_STATES

#: The reading a pool earns. Five, because the two that matter are opposite
#: answers to the experiment's question and the other three are the ways it can
#: fail to answer.
READING_BIAS_ON_MANIFOLD = "bias_installed_within_protein_region"
READING_BIAS_OFF_MANIFOLD = "bias_installed_outside_protein_region"
READING_REDISTRIBUTION = "redistribution_of_pretraining_memory"
READING_NO_OPTIMISATION = "no_optimisation"
READING_UNRESOLVED = UNRESOLVED
READINGS: tuple[str, ...] = (
    READING_BIAS_ON_MANIFOLD,
    READING_BIAS_OFF_MANIFOLD,
    READING_REDISTRIBUTION,
    READING_NO_OPTIMISATION,
    READING_UNRESOLVED,
)

#: Degeneracy bounds, declared before any arm was trained. Breaching one does
#: not make a bias absent; it decides which of the two installed-bias readings
#: applies, because a bias reached by leaving the protein-like region is the
#: project's own stronger claim rather than a refutation of it.
DEGENERACY_BOUNDS: Mapping[str, str] = {
    FAMILY_PLDDT: (
        "the pool's mean CA-pLDDT interval does not lie more than 10 points below "
        "the base pool's point estimate"
    ),
    FAMILY_DISTINCT_5MER: (
        "the pool's mean distinct 5-mer fraction interval does not lie below the "
        "base pool's own 5th percentile"
    ),
    FAMILY_LONGEST_RUN: (
        "the pool's mean longest single-residue run interval does not lie above "
        "twice the base pool's 95th percentile"
    ),
    FAMILY_COMPOSITION_ENTROPY: (
        "the pool's mean composition entropy interval does not lie more than one "
        "bit below the base pool's point estimate"
    ),
}

PLDDT_DEGENERACY_MARGIN = 10.0
COMPOSITION_ENTROPY_DEGENERACY_MARGIN = 1.0
LONGEST_RUN_DEGENERACY_MULTIPLE = 2.0
DISTINCT_5MER_BASE_QUANTILE = 0.05
LONGEST_RUN_BASE_QUANTILE = 0.95

DECISION_RULES = (
    "Optimisation happened for a pool when its reward-gain interval under its "
    "own reward lies wholly above zero. A pool that fails this is read "
    f"{READING_NO_OPTIMISATION!r} and nothing further about it is read as bias "
    "or as redistribution; a failed optimisation is a failed optimisation and is "
    "reported as one.\n"
    f"Redistribution is the reading when optimisation happened and k* is at most "
    f"{K_REDISTRIBUTION}: an explicit selection over the base model's own samples, "
    "under the identical reward and with no weight update, matched the trained "
    "policy.\n"
    f"An installed bias requires all three of: the reward-gain interval wholly "
    f"above zero; k* above {K_REDISTRIBUTION}; and the share of the pool's samples "
    "beyond the base pool's own 0.99 quantile of base-model NLL bounded above "
    "0.01, which is that share's null value. The first says the optimisation "
    "moved the reward, the second that selection could not have, and the third "
    "that the policy placed mass where the base model's memory did not.\n"
    "An installed bias is then read within or outside the protein-like region by "
    "the declared degeneracy bounds. Both are installed biases; which one it is "
    "decides what the result means and neither is hidden.\n"
    "Anything else is unresolved, which is not evidence that the capability is "
    "absent."
)

FALSIFICATION = (
    "The project's thesis is falsified, on this policy and these rewards, if the "
    "environment arm earns an installed-bias reading within the protein-like "
    "region: post-training against a reward that no learned model supplied would "
    "then have installed a bias the pre-training memory did not contain, without "
    "leaving the region the memory describes. It is falsified more directly if "
    "the memory-derived arm does, because that reward is computed by a model "
    "trained on the policy's own corpus and the thesis says such a reward can "
    "only redistribute.\n"
    "The thesis's predicted pattern is the memory-derived arm reading "
    "redistribution while the environment arm reads an installed bias outside the "
    "protein-like region: a bias can be installed, and installing one against a "
    "target the memory does not encode costs the protein-like region, because the "
    "memory is the region.\n"
    "Three outcomes would leave the question open rather than answer it: no arm "
    "optimising at all, which indicts the optimiser; every arm reading "
    "unresolved, which indicts the sample sizes; and the environment arm "
    "optimising while the memory-derived arm does not, which would say the "
    "instrument differs between arms rather than the provenance."
)

BOUNDS = (
    "One policy, one scale, one algorithm and two rewards. A null here does not "
    "show that no reinforcement learning can install a design bias in a protein "
    "language model, and a positive does not show that a design bias is useful: "
    "net charge inside a window is a declared target, not a function.\n"
    "Every structural quantity is predicted. No sequence in this experiment has "
    "a measured folding, stability or activity.\n"
    "The memory-derived reward shares its instrument with the memory-derived "
    "endpoint, so a gain on it is partly a statement about agreement with the "
    "reward's own scorer. That is why the best-of-k control is not optional and "
    "why the environment arm exists at all: its reward's instrument is "
    "arithmetic.\n"
    "The fold is not bit-reproducible at a fixed seed, as this project has "
    "measured twice, so a reward paid during training is conditional on its run. "
    "The between-realisation spread at the declared sample count is measured and "
    "recorded; it is not removed.\n"
    "Identity to a UniRef50 snapshot is similarity to a snapshot and not "
    "pretraining exposure, and the base's declared training mixture is not that "
    "snapshot. The project's existing non-identifiability finding applies "
    "unchanged.\n"
    "The resampling unit is the generated sequence within stream. This "
    "quantifies the decoder's sampling noise at fixed weights. It is not "
    "weight-level uncertainty: each arm is one training run, and nothing here "
    "covers the variation between runs."
)

# ------------------------------------------------------- declaration checks


def _check_declarations() -> None:
    """One name means one thing, and every declared size is reachable."""

    if set(TRAINED_ARMS) != set(sr.REWARD_NAMES):
        raise AssertionError(
            f"the arms are the rewards; got {TRAINED_ARMS} against {sr.REWARD_NAMES}"
        )
    for item in ARMS:
        if item.needs_structure != sr.REWARD_NEEDS_STRUCTURE[item.reward]:
            raise AssertionError(
                f"{item.name}: declares needs_structure={item.needs_structure} against "
                f"the reward's own {sr.REWARD_NEEDS_STRUCTURE[item.reward]}"
            )
    if len(set(EVALUATED_POOLS)) != len(EVALUATED_POOLS):
        raise AssertionError("a pool name appears twice")
    if POOL_BASE in CONTRAST_POOLS:
        raise AssertionError("the base pool is the control and carries no contrast column")
    if sorted(K_GRID) != list(K_GRID) or K_GRID[0] != 1:
        raise AssertionError("the k grid is increasing and starts at one")
    edges = list(LENGTH_STRATUM_EDGES)
    if edges != sorted(edges) or len(set(edges)) != len(edges) or len(edges) < 3:
        raise AssertionError(f"the length strata are strictly increasing; got {edges}")
    if edges[0] != sr.STRUCTURE_SUPPORT[0]:
        raise AssertionError(
            f"the first length stratum begins at the structural support's own lower "
            f"bound; got {edges[0]} against {sr.STRUCTURE_SUPPORT[0]}"
        )
    if edges[-1] != MAX_NEW_TOKENS + 1:
        raise AssertionError(
            f"the last length stratum ends at the declared completion budget; got "
            f"{edges[-1] - 1} against {MAX_NEW_TOKENS}"
        )
    if length_stratum(sr.STRUCTURE_SUPPORT[0] - 1) is not None:
        raise AssertionError("no stratum may reach below the structural support")
    if CHARGE_LADDER[0] != sr.CHARGE_WINDOW:
        raise AssertionError(
            f"the ladder's first rung is the declared charge window; got "
            f"{CHARGE_LADDER[0]} against {sr.CHARGE_WINDOW}"
        )
    widths = {high - low for low, high in CHARGE_LADDER}
    if len(widths) != 1:
        raise AssertionError(
            f"every rung of the ladder has the declared window's width, or a rung is "
            f"harder for two reasons at once; got widths {sorted(widths)}"
        )
    lows = [low for low, _ in CHARGE_LADDER]
    if lows != sorted(lows) or len(set(lows)) != len(lows):
        raise AssertionError("the ladder is strictly increasing in difficulty")
    if K_REDISTRIBUTION not in K_GRID:
        raise AssertionError(
            f"the redistribution threshold {K_REDISTRIBUTION} is a point of the declared "
            f"grid {K_GRID}, or k* could never equal it"
        )
    if K_GRID[-1] * DEEPEST_K_POOL_MULTIPLE > BASE_ATTEMPTS_PER_STREAM * STREAMS:
        raise AssertionError(
            f"the deepest declared k of {K_GRID[-1]} needs a base pool of at least "
            f"{K_GRID[-1] * DEEPEST_K_POOL_MULTIPLE}; the declaration gives "
            f"{BASE_ATTEMPTS_PER_STREAM * STREAMS}"
        )
    for family, size in FAMILY_SIZES.items():
        if size < 1 or size > MAX_FAMILY_COLUMNS:
            raise AssertionError(f"family {family} declares {size} columns")
        if family not in FAMILY_SEED_OFFSETS:
            raise AssertionError(f"family {family} has no declared seed block")
    if len(set(FAMILY_SEED_OFFSETS.values())) != len(FAMILY_SEED_OFFSETS):
        raise AssertionError("two families share a seed block")
    if set(DEGENERACY_BOUNDS) - set(FAMILY_SIZES):
        raise AssertionError("a degeneracy bound names a family that is not declared")
    if set(READINGS) & set(STATES) != {UNRESOLVED}:
        raise AssertionError(
            "the readings borrow exactly one finding-state name, which is unresolved; "
            f"got {sorted(set(READINGS) & set(STATES))}"
        )
    if SUPPORTED in READINGS or BOUNDED_NEGATIVE in READINGS or NON_IDENTIFIABLE in READINGS:
        raise AssertionError("a reading is not a finding state and must not reuse its names")


_check_declarations()


def contract() -> dict[str, Any]:
    """The whole declaration as one record, for a run note or a result file."""

    return {
        "schema": SCHEMA,
        "campaign": CAMPAIGN,
        "declared_in": "docs/rl-environment-provenance.md",
        "policy": {
            "arm": POLICY_ARM,
            "checkpoint": str(policy_checkpoint()),
            "parameters": POLICY_PARAMETERS,
            "published_dpo_sibling": str(published_dpo_checkpoint()),
        },
        "arms": [
            {"name": item.name, "reward": item.reward,
             "needs_structure": item.needs_structure,
             "provenance": item.provenance, "question": item.question}
            for item in ARMS
        ],
        "reward_constants": {
            "ph": sr.REFERENCE_PH,
            "pka_set": sr.PKA_PROVENANCE,
            "charge_window": list(sr.CHARGE_WINDOW),
            "charge_half_width": sr.CHARGE_HALF_WIDTH,
            "length_window": list(sr.LENGTH_WINDOW),
            "length_half_width": sr.LENGTH_HALF_WIDTH,
            "low_complexity_window": sr.LOW_COMPLEXITY_WINDOW,
            "low_complexity_entropy_bits": sr.LOW_COMPLEXITY_ENTROPY_BITS,
            "low_complexity_limit": sr.LOW_COMPLEXITY_LIMIT,
            "low_complexity_provenance": sr.LOW_COMPLEXITY_PROVENANCE,
            "plddt_scale": sr.PLDDT_SCALE,
            "empty_completion_reward": sr.EMPTY_COMPLETION_REWARD,
        },
        "algorithm": settings().as_record(),
        "hyperparameter_note": HYPERPARAMETER_NOTE,
        "reward_calls_per_arm": reward_calls(),
        "pools": {
            pool: {"attempts_per_stream": pool_attempts(pool), "seeds": list(pool_seeds(pool))}
            for pool in EVALUATED_POOLS
        },
        "streams": STREAMS,
        "minimum_units_per_pool": MIN_UNITS_PER_POOL,
        "double_folded_pools": list(DOUBLE_FOLDED_POOLS),
        "cross_fit_note": CROSS_FIT_NOTE,
        "length_strata": {"edges": list(LENGTH_STRATUM_EDGES),
                          "names": list(LENGTH_STRATUM_NAMES),
                          "rule": STRATIFICATION_RULE,
                          "structural_rewards": list(structural_rewards())},
        "objective_attainment": {"families": dict(OBJECTIVE_ATTAINMENT_FAMILY),
                                 "note": ATTAINMENT_NOTE},
        "rung_rules": {"first": {"rule": RUNG_RULE,
                                 "selected": list(RUNG_RULE_FIRST_SELECTION),
                                 "voided": RUNG_RULE_VOIDED},
                       "second": {"rule": RUNG_SHAPED_RULE,
                                  "bound": RUNG_SHAPED_BOUND,
                                  "budget": RUNG_SHAPED_BUDGET,
                                  "selected": list(sr.HARD_CHARGE_WINDOW)
                                  if sr.hard_window_declared() else None}},
        "k_grid": list(K_GRID),
        "k_redistribution": K_REDISTRIBUTION,
        "k_star_rule": K_STAR_RULE,
        "support_tail_quantile": SUPPORT_TAIL_QUANTILE,
        "support_note": SUPPORT_NOTE,
        "families": dict(sorted(FAMILY_SIZES.items())),
        "band": {"statistic": "native maximum absolute centered deviation",
                 "draws": BAND_DRAWS, "seed": BAND_SEED, "confidence": BAND_CONFIDENCE},
        "readings": list(READINGS),
        "degeneracy_bounds": dict(DEGENERACY_BOUNDS),
        "decision_rules": DECISION_RULES,
        "falsification": FALSIFICATION,
        "bounds": BOUNDS,
    }
