# R6 更新与适应 — Post-training adaptation

What changes between a base checkpoint and its post-trained sibling? This module holds documented training-stage and checkpoint contrasts, not randomised controlled interventions: the ProLLaMA two-stage lineage, the three ProtGPT3 preference-optimised pairs, the ProGen2 fine-tune onto seven Pfam families, and the stopped reward-provenance campaign. Prediction, generation and predicted structure are read on the same pair, before and after.

## The question

Prediction contrasts score a tuned checkpoint and its base on identical supplied inputs; generation contrasts compare their own generated pools under the declared protocol. Each experiment retains its own support, inferential family and margins, including simultaneous families spanning pairs and strata where declared. Placing experiments in this module does not merge those designs. Checkpoint descent is verified as provenance; it does not by itself identify a causal effect of training.

## Code

- The ProLLaMA lineage and its one rendering: [`src/capability/models/joint_lineage.py`](../../src/capability/models/joint_lineage.py)
- The stage pair, its structures and its phenotype comparison: [`src/capability/generation/stage_pair.py`](../../src/capability/generation/stage_pair.py), [`stage_pair_structures.py`](../../src/capability/generation/stage_pair_structures.py), [`stage_pair_phenotype.py`](../../src/capability/generation/stage_pair_phenotype.py), run by [`scripts/capability/generation/analyse_stage_pair_phenotype.py`](../../scripts/capability/generation/analyse_stage_pair_phenotype.py) and [`summarise_stage_pair_structures.py`](../../scripts/capability/generation/summarise_stage_pair_structures.py)
- The objective-tuned pairs and their analysis: [`src/capability/generation/objective_tuned_arms.py`](../../src/capability/generation/objective_tuned_arms.py), [`objective_tuned_analysis.py`](../../src/capability/generation/objective_tuned_analysis.py), [`scripts/capability/generation/analyse_objective_tuned.py`](../../scripts/capability/generation/analyse_objective_tuned.py)
- The prediction side-effect on mutation ordering: [`src/capability/extensions/objective_tuned_prediction.py`](../../src/capability/extensions/objective_tuned_prediction.py), [`scripts/capability/extensions/score_objective_tuned_prediction.py`](../../scripts/capability/extensions/score_objective_tuned_prediction.py)
- Cross-scoring preparation: [`scripts/capability/evaluation/prepare_stage_cross_scoring.py`](../../scripts/capability/evaluation/prepare_stage_cross_scoring.py)

## Inputs

Locally staged checkpoints, named through the environment rather than the tree, and the campaign's own frozen cohorts: `results/R6/stage_pair_cross_scoring_frozenE18_20261011/cohort` and `results/R6/stage_pair_family_frozenE18_20261011`. The prediction contract is tracked in [`configs/objective_tuned_prediction.json`](../../configs/objective_tuned_prediction.json).

## Evidence

Local result trees: `results/R1/adaptation_lineage_20260813` and `results/R1/second_stage_capability_20260826` for the lineage; `results/R6/stage_pair_20261008`, `stage_pair_phenotype_20261008`, `stage_pair_cross_scoring_frozenE18_20261011`, `stagepair_cross_analysis_20261011` and `stage_pair_family_frozenE18_20261011` for the stage pair; `results/extensions/objective_tuned_*` for the tuned pairs, including `objective_tuned_prediction_20261110` and the later filter, base-likelihood, support-containment and selection-curve studies; `results/external_baseline_analysis/length_adjusted_contrasts_20261110`, `objective_tuned_cross_likelihood_20261114` and `objective_tuned_identity_20261114` for the folding contrasts; `results/long_cap_termination_20261110` for the 768-residue companion cohort; `results/extensions/rl_environment_provenance_20261114` for the stopped campaign.

## Shared boundaries

The lineage is scored under the bare `Seq=<...>` rendering, which the module's own header records as outside the instruction format Stage 2 was tuned for: the common bare interface is a valid protocol-specific comparison and not a measurement in Stage 2's native format, so a Stage 2 reading at or below Stage 1's is not evidence that instruction tuning removed a capability. Both stages are read under the token unit, so their magnitudes are not commensurable with residue-unit families. The fine-tune's local gain is retained, and family-statistical relearning is a plausible explanation rather than an identified mechanism. The tuned arms' mutation-effect prediction is a paired difference in the forward and reverse increments against Profile, Potts and GEMME together, whose construction, ladder and caveats belong to statistical memory; what this module adds is the difference between a base and its sibling. Length and the generation cap bind the structural contrasts, and identity to the natural record is not pretraining exposure. The reward-provenance campaign stopped without completing evaluation: its artifacts are retained for a later assessment and supply no conclusion here. Generation tooling is shared with generative expression, which owns what a model expresses; this module owns the before/after contrast.
