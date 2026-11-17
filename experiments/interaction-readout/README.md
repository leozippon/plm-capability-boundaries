# R3 相互作用信息 — Interaction readouts

Do single-mutation signals resolve double-mutant interactions, and what do the fitted couplings say about three-dimensional contact? This module reads the likelihood and the explicit coupling model on a non-additive target: a measured double-mutant cycle, adjusted for the singles that were actually measured.

## The question

The target is an interaction increment, not an additive effect, so each panel states in advance which nested baseline it is an increment over and on which groups that baseline is fitted. Two residual panels are retained with their own support — the original and the clean, indel-excluded one — and a number names which of them it sits on; on that lane the likelihood's forward increment is bounded rather than resolved. Contact evidence is auxiliary: it puts the coupling norm and a model's per-pair propagation response on one contact-identification axis and so qualifies the parameters the same lane fitted, without showing that those parameters hold less information than the statistics do.

## Code

- Pairwise extraction, the declared roster and the cycle target: [`src/capability/interactions/pairwise_epistasis.py`](../../src/capability/interactions/pairwise_epistasis.py), [`scripts/capability/interactions/extract_pairwise_epistasis.py`](../../scripts/capability/interactions/extract_pairwise_epistasis.py), [`fit_pairwise_epistasis.py`](../../scripts/capability/interactions/fit_pairwise_epistasis.py)
- Measured-singles correction and the residual panel: [`fit_measured_singles_correction.py`](../../scripts/capability/interactions/fit_measured_singles_correction.py), [`run_residual_panel.py`](../../scripts/capability/interactions/run_residual_panel.py), [`refit_likelihood_on_residual.py`](../../scripts/capability/interactions/refit_likelihood_on_residual.py)
- Contact qualification: [`src/capability/interactions/contact_enrichment.py`](../../src/capability/interactions/contact_enrichment.py), [`scripts/capability/interactions/measure_contact_epsilon_enrichment.py`](../../scripts/capability/interactions/measure_contact_epsilon_enrichment.py), and the response-side reading in [`scripts/capability/position/measure_contact_discrimination.py`](../../scripts/capability/position/measure_contact_discrimination.py)
- The interaction lane of the shared hierarchy: [`src/capability/extensions/explicit_statistics_lanes.py`](../../src/capability/extensions/explicit_statistics_lanes.py)

## Inputs

Local and not distributed: `data/pairwise_assets`, plus the retained contact annotations and the structure inputs the response side reads. The pairwise cohort descriptor is tracked in [`configs/explicit_statistics_cohorts.json`](../../configs/explicit_statistics_cohorts.json).

## Evidence

Local result trees: `results/R3` holds the residual panels (`pairwise_residual_full_panel_20260927`, and the indel-excluded `pairwise_residual_full_panel_indel_excluded_20260928`), the measured-singles panel, the corrected contact bootstrap and the corrected cycle labels; `results/pairwise_epistasis_20260924` retains the original extraction; `results/extensions/contact_discrimination_20261114` holds the contact-discrimination reading with its run note; `results/extensions/explicit_statistics_lanes_20261110/interaction` holds the lane's complete-case family.

## Shared boundaries

GEMME is non-identifiable in this lane by construction, because a double-mutant cycle contrast of a score that sums per-site scores is identically zero, so the hierarchy stops at profile and couplings. The lane sits near its own floor, and a small share there describes the floor rather than an equivalence. The cycle target is label-assisted in the sense that the measured singles enter the baseline, so it is not a zero-shot task. Side-by-side contact columns cannot separate "the model carries what the couplings lack" from "both carry it and one is noisier", and the bidirectional refit that would decide it left its records unvalidated and unread. `prollama` and `prollama-stage-1` are ordinary single-checkpoint panel arms here; the stage-to-stage comparison belongs to post-training adaptation.
