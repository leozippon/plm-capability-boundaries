# R3 相互作用信息 — Interaction readouts

Do single-mutation signals resolve double-mutant interactions, and what do the fitted couplings say about three-dimensional contact? This module reads the likelihood and the explicit coupling model on a non-additive target: a measured double-mutant cycle, adjusted for the singles that were actually measured.

## The question

The target is an interaction increment, not an additive effect, so each panel states in advance which nested baseline it is an increment over and on which groups that baseline is fitted. Two residual panels are retained with their own support — the original and the clean, indel-excluded one — and a number names which of them it sits on; on that lane the likelihood's forward increment is bounded rather than resolved. Contact evidence is auxiliary: it puts the coupling norm and a model's per-pair propagation response on one contact-identification axis and so qualifies the parameters the same lane fitted, without showing that those parameters hold less information than the statistics do.

## Code

- Pairwise extraction, the declared roster and the cycle target: [`src/capability/interactions/pairwise_epistasis.py`](../../src/capability/interactions/pairwise_epistasis.py), [`scripts/capability/interactions/extract_pairwise_epistasis.py`](../../scripts/capability/interactions/extract_pairwise_epistasis.py), [`fit_pairwise_epistasis.py`](../../scripts/capability/interactions/fit_pairwise_epistasis.py)
- Measured-singles correction and the residual panel: [`fit_measured_singles_correction.py`](../../scripts/capability/interactions/fit_measured_singles_correction.py), [`run_residual_panel.py`](../../scripts/capability/interactions/run_residual_panel.py), [`refit_likelihood_on_residual.py`](../../scripts/capability/interactions/refit_likelihood_on_residual.py)
- Contact qualification and bidirectional ablation: [`src/capability/interactions/contact_enrichment.py`](../../src/capability/interactions/contact_enrichment.py), [`scripts/capability/interactions/measure_contact_epsilon_enrichment.py`](../../scripts/capability/interactions/measure_contact_epsilon_enrichment.py), [`scripts/capability/position/measure_contact_discrimination.py`](../../scripts/capability/position/measure_contact_discrimination.py), and [`scripts/capability/position/fit_contact_bidirectional.py`](../../scripts/capability/position/fit_contact_bidirectional.py)
- The interaction lane of the shared hierarchy: [`src/capability/extensions/explicit_statistics_lanes.py`](../../src/capability/extensions/explicit_statistics_lanes.py), plus the exploratory exact pseudo-log-likelihood double-mutant cycle evaluator [`src/capability/general_plm/r3_cycle_eval.py`](../../src/capability/general_plm/r3_cycle_eval.py)

## Inputs

Local and not distributed: `data/pairwise_assets`, plus the retained contact annotations and the structure inputs the response side reads. The pairwise cohort descriptor is tracked in [`configs/explicit_statistics_cohorts.json`](../../configs/explicit_statistics_cohorts.json).

## Evidence

Local result trees: `results/R3` holds the residual panels (`pairwise_residual_full_panel_20260927`, and the indel-excluded `pairwise_residual_full_panel_indel_excluded_20260928`), the measured-singles panel, the corrected contact bootstrap and the corrected cycle labels; `results/pairwise_epistasis_20260924` retains the original extraction; `results/extensions/contact_discrimination_20261120` and `results/extensions/contact_bidirectional_20261120` hold the indexing-corrected 23-arm contact-discrimination and nested-CV bidirectional ablation readings (the defective `2026-11-14` pair was quarantined and relocated to `archive/results/extensions/` per `archive/paths.tsv`); `results/extensions/explicit_statistics_lanes_20261110/interaction` holds the lane's complete-case family; `results/extensions/general_plm_20261120` holds the exploratory 29-arm exact pseudo-log-likelihood epistasis evaluations.

## Shared boundaries

GEMME is non-identifiable in this lane by construction, because a double-mutant cycle contrast of a score that sums per-site scores is identically zero, so the hierarchy stops at profile and couplings. The lane sits near its own floor, and a small share there describes the floor rather than an equivalence. The cycle target is label-assisted in the sense that the measured singles enter the baseline, so it is not a zero-shot task. The indexing-corrected nested-CV bidirectional ablation (`contact_bidirectional_20261120`) resolves the side-by-side contact ambiguity: explicit couplings add simultaneously positive pure reverse increments across all 16 full-alphabet arms at every depth, while the four bidirectional models add complementary marginal forward signal concentrated on shallow alignments and causal models add zero forward signal; the contact reading remains bounded to short 37–72-residue domains. `prollama` and `prollama-stage-1` are ordinary single-checkpoint panel arms here; the stage-to-stage comparison belongs to post-training adaptation.
