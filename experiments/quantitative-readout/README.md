# R2 定量能力 — Quantitative readouts of measured phenotypes

Do released checkpoints predict measured quantitative phenotypes of the same proteins — abundance, single-substitution stability and the added phenotypes — beyond sequence controls and explicit evolutionary statistics, and how far does a family-fitted readout transfer? This module holds the abundance and stability endpoints and the panel machinery that reads prediction as a conditional increment.

## The question

Each endpoint supplies its own labels, support and metric, and the answer is stated as an increment over one named rung of a control hierarchy rather than as an absolute score. A readout fitted in training folds is label-informed by construction, so a resolved reverse increment says the controls add beyond the likelihood and a forward bound says how much the likelihood can add beyond them; the two are read apart. The added phenotype strata are separate panels on their own cohorts rather than extensions of either endpoint.

## Code

- Stability endpoints, gates and the matched-singles machinery: [`src/capability/stability/ranking_error.py`](../../src/capability/stability/ranking_error.py), with producers under [`scripts/capability/stability/fit_stability_ranking_panel.py`](../../scripts/capability/stability/fit_stability_ranking_panel.py)
- Abundance confirmation and its endpoint qualification: [`src/capability/mutation/external_confirmation.py`](../../src/capability/mutation/external_confirmation.py), [`scripts/capability/mutation/declare_external_confirmation.py`](../../scripts/capability/mutation/declare_external_confirmation.py), [`scripts/capability/mutation/qualify_external_confirmation_endpoint.py`](../../scripts/capability/mutation/qualify_external_confirmation_endpoint.py)
- Roles, rungs, matched support and the simultaneous band shared by every lane: [`src/capability/extensions/explicit_statistics_lanes.py`](../../src/capability/extensions/explicit_statistics_lanes.py)
- The lane runner and its per-lane adapters: [`scripts/capability/extensions/explicit_statistics_lanes/fit_explicit_statistics_lanes.py`](../../scripts/capability/extensions/explicit_statistics_lanes/fit_explicit_statistics_lanes.py), [`explicit_statistics_lane_adapters.py`](../../scripts/capability/extensions/explicit_statistics_lanes/explicit_statistics_lane_adapters.py)
- Added-phenotype breadth and strata: [`scripts/capability/extensions/fit_phenotype_breadth.py`](../../scripts/capability/extensions/fit_phenotype_breadth.py), [`analyse_phenotype_strata.py`](../../scripts/capability/extensions/analyse_phenotype_strata.py)

## Inputs

Local and not distributed: `data/megascale_tsuboyama2023`, `data/domainome_beltran2025`, `data/phenotype_followups_20261007`, with the added-phenotype cohorts staged beside them. The cohort descriptors are tracked in [`configs/explicit_statistics_cohorts.json`](../../configs/explicit_statistics_cohorts.json).

## Evidence

Local result trees: `results/R4` for the stability panels, gates and replay; `results/R1/external_confirmation_20260924` for abundance; `results/extensions/explicit_statistics_lanes_20261110` for the primary complete-case families and the archived-set bounds, with its `run-note.md` carrying the rung tables and the added-phenotype strata under `results/extensions/explicit_statistics_lanes_2026111{1,2}`; `results/extensions/bidirectional_ablation_20261106/r2` for the earlier abundance and stability ablation. The stability grid the summary reads as its phenotype lane lives under `results/external_baseline/`.

## Shared boundaries

Stability's baselines differ by what entered the arm, not by column list: its response block reads composition and a bounded profile block, so a matched baseline with no profile column is not profile-free. The complete case keeps only the rows where every block of the hierarchy exists, and the groups it costs travel with every number. The interaction lane reuses this module's declaration and runner but belongs to interaction readouts. Nothing here is pooled across endpoints, strata or channels, and a stratum under the resampling floor is reported unresolved rather than widened.
