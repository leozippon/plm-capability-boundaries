# Experiment modules

This directory defines the current scientific modules R1–R6, grouping measurements by the question each answers. It is navigation documentation only: no dataset, result tree, schema, campaign identifier or artifact moved, and the legacy `R1`–`R6` directories under `results/` keep their names and their contents.

| Module | Question | Where its detail is | Current result |
| --- | --- | --- | --- |
| [R1 统计属性 — Statistical memory](statistical-memory/README.md) | On natural sequence and on supplied variants, how does a checkpoint's likelihood behave against explicit evolutionary statistics — natural-fragment preference, the observed-residue conditional, mutation-effect ordering and the Profile/Potts/GEMME ladder? | mutation, position, readouts, baselines and the ladder, with the fragment and conditional producers | Functional predictive readouts overlap strongly with explicit statistics; this does not identify all information in the parameters. |
| [R2 定量能力 — Quantitative readouts](quantitative-readout/README.md) | Do the same checkpoints predict measured quantitative phenotypes — abundance, single-substitution stability and the added phenotypes — beyond sequence controls and explicit statistics, and how does that transfer across families? | stability, external confirmation and the explicit-statistics lane panels | Local quantitative improvements occur; additional gains across families are not consistent. |
| [R3 相互作用信息 — Interaction readouts](interaction-readout/README.md) | Do single-mutation signals resolve double-mutant non-additivity and coupling, and what do the fitted couplings say about contact? | pairwise extraction, the nested measured-singles residual panel and the contact qualification | No statistically supported additional interaction gain was found on the qualified comparisons; this is not proof of absent information. |
| [R4 非参数化记忆的使用 — Context access](context-access/README.md) | How does access to external homologous sequence affect mutation scoring against unrelated or absent context, and generation conditioned on a homologous sequence prefix? | homology retrieval, crossed scoring controls, remote-homology qualification and historical E11 conditioned generation | Predictive benefits depend on the model; E11 establishes no clear structural-confidence gain within its interface and power limits. |
| [R5 参数化记忆的生成能力 — Generative expression](generative-expression/README.md) | What do fixed weights express under free or native family-conditioned sampling, compared with natural fragments and explicit statistical samplers, and what do Profile guidance and screening add? | generation, decoding, forced-anchor generation, the generated side of evaluation and the modification-extent ladder | Generated sequences have family-statistical features but differ from family-specific statistical samples; likelihood does not always select better candidates. |
| [R6 更新与适应 — Post-training adaptation](post-training-adaptation/README.md) | What changes between a base checkpoint and its post-trained sibling, read on prediction, generation and predicted structure? | the ProLLaMA stage pair, the ProtGPT3 and ProGen2 objective-tuned pairs and their folding cohorts | Post-training changes readouts and generation, with some local confidence gains rather than demonstrated general design improvement. |

## Reading rules

The modules group questions, not designs. Two experiments in one module do not thereby share a sample, a resampling unit, a baseline or an estimator; each keeps the support, controls and interval procedure its own declaration names, and nothing is pooled across modules. A module is the right place to start looking for evidence, never the unit a number is read in. Cross-family interference and potential forgetting are explanatory hypotheses, not established mechanisms.

R1–R6 in this index name the current scientific modules. Historical `results/R*` storage paths and historical evidence-lane or campaign labels are separate namespaces, not earlier spellings of these modules. References to those older labels must be marked **historical**, with their defining document or full path; their original identifiers stay unchanged. For example, historical storage `results/R2/` supports current R1, while the historical evidence lane R3 includes stability held in historical storage `results/R4/` and `results/external_baseline/`. The crosswalk below maps storage to the current modules without renumbering any manuscript section or retained artifact.

## Legacy storage crosswalk

Every path in this table belongs to the historical storage namespace. A storage directory is not a current module, and one directory can serve several modules. The mapping records ownership of the question, not of the bytes; retired work is listed only to locate its history, not as current evidence.

| Storage path | Modules that read it |
| --- | --- |
| `results/R1/` | statistical memory (matched discrimination, native DMS, position terms, ProGen fitness and scale, native expression strata; retired pair ablation as historical material only); quantitative readouts (external confirmation); post-training adaptation (adaptation lineage, second-stage capability, joint-mode and stage qualification) |
| `results/R2/` | statistical memory (readout panels, class sweep, depth panel) |
| `results/R3/`, `results/pairwise_epistasis_20260924/` | interaction readouts |
| `results/R4/` | quantitative readouts (stability panels, gates, replay) |
| `results/R5/` | context access |
| `results/R6/` | context access (historical E11, `e11_completed_20261011`, where external homologous sequence conditions generation); generative expression (`generation_*`, `progen3_generation_*`, `progen3_robustness_*`, `conditioned_generation_*`, `e20_*`, `decoding_*`, `forcing_*`, `unconditional_diversity_*`, `ladder_*`, structure-instrument and replication runs); post-training adaptation (`stage_pair_*`, `stagepair_cross_analysis_*`, `objective_tuned_*`, cross-family structure cells) |
| `results/extensions/` | split by file: `statistical_ladder_*`, `family_distribution_*`, `family_conditionals_*` house statistical memory; `explicit_statistics_lanes_*` house quantitative readouts and, in its interaction lane, interaction readouts; `homology_context_*` belongs to context access and `generated_mutation_*` to interaction readouts; `objective_tuned_*`, `rl_environment_provenance_*` house post-training adaptation; `contact_discrimination_*` is the contact qualification of interaction readouts; `bidirectional_ablation_20261106/r1` and `r3` are statistical memory and interaction readouts, its `r2` is quantitative readouts |
| `results/external_baseline/`, `results/external_baseline_analysis/`, `results/long_cap_termination_20261110/` | post-training adaptation (fold cells, length-adjusted contrasts, the 768-residue companion cohort); the stability grid inside `external_baseline/` is the quantitative-readout panel |
| `results/shared/`, `results/audit_*` | nothing of their own; support, receipts and reporting products consumed by the modules above |

Four distinctions keep the classification explicit:

- `prollama` and `prollama-stage-1` remain single-checkpoint panel entries wherever scored; only their stage-to-stage contrast belongs primarily to current R6.
- Family-distribution and family-conditionals score fixed natural sequences: current R1, not context access or generative recovery.
- `objective_tuned_*`, `long_cap_termination_*` and length-adjusted training contrasts belong primarily to current R6; shared generation evaluation tools are cross-referenced from current R5.
- Contact evidence is auxiliary to current R3, not an independent demonstration that model parameters contain information absent from explicit statistics.

## Where the narrative is

The per-experiment record is the experiment registry and the current design is the summary; both are held locally beside the data and results and are not distributed with the code. `results/README.md` is the storage guide, `data/README.md` the input guide, and both say which names must not change.
