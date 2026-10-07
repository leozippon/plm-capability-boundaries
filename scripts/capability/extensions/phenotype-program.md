# Biological phenotype and task extension program

This program asks how broadly model-likelihood mutation information predicts biological measurements, and where that transfer weakens or changes character. It supersedes the earlier restriction to assessing one additional phenotype. It does not replace frozen manuscript results. Active work includes stability ranking and protease robustness, executed cross-phenotype consistency, local endpoint coverage, and public dataset discovery; results and qualification decisions are recorded separately as they become available.

## Evidence levels

Keep three forms of evidence distinct throughout acquisition, fitting and reporting:

1. **Existing-support task stratification:** evaluate retained held-out predictions by an independently adjudicated endpoint class. This is retrospective sensitivity of the existing cohort, not independent confirmation.
2. **Matched multi-phenotype support:** attach different experimental labels to the same variants/proteins where available. This can isolate phenotype differences from sequence/support differences, but does not create more independent proteins. Measured-phenotype conditioning is explicitly label-assisted.
3. **Independent task cohorts:** admit new biological support after exact-sequence, family and homology overlap checks. New assay names or more mutations on known proteins do not establish independence.

Likelihood must remain an input feature, not be replaced by a previously fitted phenotype prediction. Native scalar scores, within-assay score ranks and fitted predictions are different products and are not interchangeable across supports.

## Endpoint qualification

Systematically consider direct molecular activity, affinity/binding/PPI, cellular expression/trafficking/secretion, fitness/growth and other informative molecular or cellular tasks. Do not reject a useful endpoint because acquisition or new code is required. Do not promote it merely because a convenient table exists.

For each source, retain a machine-readable record of public accession/URL, access terms, version/retrieval date and file hashes; endpoint definition, units and direction; WT and mutation mapping; single-substitution, censoring and missingness rules; replicate/condition structure; protein and family counts; overlap with the anchor and other cohorts; measurement reliability; baseline qualification plan; scoring coverage; and evidence level. Unknown fields remain unknown, not inferred from dataset titles. Protein counts, independent family counts, assay counts and measurements must be separate.

Baseline features should address the endpoint's plausible non-model explanations and be qualified with held-family predictions on its admitted support. A baseline that works for MSE need not be appropriate for ranking: stability's existing S and S2 controls are a concrete example. Interface context, expression or assay conditions may require additional controls; feature availability and label-assisted components must be explicit.

Ranking is suitable for relative effects within meaningful experimental backgrounds. Quantitative error is appropriate when units, calibration, censoring and measurement error support it. Rank-target MSE is not physical phenotype error, and heterogeneous assay values must not be pooled into an artificial common MSE. Classification endpoints need declared thresholds, class prevalence and suitable discrimination/calibration metrics. Additional metrics must answer distinct questions rather than multiply chances of significance.

Enough independent, reliable biological support and a useful qualified baseline—not an arbitrary mutation count—determine whether a cohort can support a main-text conclusion. Small single-family or few-protein studies can remain useful mechanistic or descriptive sensitivities without being presented as broad generalization.

## Execution and inference boundaries

Preserve one explicit ordered variant/support registry within every matched comparison, including exact WT/state identifiers, source hashes, endpoint labels and family membership. Reuse original partitions where verifiable; reconstructed or new partitions must be identified. Keep training-only tuning/calibration and paired control versus control-plus-likelihood evaluation.

Primary endpoints, contrasts and multiplicity families must be declared before reading their model effects. Resample biological families/backgrounds jointly across matched designs and dependent channels. Fold seeds are sensitivity analyses, not independent replicates. Cross-model analyses are descriptive fixed-panel or release-aware summaries; checkpoints and related releases are not independent training replicates. Cross-endpoint claims require an explicit dependence/multiplicity account rather than counting separately positive tests as independent confirmations.

Use retained predictions or scalar scores only after identity, scale, split and source-provenance checks. Do not reconstruct per-variant order from group MSE, mix separately normalized rank tables as native scores, or disguise discrepant local replays as original recovered outputs. Missing-input branches must emit exact recovery requirements and stop the affected analysis without preventing valid independent preparation.

All feasible annotation, acquisition, support construction, baseline work, CPU fitting and resampling may proceed locally with resource receipts. New substantial model inference waits for H200 access. Before requesting scoring, join existing coverage and list only genuinely missing model–state combinations, with native checkpoint/interface, sequence/window identity and output requirements. Scalar-phenotype work does not require token-level likelihood arrays unless a separately declared positional question is being tested.

## Existing-support stratification protocol

An immediate CPU sensitivity uses the completed local progression panel's fixed BPL/BMPL held-out predictions on its exact 25,728 rows, 201 assays and 163 families. These arrays are fitted predictions, not raw model features. Join assay metadata to form Activity, Binding, Expression, OrganismalFitness and Stability strata; these are coarse source labels, not yet adjudicated mechanistically pure task collections.

The primary contrast is within-assay Spearman(BMPL) minus Spearman(BPL). Average split contrasts within assay and assays within family for each category. Use 10,000 shared draws of the original biological family IDs across categories and models, preserving dependence when a family spans categories. Declare one simultaneous Spearman family across the supported 33-model × five-category comparisons; do not infer a main-text task ranking from unadjusted per-category positives. Persist category-specific coverage and explicit treatment of a bootstrap draw with no contributing family. Any secondary rank-target squared error is not physical phenotype error.

This evaluates the same mixed-cohort-trained readouts within strata; it neither refits on each phenotype nor establishes category-specific baseline qualification. It can guide which tasks need better controls or independent cohorts, while remaining separate from external confirmation and from the model-level cross-phenotype consistency analysis.

## Completed model-level consistency

The [cross-phenotype consistency analysis](phenotype-consistency.md) has been executed and independently reproduced from retained model-level results. Positive overall release-level associations persist under release deletion and shared-parent exclusion, but within-category associations are heterogeneous. This is descriptive fixed-panel evidence, not checkpoint-independent inference or a manuscript-placement decision.

## Work products

New outputs live under `results/extensions/phenotype_followups_20261007/`, separated into stability, consistency, discovery and subsequent task-specific preparation/results. Public payloads, when needed, are staged separately under `data/phenotype_followups_20261007/`. These local data/results remain ignored; code, tests and operating documentation are committed normally. Frozen outputs and unrelated manuscript/reporting changes are outside the write scope.

The earlier [Result 3 assessment](result3-followup-assessment.md) supplies historical status and known limitations; it is not a veto on broader acquisition. Dataset discovery, qualification, readiness and completed numerical results are different stages and must not be conflated.
