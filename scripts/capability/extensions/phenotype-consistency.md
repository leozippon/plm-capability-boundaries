# Cross-phenotype consistency of likelihood increments

The fixed-panel descriptive analysis is complete. It compares retained ProteinGym ranking, Domainome abundance ranking/MSE and stability MSE increments without new fitting or model inference. No manuscript-placement decision is made here; these associations are not an independent test of a shared latent capability.

## Results

Three split-specific point estimates are averaged before comparing models. Release means give equal weight to checkpoints within each of the sixteen previously declared release groups. The following Spearman correlations describe effect-size ordering; leave-one-release-out ranges are sensitivity ranges, not confidence intervals.

| Endpoint pair | 33 checkpoints | 16 release means | Leave-one-release-out range | Exclude Llama2 and ProLLaMA releases |
| --- | ---: | ---: | ---: | ---: |
| ProteinGym rank–abundance rank | 0.824 | 0.871 | 0.843–0.907 | 0.899 |
| ProteinGym rank–stability MSE | 0.793 | 0.762 | 0.711–0.886 | 0.820 |
| Abundance rank–stability MSE | 0.809 | 0.782 | 0.736–0.864 | 0.776 |

The complete output also retains abundance MSE comparisons and descriptive Pearson coefficients. The fourteen ProteinGym-ranking positives exactly match the fourteen abundance-ranking positives under the respective historical criteria. ProGen3-3B and ProLLaMA Stage 2 are positive for both ranking endpoints and stability MSE, but neither is abundance-MSE positive; its two positives are ProGen2-xlarge and ProteinGLM. Positivity sets and effect-size correlations answer different questions.

A supplementary sensitivity uses the frozen broad release categories without redefining them. Each category contains eight releases:

| Endpoint pair | Protein-specialized/adapted releases | General-purpose text/scientific language–protein releases |
| --- | ---: | ---: |
| ProteinGym rank–abundance rank | 0.714 | 0.238 |
| ProteinGym rank–stability MSE | 0.429 | 0.690 |
| Abundance rank–stability MSE | 0.857 | 0.238 |

Thus broad-category separation is not the entire descriptive pattern, but the overall association conceals substantial heterogeneity. The small strata and restricted effect ranges prevent strong conclusions about differences between these correlations. The frozen category names are not randomized protein-exposure or independent-training labels.

## Interpretation boundaries

There are no checkpoint-iid or release-iid p-values, confidence intervals or bootstrap claims. Related checkpoints and releases share ancestry; the sixteen groups are not sixteen independent training histories. No model-level association establishes a causal common mechanism or guarantees prediction on another phenotype.

Baselines differ: ProteinGym uses qualified local/profile controls, abundance uses checkpoint-specific S or S_T, and stability uses its MSE-qualified S. Endpoint units and original positivity criteria differ. These effect sizes must not be pooled into one capability score, and unresolved effects are not zeros. ProteinGym includes stability assays; shared biological support was not removed in this descriptive run. A biological-group bootstrap would require complete joint group vectors and an explicit overlap map, not scalar confidence intervals.

## Reproduction and provenance

Use `scripts/capability/extensions/analyse_phenotype_consistency.py --out NEW_DIRECTORY` with the validated Python environment. The CLI pins numerical work to one CPU thread and refuses an existing output directory. `--within-category-source COMPLETED_DIRECTORY` produces only the additional frozen-category sensitivity, after verifying the initial receipt.

Outputs are under `results/extensions/phenotype_followups_20261007/consistency/`: joined checkpoint effects, release means, source cells, all pairwise statistics, leave-one-release-out results, shared-parent sensitivity and source/resource/completion receipts. `within_category/` contains the supplementary tables and labelled scatter source. Earlier results were not overwritten. A parent execution with the final source reproduced all initial association and sensitivity point values exactly in `parent_replay_validation/`, and hashes for all three receipt sets were verified. Four tests passed, including missing/duplicate support, strata, ties and constants.

See the [phenotype program](phenotype-program.md) for the separate task-stratification, stability and independent-cohort workstreams. This model-level description does not fill missing stability predictions or establish new task-level generalization.
