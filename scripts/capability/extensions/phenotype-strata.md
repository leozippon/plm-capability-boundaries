# Existing-support phenotype strata

This completed analysis evaluates fixed held-out predictions within ProteinGym metadata categories. It is retrospective task-stratified sensitivity, not an independent phenotype cohort, new endpoint-specific fit, or replacement for the separate Domainome and 101-family stability panels.

## Support and inference

All 33 models and three split seeds use the same 25,728 variants in 201 assays and 163 fitting families. The support contains 20,568 strict singles and 5,160 multiple substitutions. The larger registry's other 1,983 rows are outside this common OOF support. Direct sample/assay/mutation/group reconciliation confirms that exported group IDs and the production fitting groups agree exactly; an earlier 164-group inventory statement was an inventory error, not a data discrepancy.

The contrast is within-assay Spearman(BMPL) minus Spearman(BPL), using the existing local progression's fitted prediction arrays. Split contrasts are averaged within assay, then assays within category/family. Ten thousand shared resamples of the original family IDs preserve dependence across models and categories, including families represented in more than one category. One simultaneous family covers all 165 model/category comparisons using a centered maximum standardized deviation with fixed category-specific standard errors. One draw lacking any represented family for a category was jointly rejected and redrawn, as declared; missing category support was never treated as zero evidence.

## Results

Ranges below are model-specific point-estimate ranges, not confidence intervals. Counts require positive simultaneous 95% lower bounds across the joint 165-contrast family.

| Metadata category | Assays | Families represented | Variants | Increment range | Positive models |
| --- | ---: | ---: | ---: | ---: | ---: |
| Activity | 38 | 34 | 4,864 | −0.00064 to +0.02928 | 2/33 |
| Binding | 12 | 11 | 1,536 | −0.00079 to +0.03390 | 1/33 |
| Expression | 16 | 16 | 2,048 | −0.00061 to +0.03800 | 7/33 |
| Organismal fitness | 69 | 53 | 8,832 | −0.00050 to +0.03179 | 10/33 |
| Stability | 66 | 66 | 8,448 | −0.00029 to +0.01538 | 0/33 |

Activity positives are ProGen2-base and ProteinGLM; the binding positive is ProGen3-112M. Full model/category estimates, pointwise and simultaneous intervals, and per-class baseline performance are retained in the output tables. No model/category simultaneous interval is wholly negative.

These findings locate unevenly resolved residual likelihood signal within the existing panel. They do not rank intrinsic task difficulty: category sizes, uncertainty, assay composition and effect ranges differ. Stability here is the ProteinGym metadata stratum, not the separately qualified external stability experiment. Its unresolved increments neither replace that experiment nor establish no stability information.

## Limitations and reproduction

The metadata classes mix measurement types; Activity is not uniformly direct catalytic activity, and Binding is not uniformly direct affinity. Replicate and noise qualification was not added. All readouts were trained on the same mixed population, and their baseline was qualified for the full anchor rather than independently for each category. Evaluation targets are standardized ranks; no physical-unit MSE claim is made. Uncertainty conditions on fitted predictions and does not include retraining or tuning uncertainty.

Run `scripts/capability/extensions/analyse_phenotype_strata.py --out NEW_DIRECTORY` with the validated Python environment and two numerical threads. The runner refuses overwrites and verifies producer artifacts, sample/target identities, exported group identities and nested fold membership. Five tests cover alignment, group reconciliation, seed/family aggregation, shared-category resampling and nonestimable inputs.

Final results are under `results/extensions/phenotype_followups_20261007/strata/final/`. The parent verified every final output digest and exact agreement of all numerical estimates/intervals with the prior execution after replacing the erroneous inventory caveat by an executable reconciliation. Earlier outputs remain separate. No new model fitting, inference or H200 access was used.

Use these results to guide the [broader phenotype program](phenotype-program.md), including stronger endpoint-specific controls and genuinely independent biological support, not as five new external confirmations.
