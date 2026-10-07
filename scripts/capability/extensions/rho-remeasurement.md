# Fixed-readout RHO remeasurement

This completed sensitivity applies existing held-out predictions to new measurements of the same RHO protein states. It is not a new protein/family, newly fitted likelihood increment, or independently qualified phenotype baseline. No fitting, model inference or noise resampling was performed.

## Matched comparison

The new RHO WT exactly matches `OPSD_HUMAN_Wan_2019`. Its original all-model support contains 128 variants. The surface-antibody method admits 123 of those states; the membrane-proximity method admits all 128. Both methods are therefore evaluated on the same **123-state intersection**, retaining all 33 models and three historical split seeds. Exact full protein-state hashes, original sample/target order, source receipts and held-family partitions were checked.

BPL contains the existing sequence/substitution, profile and local-window controls; BMPL additionally contains likelihood. The comparison is Spearman of their previously fitted predictions against each new measured endpoint, followed by BMPL minus BPL and paired method differences. Prediction features and coefficients are unchanged. Evaluation ranks are computed on the common support; this is not a new fit on the remeasured labels.

## Descriptive findings

The two measured methods have Spearman **0.77655** on the matched states. Correlations with the old labels are **0.76525** for surface antibody and **0.81241** for membrane proximity. Eleven source-discordant states remain in the matched set rather than being removed because of disagreement.

| Quantity | Surface-antibody method | Membrane-proximity method |
| --- | ---: | ---: |
| Mean-seed BMPL−BPL increment range across models | −0.01381 to +0.08128 | −0.02193 to +0.07556 |
| Positive / negative point estimates | 18 / 15 | 18 / 15 |

Equal sign counts do not imply identical model behavior: sixteen models are positive for both methods, thirteen negative for both, and four change sign. All model/seed values and method differences are retained without selecting favorable models. These are descriptive signs, not significant-effect counts.

Only one protein/family is represented. There are no population confidence intervals, family bootstrap, checkpoint-iid tests or broad generalization claims. The source reports two versus eight study/method replicates, not per-variant replicate counts or independent proteins. Original SE and QC fields remain descriptive; no independent Gaussian noise replicates were invented. Protein-state equality does not equate DNA genotypes, reporter constructs or measurement mechanisms. Rank-scale predictions are not evaluated as physical-unit MSE.

## Reproduction and acceptance

Use `scripts/capability/extensions/analyse_rho_remeasurement.py --out NEW_DIRECTORY` with the accepted membrane preparation and local progression outputs available. Relative CLI paths are normalized once relative to the project root; source paths are independent of the selected preparation output directory. The accepted producer's code/output hashes and source receipts are required.

Final artifacts are `results/extensions/phenotype_followups_20261007/remeasurement/final-verified/`. The corrected run completed in 21.98 seconds on two CPU threads. Nine tests passed. Parent acceptance verified all fourteen output hashes and three implementation hashes; model/seed contrasts, summary, coverage and paired row registry are byte-identical to the initial execution. The final receipt correctly binds the accepted membrane product and method-specific replication metadata. A failed relative/absolute-path attempt is preserved separately and contributes no scientific output.

See [membrane preparation](membrane-cohort.md) and the [broader program](phenotype-program.md). This result supplies focused remeasurement evidence and does not substitute for the still-blocked full stability/protease model comparisons.
