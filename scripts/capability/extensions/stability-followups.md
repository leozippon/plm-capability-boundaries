# Stability ranking and measurement-channel follow-ups

The exact-support CPU preparation is complete and full channel-control qualification is running. Model-increment ranking and channel comparisons remain blocked at missing or unaccepted model outputs; no approximate reconstruction or new inference is used.

## Prepared support and boundaries

The registry contains 25,856 strict singles, 5,664 sites and 101 backgrounds/families, bound to the frozen cohort and exported sample IDs. Static control matrices, state features and original outer memberships are retained. Inner memberships are regenerated from the historical recipe and are not claimed to be recovered serialized originals.

Combined, trypsin and chymotrypsin absolute WT/mutant measurements were joined from the original parquet data under frozen QC and median aggregation. All derived channel deltas reproduce retained labels within 1e-12. The nonlinear response control G is a fold-dependent source binding, not an all-label feature matrix; its calibration uses training states only. A three-channel pilot on the first original outer fold produced finite 25,856 × 4 G blocks. Mean family-level trypsin–chymotrypsin Spearman is 0.90435, a descriptive measurement statistic rather than independent replication or equivalence evidence.

All 33 local baseline-only prediction sets passed sample/label/outer-fold alignment checks. Native scalar archives exist locally for five models, but their paired replays are still marked `reproduced_metric_mismatch` and lack independently accepted native-score provenance. The other 28 models have no local scalar product. Neither frozen stability-MSE-positive model is among the five. These facts are enumerated in the model-recovery manifest; group MSE vectors cannot supply missing prediction ranks.

## Distinct analyses

- **Same-prediction ranking:** evaluate within-background Spearman from MSE-trained control/control-plus-M OOF predictions. S2 is the historical correlation-qualified primary comparison; S provides the same-prediction ranking-versus-MSE sensitivity. Keep their inference families separate.
- **Rank-trained sensitivity:** refit standardized within-background rank targets through the declared nested workflow. This changes the training objective and must not be conflated with evaluating a new metric on the old fits.
- **Combined-trained channel reevaluation:** evaluate the same combined-trained OOF predictions against each protease label. This is not channel-specific training.
- **Channel-trained robustness:** qualify controls and fit each channel on identical variants and recorded outer partitions. Channel-specific absolute/WT labels are required for G, with training-only calibration.

Rank inference averages split contrasts within family and resamples families jointly with 10,000 draws across all 33 models. Nonestimable groups block a purported full-panel result rather than being silently deleted. Channel families include paired channel differences; a nonsignificant difference is not equivalence. Reusing the two frozen positive models is focused robustness, not independent confirmatory selection.

Channel-trained receipts bind the qualification file's path/hash, cohort and channel-label hashes, and exact S/S2 feature sets. The loader rejects missing/mismatched bindings. A primary matched-channel contrast additionally requires identical qualified S/S2 block sets across channels; channel-specific fitted G parameters are allowed. If the selected sets differ, a common baseline must be explicitly qualified before interpreting a channel difference. Channel-specific estimates alone cannot remove that confounding.

## Executable workflow

Use `scripts/capability/extensions/prepare_stability_followups.py` with the validated Python environment and bounded numerical threads. Operations are `prepare`, `calibration-pilot`, `qualify-channels`, `fit`, `rank-summary` and `matched-summary`. Fresh child output directories under the stability extension scope preserve earlier receipts. The generated `procedure.json` documents exact verified-input schemas; the current code also enforces the later qualification-binding repair.

The active baseline-only qualification command is:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
/home/lzp/miniconda3/envs/ct/bin/python scripts/capability/extensions/prepare_stability_followups.py \
  qualify-channels --threads 4 \
  --out results/extensions/phenotype_followups_20261007/stability/channel-qualification-repaired-20261007
```

This operation evaluates all three channels and the historical candidate-control ladder without M. Its start receipt binds code and inputs before loading or fitting. Completion is not a model-likelihood result. Model fits require fully verified scalar provenance, and summary operations require complete, qualified paired prediction manifests; the five discrepant replays are not automatically admitted.

Prepared outputs are under `results/extensions/phenotype_followups_20261007/stability/`, including support/label/control matrices, source bindings, partitions, model-recovery inventory, procedures, resources and verification. A scoped audit confirmed CPU qualification readiness and found a missing downstream noncombined-qualification gate; it was repaired with negative tests. The parent reran all 24 stability-followup tests successfully. Static tooling still reports environment/import-resolution and intentional fail-fast loader findings; it is not claimed clean.

Recover original scalar or paired OOF products first. Only genuinely missing model–state scoring should use H200 after access returns; scalar phenotype comparisons do not require positional likelihood arrays. Frozen manuscript results remain untouched.
