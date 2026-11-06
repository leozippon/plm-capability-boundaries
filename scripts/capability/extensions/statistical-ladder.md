# Statistical-baseline ladder

This extension adds the declared R1 statistical baselines to the frozen mutation-ranking panel. The scientific design, support, splits, aggregation and uncertainty are fixed in `docs/statistical-baseline-ladder.md`; this note covers only how to run it and what the inputs and outputs are. The implementation is `src/capability/extensions/statistical_ladder.py` with the runner `scripts/capability/extensions/fit_statistical_ladder.py`.

## Support, arms and designs

Every design is fitted with the historical nested ridge on the original three seeded five-outer/four-inner biological-family partitions over the complete support of 25,728 variants, 201 assays, 163 families and 174 wild types. No split is regenerated, no row is dropped and no variant is admitted as an independent sample: the seeds are averaged within assay, assays are averaged within family and families are then equally weighted.

The three arms of a comparison are the traditional control `C`, the model readout `M` and their union `CM`. `M` is the retained scalar likelihood reranked within assay; `P` is the 14 historical profile features; `C0` is the historical `B+P+L` control. The two new scalars enter exactly like `M`, as one assay-reranked column each, with the raw values kept only for the descriptive layer.

| Control set | C | M | CM |
| --- | --- | --- | --- |
| `P` | `P` | `M` | `MP` |
| `P+E` | `PE` | `M` | `MPE` |
| `P+G` | `PG` | `M` | `MPG` |
| `P+E+G` | `PEG` | `M` | `MPEG` |
| `C0` | `BPL` | `M` | `BMPL` |
| `C0+E+G` | `BPLEG` | `M` | `BMPLEG` |

Design names are their blocks concatenated in the canonical order `B, M, P, L, E, G`, so `MP` and `BMPL` are the historical designs with the same name. Each set contributes two paired directions — joint minus the traditional control, and joint minus the model-only readout — with the subtraction reversed for rank MSE, so a positive value always means improvement. Six sets, 33 models and two directions give the declared 396-hypothesis family per metric, with Spearman and rank MSE declared as separate families.

In the full stage, `M`, `MP`, `BPL` and `BMPL` are not refitted. They are read from `results/extensions/information_progression_20261006/full` after their archive digest, sample identities, target, realized fold membership and per-assay metrics are checked against that frozen run; a single mismatch refuses the run. Designs without `M` are model-independent and are fitted once per seed. The primary family uses the shared `bidirectional.paired_bands` native maximum-centred-deviation bootstrap (10,000 draws, seed 20261106) and the project's existing fixed-standard-error helper as the explicitly labelled standardized-maxT sensitivity.

## Upstream score tables

The full ladder reads the two upstream scoring documents through a common input directory. Its `potts` and `gemme` entries may be symbolic links to the accepted scoring directories; the converged Potts execution writes to `potts-v2`, not the earlier `potts` pilot. For example:

```
results/extensions/statistical_ladder_20261106/accepted-scores/
  inputs/{manifest.json,support.json,variant_table.json,msa/}   # optional cross-check
  potts/scores.json    # required: per-sample records, column potts_delta
  gemme/scores.json    # required: per-sample records, column gemme_score
  potts/scores.npz     # optional: cross-checked against its document
  gemme/scores.npz
```

A table is a JSON array of records, an object listing them under `samples`, `records`, `variants` or `rows`, or an object keyed by `sample_id`. Every record needs a unique `sample_id`; the required column must be finite on every sample. Records may carry `profile_delta`, which is checked against the retained cohort profile score, and textual annotations such as the Potts background `status`, which are never read as scores. Whatever metadata the document declares outside its sample records is bound by digest and echoed into `scores.json`; a document without metadata is refused, so a bare array dump cannot stand in for an official baseline table. When `scores.npz` is also present its sample order and numeric columns must agree with the document.

Missing or extra samples, duplicate keys, non-finite values and absent columns all fail immediately with a counted reason instead of imputing, truncating or reordering. If `inputs/variant_table.json` exists, its `sample_id` order must equal the frozen support order.

A finite number is not by itself evidence of a finished baseline, so the full stage also enforces the completion contract both exporters declare. Potts must report `counts.complete` true, 25,728 samples all carrying a Potts value, and 174 backgrounds expected, resolved and fitted with none pending, failed or unconverged; every row's `status` must be `fitted`, which keeps the declared fixed-low-depth policy branch admissible because it is a fitted background with its own regularization. GEMME must report `counts.complete` true, 25,728 samples expected and scored, 174 wild types expected and complete, an empty `incomplete` list, and three aggregated runs per row; every row's `status` must be `scored` with `n_runs` equal to the declared `repeats`. An all-zero official score from a shallow alignment is a legitimate `scored` value and is never dropped: it enters the table as a constant predictor, and only the descriptive layer reports it as not estimable. A document that declares completion nowhere, or whose counts, row statuses and array dump disagree, is refused; so is a table whose finite values came from an unconverged background or a partly scored wild type, because those would otherwise become baseline results.

Potts is the retained plmc full-sequence energy difference. GEMME is the mean of the declared three official runs per wild type, ranked within assay exactly like `M`; the run-to-run spread is tool stochasticity (JET2 samples cluster partitions and exposes no seed), not biological replication, so the runs are averaged once upstream and never resampled here. The official multi-mutant GEMME value is the same run's per-site sum, so it is accepted as the official baseline but is not read as a non-additive epistasis score. `--potts-column` and `--gemme-column` override the column names if an upstream table differs.

## Running

The runner executes `tests/extensions/test_statistical_ladder.py` before touching any data and refuses to start if those tests fail. Output directories are never reused.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/lzp/miniconda3/envs/ct/bin/python scripts/capability/extensions/fit_statistical_ladder.py \
  --stage profile_only --threads 2 \
  --out results/extensions/statistical_ladder_20261106/checks/profile-replay
```

`--stage profile_only` is a declared partial pilot: only `P`, `M`, `P+M` and the original `C0` control are computed, no Potts or GEMME table is read, and the receipt is written as `partial_profile_only` with `ladder_complete: false`. It also refits `M` and `MP` for every model it covers and requires bit-exact reproduction of the archived predictions, the archived design-matrix digests and the archived alpha choices. That record is `reuse-pilot.json`.

The complete ladder additionally needs both score tables, all 33 models and all three seeds:

```bash
... fit_statistical_ladder.py --stage full --threads 2 \
  --scores results/extensions/statistical_ladder_20261106/accepted-scores \
  --reuse-pilot results/extensions/statistical_ladder_20261106/checks/profile-replay/reuse-pilot.json \
  --out results/extensions/statistical_ladder_20261106/full
```

The full stage requires a passing reuse pilot that covers every model in the run and used the same number of BLAS threads, because reuse is only exact under the same numerics. `profile-panel` holds the accepted profile stage exactly as it was run; its receipt binds the digests of the source that produced it, so the current runner refuses it as not matching the current code and it stays the accepted original evidence rather than a launch permit. The freshly rerun pilot under `checks/profile-replay` is the one the full stage reads. A 33-model `profile_only` run costs about five minutes and is the cheapest way to produce that receipt; the full stage's own new fits, dominated by the two wide `C0+E+G` designs, are the remaining cost.

## Outputs

`contract.json` records the frozen registry, realized scope, support digest, declared hypothesis family, bootstrap, sensitivity, reuse policy, code hashes and provenance. `reuse-verification.json` lists each reused cell with its archive digest and the largest recomputed metric difference; `reuse-pilot.json` holds the reproduction evidence; `scores.json` binds the two upstream tables and their metadata digests. `cells/` holds the newly fitted designs only, each archive read back through the same verifier that admitted the reused ones. `assay-metrics.json` holds per-assay held-out Spearman and rank MSE, `aggregate-<metric>.npz` the family-level contrast and absolute matrices with their labels, `contrasts.json` the per-hypothesis points, pointwise and simultaneous intervals with verdicts plus the standardized-maxT sensitivity, and `zero-shot.json` the unsupervised assay-internal ranking of the raw profile, Potts and GEMME scores. Only `completion.json` attests a status: `complete` for the full ladder, `partial_profile_only` otherwise.

An incomplete or unconverged upstream table is refused even when its scores are finite. Code changes invalidate an existing reuse receipt because `reuse-pilot.json` binds the script and module digests; run the `profile_only` stage into a fresh `checks/profile-replay` after such a change and leave the accepted `profile-panel` untouched. `--models` can limit that stage to a smoke test, but a subset receipt cannot admit the full 33-model ladder. Execution status and scientific findings belong in the project log and evidence summary rather than this usage guide.

## Limits

The ladder reports predictive contrasts under the historical blocks, training-only scaling and tuned ridge. Overlapping feature content, the retained basis and the penalty geometry all affect increments, so they are neither exclusive knowledge nor additive shares, and an unresolved interval is not equivalence. Intervals condition on fitted held-out predictions and exclude refitting. Potts and GEMME enter as single scalars, so per-position and per-mutation structure stays outside this analysis. These contrasts cannot by themselves establish a shortfall in the distributional bias a design method would need.
