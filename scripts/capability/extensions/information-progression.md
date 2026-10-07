# Likelihood-first information progression

This new Result 1 extension asks which declared feature blocks improve family-held-out mutation ranking after fitting retained scalar likelihood. The complete two-panel CPU fit and independent numerical acceptance have passed. Profile and local-window blocks retain predictive complementarity to likelihood across the full model panel; structural ranking increments remain unresolved under the declared simultaneous inference. The frozen manuscript and historical artifacts remain untouched. The separate implementation is `src/capability/extensions/progression.py` and `scripts/capability/extensions/fit_information_progression.py`.

## Supports and feature blocks

Fit two fixed panels independently: the full anchor has 25,728 mixed single- and multiple-substitution variants in 201 assays and 163 biological families; the RSA-ready structural intersection has 2,067 strict singles in 30 Tsuboyama stability assays and 30 families. Every design within a panel uses exactly the same rows. Do not splice the full-panel likelihood/profile/window progression onto the structural endpoint: refit the complete progression on structural support. Structural comparisons describe a selected stability subset, not representative coverage of the full panel.

Concatenated letters denote these blocks; their meanings are specific to this extension, not aliases for historical design names.

| Block | Columns | Definition |
| --- | ---: | --- |
| M | 1 | Retained scalar likelihood, reranked within each panel's assays and fitted by nested ridge; the M endpoint is not raw likelihood–phenotype correlation. |
| P | 14 | Historical profile block: profile-score rank; mean WT and mutant column frequencies, minimum mutant and maximum WT frequencies; mean/minimum/maximum mutated-column entropy; mutated-column support; mean substitution log odds; WT supported-column fraction, log10 effective alignment depth, maximum query identity and mean supported-column entropy. |
| L | 111 | Historical `wall` block: mutation-centred window chemical-class composition, substitution chemistry, context chemistry and their interactions, plus fixed BLOSUM62 compatibility with window residues. This is not a site-specific profile block or the earlier k-mer block. |
| B | 444 | Sequence/substitution descriptors: 400 directed substitution counts, mean and standard deviation of relative mutation positions, substitution burden, 20 WT and 20 mutant composition fractions, and sequence length divided by 1,024. |
| S | 7 | WT RSA and six RSA × substitution-chemistry changes, using L's hydropathy, charge, volume, polarity, helix-propensity and sheet-propensity scales. |

## Designs and predictive contrasts

Let F = MPL on full support and F = MPLS on structural support. A contrast is the held-out performance of the larger design minus that of its stated comparator; for rank-MSE reverse the subtraction so positive always means improvement.

| Panel or analysis | Designs or contrasts |
| --- | --- |
| Full-panel fits | M, MP, MPL, BM, BMP, BMPL, BPL, BML |
| Structural-panel fits | M, MP, MPL, MPLS, BM, BMP, BMPL, BMPLS, BPLS, BMLS, BMPS |
| Primary progression | MP − M; MPL − MP; structural only: MPLS − MPL |
| Supplementary B-adjusted progression | BMP − BM; BMPL − BMP; structural only: BMPLS − BMPL |
| Supplementary conditional contributions | BF minus each design obtained by removing M, P, L or, on structural support, S, always retaining B |
| Supplementary B increment | BF − F |

Thus the primary question follows M → MP → MPL [→ MPLS], while the supplementary ladder follows BM → BMP → BMPL [→ BMPLS]. Drop-one comparisons condition on all other declared blocks; they do not replace the sequential question.

Project the original three seeded partitions, each with five outer and four inner family folds, onto each panel. Verify identical realized membership across designs rather than generating new folds. Rerank M, P column 0 and the phenotype target within the same retained assay rows for each panel. Use the historical ridge alpha grid (0.01, 0.1, 1, 10, 100), training-only weighted centering/scaling and inner rank-MSE selection. Tune each design separately; preserve equal family weights and equal assay weights within family.

## Inference

Primary Spearman simultaneous intervals cover all 33 models × two full-panel or three structural-panel sequential contrasts, separately for each panel. Supplementary Spearman intervals use a separate multiplicity family containing every B-adjusted sequential, conditional drop-one and B-increment contrast across the 33 models within that panel. Secondary rank-MSE intervals cover all declared contrasts across the 33 models in another separate family per panel.

Use 2,000 shared family-bootstrap draws for paired comparisons. Average seed-specific metrics within assay, then assays within biological family, giving each family equal weight. Intervals condition on fitted predictions: they do not include refitting or training-history uncertainty. Retain exact row identities, source hashes, projected and realized folds, tuning records, held-out predictions and the declared multiplicity families in new extension outputs, with CPU resource checks and runtime receipts.

## Completed findings

The table reports the range of mean Spearman increments across models and the number with a positive simultaneous 95% interval in the declared primary family. Ranges are point-estimate ranges, not confidence intervals.

| Fixed panel | Added block | Increment range | Simultaneously positive |
| --- | --- | ---: | ---: |
| Full anchor | P given M | +0.05755 to +0.41656 | 33/33 |
| Full anchor | L given MP | +0.04809 to +0.07214 | 33/33 |
| Structural subset | P given M | +0.10216 to +0.45315 | 30/33 |
| Structural subset | L given MP | +0.12782 to +0.16109 | 33/33 |
| Structural subset | S given MPL | +0.03952 to +0.05140 | 0/33 |

For example, ProteinGLM's full-anchor mean Spearman progresses from 0.38984 (M) to 0.44739 (MP) to 0.49576 (MPL); ProGen2-xlarge progresses from 0.39206 to 0.45036 to 0.49845. Thus even models with useful likelihood scores retain complementary profile and window predictors under the specified readout.

In supplementary full-anchor contrasts, P remains positive for all 33 models given BML. L given BMP is positive for 24/33. The model increment given BPL is positive for ProGen2-xlarge and ProGen3-3B (2/33) under this extension's pooled-split supplementary simultaneous family. That count does not replace the frozen manuscript's differently defined splitwise criterion.

On structural support, P remains positive for all 33 models given BMLS. L given BMP is positive for all 33, but L given BMPS is unresolved for all; this is consistent with order-dependent overlap under the tested basis, not proof that RSA contains all local information. S given BMPL and M given BPLS are likewise unresolved for all models. Primary S point estimates are positive throughout, and the secondary rank-MSE increment S given MPL is simultaneously positive for 30/33, so unresolved Spearman intervals must not be described as absence of structural information.

The primary structural simultaneous band has a common half-width of 0.10826, whereas its S increments are about 0.04–0.05. This follows the predeclared joint maximum-absolute-deviation procedure across all primary model/contrast cells; higher-variance contrasts widen the common band. These counts therefore depend on the declared inference family and do not contradict the earlier, differently defined structural-overlap analysis.

## Execution and outputs

From the repository root, use the validated Python environment:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/lzp/miniconda3/envs/ct/bin/python scripts/capability/extensions/fit_information_progression.py \
  --authorize-full --threads 2 --out results/extensions/information_progression_20261006
```

The output directory must not exist. The runner tests the extension before fitting, then saves source/code contracts, named-column inventories and label-blind redundancy censuses, exact sample IDs, projected partitions, per-cell OOF predictions and realized fold/tuning audits. Each panel receives paired assay metrics and bootstrap contrasts. Only the final `completion.json` attests a complete run; `progress.json` or individual prediction files do not.

Production completed 198 panel/model/split cells and 1,881 design prediction vectors in 5,193.73 seconds (86.56 minutes), using two CPU threads and approximately 1.93 GB peak RSS. Independent acceptance checked all 2,293 receipted outputs, declared source/code hashes, identities and 9,405 outer-fold records; recomputed metric discrepancies were at most 1.60×10⁻¹⁴. Validation did not refit models or rerun bootstrap draws. A validator-only numeric-versus-string family-ordering assertion was corrected; production outputs were unchanged.

Production artifacts are under `results/extensions/information_progression_20261006/`; the independent accepted summary is `results/extensions/information_progression_20261006_validation/result-summary.json`. These local, ignored outputs retain all model estimates and provenance; they are separate from frozen manuscript evidence.

## Interpretation and limitations

These are estimable predictive contrasts, not an identifiable decomposition into exclusive biological knowledge. Feature blocks overlap: L contains mutation chemistry and fixed evolutionary substitution statistics, while S borrows L's chemistry. With zero-based P indices, strict singles give exact duplicates P1 = P4, P2 = P3 and P5 = P6 = P7; B's burden equals the sum of its substitution counts, and L's six standalone chemistry changes lie in B's substitution-count span on singles. Preserve the historical basis rather than silently pruning duplicates: ridge penalties depend on the basis even when its linear span is unchanged.

Collinearity, ridge regularization and progression order affect increments; held-out increments may be negative. A zero increment establishes neither that a model already knows the block nor that the block contains no information. Do not interpret increments as causal effects or knowledge fractions, and do not sum them into exclusive biological shares.

All declared fits were completed from existing scalar ranks, feature inputs, structural annotations and folds on CPU. No native token-NLL recovery, new model inference or H200 computation was needed for this extension. Expanded structural coverage, new features and positional likelihood interpretation are outside its scope.
