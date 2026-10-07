# Likelihood-first information progression

This new Result 1 extension asks which declared feature blocks improve family-held-out mutation ranking after fitting retained scalar likelihood. CPU fitting is pending; no progression results are available. The frozen manuscript and historical artifacts remain untouched. The separate implementation is `src/capability/extensions/progression.py` and `scripts/capability/extensions/fit_information_progression.py`; this protocol does not prescribe an unverified command-line interface.

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

## Interpretation and limitations

These are estimable predictive contrasts, not an identifiable decomposition into exclusive biological knowledge. Feature blocks overlap: L contains mutation chemistry and fixed evolutionary substitution statistics, while S borrows L's chemistry. With zero-based P indices, strict singles give exact duplicates P1 = P4, P2 = P3 and P5 = P6 = P7; B's burden equals the sum of its substitution counts, and L's six standalone chemistry changes lie in B's substitution-count span on singles. Preserve the historical basis rather than silently pruning duplicates: ridge penalties depend on the basis even when its linear span is unchanged.

Collinearity, ridge regularization and progression order affect increments; held-out increments may be negative. A zero increment establishes neither that a model already knows the block nor that the block contains no information. Do not interpret increments as causal effects or knowledge fractions, and do not sum them into exclusive biological shares.

All required scalar ranks, feature inputs, structural annotations and folds are available for CPU fitting. No native token-NLL recovery or H200 computation is expected for this extension. Expanded structural coverage, new features and positional likelihood interpretation are outside its scope.
