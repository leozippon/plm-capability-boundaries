# Membrane-context phenotype preparation

Identity and measurement-QC preparation is complete for three protein targets and eight source channels. This is a focused multi-label panel, not eight independent proteins or a broad family-generalization cohort. No model inference or fitting was used in preparation.

## Accepted support

| Protein/channel | Accepted observations | Distinct protein states |
| --- | ---: | ---: |
| RHO surface-antibody method | 6,341 | 6,341 |
| RHO membrane-proximity method | 6,592 | 6,592 |
| F9 strep-II channel | 8,525 | 8,525 |
| F9 heavy-chain channel | 8,527 | 8,527 |
| F9 light-chain channel | 8,528 | 8,528 |
| F9 F9-Gla channel | 8,528 | 8,528 |
| F9 pan-Gla channel | 8,528 | 8,528 |
| SGCA surface expression | 2,482 | 2,257 |

The exact accepted intersections are 6,340 RHO states across both methods and 8,524 F9 states across all five channels. Source-channel observations, missingness, QC exclusions, uncertainty and replicate fields remain row-preserved. The 17,590 exported candidate-state identities include states with missing measurements; this is not a request to infer scores for every state.

RHO exactly matches the existing `OPSD_HUMAN_Wan_2019` anchor. It is new measurement evidence on an existing protein, not a new independent protein. No exact/containment anchor matches were found for F9 or SGCA; their homology exclusion and final family independence are not established. Repeated nucleotide encodings of the same SGCA protein state remain dependent observations, not independent proteins or silently averaged labels.

## Measurements and identity gates

RHO method 1 is a surface-antibody readout and method 2 a membrane-proximity reporter. Method-specific source QC, SE and discordance flags are retained. The source reports two and eight study/method replicates respectively, not per-variant counts or independent proteins. Method 2's RNA barcode is a reporter output, not a native RHO RNA-abundance phenotype.

F9 MultiSTEP supplies secretion-related and Gla-sensitive/PTM channels, not direct coagulation catalytic activity. Source normalization and tile/biological-replicate fields are retained. Four finite-score observations with only one replicate and missing SE are excluded rather than promoted as reliable measurements.

SGCA was initially blocked too broadly. The source explicitly defines a human coding target and all possible coding SNVs. Its 1,161 coding nucleotides specify a 387-aa target; nucleotide references, sites, codon numbers, translated WT/mutant residues, protein HGVS and consequences agree across all 3,483 observations, covering every coding nucleotide and codon. This is source-anchored translation, not a guessed reading frame or isoform, and no terminal stop is required for the supplied protein target. Initiator/start-site, synonymous and nonsense observations remain outside the single-missense support.

The malformed SGCA auxiliary tail remains quarantined. It does not invalidate the independently checked measured prefix containing score, three MLS replicate values and sigma. Shifted auxiliary read counts/predictors were not repaired or used as labels. Sigma is not established as SE, score normalization is incompletely documented, and the detailed FLAG-construct context remains unknown. Native target identity and full experimental-construct identity are distinct claims.

## Outputs and validation

Run `scripts/capability/extensions/prepare_membrane_cohort.py` with a fresh output directory under the membrane extension scope. The accepted corrected outputs are `results/extensions/phenotype_followups_20261007/membrane/source-anchored-sgca/`; original outputs were preserved. They include source/WT/state manifests, row-preserving measurements, paired-channel intersections, exclusions, source-coded SGCA validation, resource/QC receipts and output hashes.

The parent checked all 17 output hashes and three implementation hashes, all preserved source/output bindings, and exact equality of 62,898 RHO/F9 observations across old and corrected preparation. Their original 15,333 candidate identities and channel intersections were unchanged. Twenty-three membrane tests passed, including negative coding-annotation checks and quarantine of unused malformed fields. Editor/import diagnostics are not claimed clean merely because runtime tests pass.

The [phenotype program](phenotype-program.md) treats this panel as a candidate paired-label diagnostic. Only three targets are represented, RHO overlaps the anchor, and phenotype-specific controls, model-state coverage and broader independent support remain separate requirements. Native full-protein state equality does not equate DNA genotypes, reporter constructs, or measurement mechanisms.
