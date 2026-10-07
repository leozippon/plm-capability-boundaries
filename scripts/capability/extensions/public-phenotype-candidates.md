# Public task candidates and qualification priorities

The 2026-10-07 discovery pass found credible routes beyond the old local abundance/stability inventory. It acquired approximately 26 MB of recorded public transfers, plus bounded decompressed binding samples, and retained a 39-candidate registry with source URLs, access terms, dates, hashes, endpoint descriptions and unresolved gates. These are candidate and data-preparation products, not model results or proof of independent family coverage.

## Candidate task families

| Task family | Concrete public support | Measurement and independence gates | Current action |
| --- | --- | --- | --- |
| Direct biochemical activity | VenusMutHub releases 130 single-mutant activity assay files; these are not 130 enzymes. Samples include exact mutant sequences and scores. | Recover file-to-publication mapping, WT/conditions, direction, units and replicates; separate kcat, kcat/Km, Km, vmax and relative activity. The advertised DOI map is missing. | Acquire/validate activity files and build exact-state/source-readiness manifests; no inference before endpoint provenance is adequate. |
| Physical PPI affinity/kinetics | Local SKEMPI2; PPB-Affinity provides additional packaging/source context. | Duplicate conditions and shared partners/homologs reduce independent support. PPB-Affinity and Venus PPI largely reuse SKEMPI, not new confirmation. Kd, kon and koff answer different questions. | Binding affinity filtering and exact partner/state mapping completed; homology, construct and baseline qualification remain gates. |
| Endogenous cellular fitness | Twelve concrete SGE gene candidates with public maps/metadata, including published BAP1, RAD51C and VHL studies. | Registered SNVs/indels are not single protein substitutions. Splicing/RNA and repeated nucleotide encodings require explicit treatment; genes are not necessarily separate families. Clinical/RF classifier aggregates are not experimental fitness. | Acquire raw experimental scores and prepare exact protein-state mappings where supported; preserve blocked transcript/source cases. |
| Ligand/PPI reporters | BindingGYM registers 25 assays; bounded samples and partner metadata acquired. Additional INSR insulin-binding and MLH1–PMS2 records have replicate/error information. | Mixed enrichment/fitness/affinity scales; repeated protein/partner lineages and anchor overlap. PDB-aligned X residues require original WT recovery, not imputation. | Qualification-ready second-tier panel; preserve source-specific metrics and partner grouping before scoring. |
| Trafficking, secretion and PTM | RHO two-method trafficking; all five F9 MultiSTEP channels; SGCA surface expression; newer KCNH2 maps. | Multiple methods/antibodies are not independent proteins. KCNH2 matches an existing anchor protein; SGCA annotation/CSV quality needs validation. F9 secretion/PTM is not catalytic clotting activity. | Prepare a focused RHO/F9/SGCA aligned-label panel, explicitly too small by itself for broad family-generalization claims. |
| Other cellular function | Public complementation, glycosylation, transcriptional, mitophagy and signalling maps, including G6PD, ADSL, FKRP/LARGE1, CRX, PRKN and TYK2. | Do not call all reporters direct enzyme activity. Some recent records lack a linked primary paper; inferred activity and raw growth are different endpoints. | Retain as named candidate classes; expand after source and family qualification rather than pool heterogeneous labels. |
| Additional physical/molecular contrasts | Nabe/ProNAB nucleic-acid affinity; Venus selectivity; IAPP nucleation reporter. | Access/license constraints, protein-versus-nucleotide mutations, signed selectivity, condition dependence and sparse independent backgrounds. | Conditional follow-up; no broad main-text eligibility inferred from catalog totals. |

## Important findings from actual source inspection

- The current ProteinGym v1.3 metadata still has the same 217 assay IDs as the old local reference: zero added/removed IDs in the actual join. Its newer baseline releases are not a new experimental cohort.
- The MaveDB catalog is useful for acquisition, not an independence label. Complete keyword result pages were retained for binding, activity, trafficking, secretion and saturation genome editing; some hits are unrelated labels or derived classifiers.
- The Venus release separates single-mutant directories, but sampled assay files lack source units/replicate/condition columns. Two differently named PPI samples share one WT, while an apparent activity/selectivity filename pair does not. Sequence and source joins must replace filename-based assumptions.
- The PPB-Affinity workbook's syntactic single-substitution subset is dominated by SKEMPI-derived records, with an additional narrow ATLAS subset. It cannot be counted as independent replication of the local affinity cohort.
- BindingGYM has 25 registered assays but 28 archived score files. Sampled labels include both singles and multis, and one sampled chain contains 37 X residues. Neither archive-file count nor PDB-aligned sequence is an automatic model-ready cohort.
- F9 has 8,528 finite missense scores per channel, not five independent proteins. The RHO author table enumerates all 6,612 possible missense variants with method-specific missingness and error fields. Candidate SGCA records include multiple nucleotide encodings of the same protein state and an auxiliary CSV-escaping issue.
- SGE counts include noncoding variants, stops and indels. BARD1's downloaded protein-HGVS column is entirely missing, requiring explicit transcript/CDS mapping rather than classifying all records as unusable or guessing translations. The selected DDX3X aggregate is a classifier; only original experimental fitness channels are eligible.

## Prioritization

Parallel qualification of direct catalytic activity, physical binding and endogenous cellular fitness is scientifically warranted because they test distinct measurements and have plausible multi-protein support. The membrane-context panel is useful for paired-label diagnostics but should not be promoted as broad independent confirmation merely because it has many variants or channels. Reporter and physical endpoints remain separate even when both portals call them binding or activity.

No all-33-model inference campaign is authorized from catalog counts alone. First fix exact sequence/label support, source/condition grouping, overlap/homology exclusions, reliability and phenotype-specific baseline qualification. Then join existing native scalar coverage and score only missing model–state combinations after H200 access is available. Additional measured phenotypes on already-scored states may require no new inference.

## Sources and retained evidence

- [VenusMutHub paper](https://doi.org/10.1016/j.apsb.2025.03.028) and [public dataset](https://huggingface.co/datasets/AI4Protein/VenusMutHub)
- [SKEMPI2 paper](https://doi.org/10.1093/bioinformatics/bty635), [PPB-Affinity paper](https://www.nature.com/articles/s41597-024-03997-4) and [record](https://zenodo.org/records/13054646)
- [BindingGYM repository](https://github.com/luwei0917/BindingGYM) and [data record](https://zenodo.org/records/12514160)
- [MaveDB update and public catalog](https://doi.org/10.1186/s13059-025-03476-y); [BAP1 SGE](https://doi.org/10.1038/s41588-024-01799-3), [RAD51C SGE](https://doi.org/10.1016/j.cell.2024.08.039), [VHL SGE](https://doi.org/10.1038/s41588-024-01800-z)
- [F9 MultiSTEP](https://www.nature.com/articles/s41594-025-01582-w) and [RHO author data](https://github.com/octantbio/rho-dms)
- [ProteinGym author repository](https://github.com/OATML-Markslab/ProteinGym) and [v1.3 record](https://zenodo.org/records/15293562)

The detailed evidence report, machine-readable candidate registry, acquisition manifests, API metadata, source files and inspection results are retained under `results/extensions/phenotype_followups_20261007/discovery/`. Explicit access failures and missing source/annotation fields are retained. No model weights, broad structure archives, fabricated identities for access forms, fitting or inference were used in discovery. See the [program](phenotype-program.md) for common evidence and execution boundaries.
