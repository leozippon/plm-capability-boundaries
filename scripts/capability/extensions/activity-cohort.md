# Public catalytic-activity cohort preparation

Acquisition and identity preparation are complete for all 130 VenusMutHub single-mutant activity files at pinned revision `08a27d07b5764af58f99d26734d096ebe2a5f9cf`. No model inference or fitting was performed. Assay files are not independent enzymes or families, and identity readiness is not measurement or modelling qualification.

## Verified support

| Quantity | Count |
| --- | ---: |
| Downloaded assay files / failures | 130 / 0 |
| Source rows, including reference/identity substitutions | 1,853 |
| Admissible single-missense identity rows | 1,722 |
| Finite single-missense score rows | 1,720 |
| Distinct exact WT sequences | 97 |
| Distinct WT–mutant sequence pairs | 1,439 |
| Canonical WT/mutant sequence states | 1,536 |
| Exact WT groups repeated across files | 24 |
| Variant states repeated across assays | 206 |

All 130 files reconstruct an internally consistent WT from their declared full mutant sequences and exactly match the author's dataset-summary WT. Identity substitutions, missing labels, ties, repeated variants and assay partitions are preserved explicitly. Seventy files have tied single-variant scores; 117 have the descriptive screen of at least five distinct finite single states and nonconstant scores. That screen is not a model-readiness or main-text gate.

No WT has an exact or containment match to the old 217-assay ProteinGym reference or retained 201-assay anchor. Homology screening is a separate stage; absence of these exact matches does not establish family independence. Background variants and repeated assay conditions must not be counted as new proteins merely because their file names differ.

## Source interpretation

The root-level DOI map advertised by the repository was missing, but targeted inspection recovered `mutant/doi.csv` and `mutant/dataset_summary.csv`. Exact dataset-ID joins give 123 unambiguous author DOI mappings; seven retain explicit question-mark uncertainty. An author DOI mapping is a source lead, not verification that every label has the stated unit or experimental construct.

For `5A71_kcat` and `5A71_kcatkm`, every released row was matched to the first substrate block of Table 1 in [the original study](https://doi.org/10.1038/s41467-019-11155-3). Their units are respectively s⁻¹ and s⁻¹ M⁻¹; source uncertainty strings are retained. This is a unit/row trace, not complete qualification of construct identity, replicate independence or all assay conditions. The other 128 files remain uninterpreted source-score assays. No file is promoted to a fully qualified modelling panel or physical-MSE comparison in this stage.

Do not pool kcat, catalytic efficiency, Km, vmax and relative activity. Km is not turnover, and numerical direction must come from the source rather than treating every `fitness_score` as the same biological quantity. A source-checked endpoint can support its own within-background ranking and appropriately calibrated quantitative metric; unresolved source scores cannot support a generic physical activity-MSE claim.

## Execution and provenance

Use `scripts/capability/extensions/prepare_activity_cohort.py --out NEW_DIRECTORY`, optionally `--offline` for immutable cached data. The runner refuses existing outputs, caps acquisition size and workers, and checks the complete unique 130-file release inventory in both online and offline modes. Raw source bytes are never overwritten; source and code hashes, row/state provenance, unresolved gates and eight scientific-output hashes are retained.

Final preparation is `results/extensions/phenotype_followups_20261007/activity/final-20261007/`, with `activity/current-result.json` pointing to its receipt. Sources are under `data/phenotype_followups_20261007/activity/`. Acquisition transferred 767,484 score-file bytes. Fourteen tests passed; the parent offline execution reproduced all eight scientific outputs byte-for-byte, verified their hashes and current code bindings, and preserved the earlier execution separately.

The state inventory is for qualification and future exact model-state coverage joins, not authorization to score every state. Source/condition/reliability adjudication, homolog grouping, independent support and phenotype-specific baseline qualification remain required. See the [multi-task program](phenotype-program.md) and [public discovery](public-phenotype-candidates.md) for the broader context.
