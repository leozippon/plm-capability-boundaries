# Quantitative binding cohort preparation

SKEMPI2 affinity-label preparation, operational sequence mapping and source-background adjudication are complete. Exhaustive CPU homology/development-overlap screening has completed and passed parent verification on the adjudicated support. No model inference, predictive fitting or main-text promotion has occurred.

## Labels and operational states

The frozen local source contains 7,085 records. First-stage admission retains 4,829 uncensored paired-affinity single-substitution records. Original affinity strings are checked before parsed values: inequalities, nonbinding/unfolded markers and approximate values are not converted into point measurements. Duplicate records and condition groups are preserved without averaging. The primary provisional target is ln(Kd_mut/Kd_WT), positive for weaker binding; physical ΔΔG uses the actual source-reported, non-assumed temperature. There are 3,336 such temperature-qualified records before sequence/background exclusions.

A bounded public acquisition obtained 315 coordinate payloads, totalling 62.9 MB compressed. Mapping uses full canonical protein-chain sequences, author numbering/insertion codes, declared mutation-site WT identity and complete partner inventories. It does not guess missing residues or equate PDB chain identity with proven full experimental-construct identity.

Source-note review found documented sequence/background disagreements that matching the focal residue alone would miss. All 110 distinct admitted Notes values were reviewed against a hash-bound control table. Known incompatibilities include engineered crystal backgrounds substituted for measured WT, murine versus human references, and omitted additional mutant substitutions. Engineering alone is not a blanket exclusion: unresolved constructs remain explicitly unknown rather than being invented or declared verified.

| Adjudicated quantity | Count |
| --- | ---: |
| Qualified operational label–reference associations | 4,108 |
| Blocked label–reference associations | 721 |
| Of those, documented construct/annotation incompatibilities | 431 |
| Operational complexes | 295 |
| Exact-sequence-sharing components | 153 |
| Complete reference-chain sequences | 546 |
| Distinct single variants | 2,941 |
| WT/mutant/reference state sequences | 3,323 |
| Provisional backgrounds meeting the ranking-support screen | 246 |

The 431 incompatible associations have no exported scoring-chain association, variant key or mutant-state hash. Old operational mappings are preserved as historical preparation products, not reused as qualified input. The remaining 4,108 associations are still `UNKNOWN_CONSTRUCT`, not universal proof of measured full-sequence equivalence. Exact-sharing components are not final independent families; model-score coverage is unverified.

## Homology and overlap protocol

The old exact batched alignment code has a hard sequence-length limit below 128 residues, so it cannot silently be applied to the new longer chains. A new CPU backend preserves its BLOSUM62 scoring, affine gap cost 11 + 1k, local optimum and tie rules with dynamic coordinates. It leaves the frozen implementation unchanged. Tests compare short pairs with both old implementations and longer pairs with the independent traceback reference; binaries remain in ignored runtime directories.

The label-blind screen uses 30% identity and 80% coverage of both sequences for family edges, with shared-partner connected-component closure. The development screen uses 50% identity and 80% query coverage, with exact/containment sharing additionally recorded. New nonexact links require at least 30 paired residues; unresolved short chains are explicitly quarantined rather than declared unrelated. The complete declared reference is 217 ProteinGym WT targets plus 478 Tsuboyama backgrounds. Lower-identity family-overlap flags are retained, so absence of a 50% hit is not proof of no homologous relationship.

Conservative candidate exclusion propagates an admitted development overlap through its connected component. Direct versus propagated exclusions, mutated-chain versus any-partner component counts, preexclusion membership and study-closure sensitivities are retained. Shared partners can produce large connected networks: this affects available support and does not demonstrate absence of binding information. Source quality, construct equivalence, reliability and phenotype-specific baseline qualification remain separate gates.

## Completed screen

The exhaustive screen evaluated 148,785 chain-family pairs and 379,470 development-reference pairs in 89.48 seconds with four CPU workers. Homology plus any-partner sharing reduces the 153 exact-sharing components to 73 operational complex components. Direct anchor links occur in 66 complexes; component propagation excludes 177 complexes in total. After the declared anchor and unresolved-short-chain exclusions, **110 complexes in 57 candidate components retain 1,271 labels and 1,034 distinct variants**.

Grouping choice matters: mutated-chain-only closure gives 117 complex components, whereas the conservative any-partner rule gives 73. Parsed-study closure reduces the preexclusion 73 to 72; unparsed references and dependence beyond shared study identifiers remain unresolved. These counts describe operational support under the declared screen, not 57 proven independent biological families or a qualified main-text result.

Partner/reference states are retained for mapping, grouping and controls; they are not automatically requests for scalar model scoring. A later likelihood manifest must identify the exact required mutated-chain WT/mutant pairs and their qualified native interfaces. Known incompatible associations remain excluded, and no missing sequence or score was imputed.

## Execution and provenance

- `prepare_binding_cohort.py --out NEW_DIRECTORY` performs source/label admission and refuses overwrites.
- `map_binding_cohort.py` performs bounded or cache-only mapping; the accepted source-adjudicated artifacts are under `results/extensions/phenotype_followups_20261007/binding/mapping/adjudicated/`.
- `qualify_binding_homology.py --mode pilot --out NEW_DIRECTORY` freezes adjudicated inputs/rules and measures CPU cost. `--mode full` requires the unchanged pilot contract and code. Legacy unadjudicated mappings are refused.

The parent reran all 55 binding/mapping/homology tests. The completed screen is under `binding/homology/screen-20261007/full/`. Parent verification checked the frozen contract, current code and evidence hashes, exhaustive pair census, and exact reconstruction of the saved component map from source inputs and saved edge evidence, without rerunning alignment. A separate verification receipt binds the derived output files. The small `binding-construct-adjudication.json` is a scientific admission-control table with source/note hashes and evidence pointers, not a raw dataset dump.

Public cache files remain under `data/phenotype_followups_20261007/binding-structures/`; raw labels and generated artifacts are not committed. This preparation serves the [multi-task phenotype program](phenotype-program.md), and does not authorize a model-scoring campaign before the remaining gates are resolved.
