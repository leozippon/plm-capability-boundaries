# Quantitative binding cohort preparation

SKEMPI2 affinity-label preparation, operational sequence mapping and source-background adjudication are complete. Exhaustive CPU homology/development-overlap screening is running on the adjudicated support. No model inference, predictive fitting or main-text promotion has occurred.

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

## Execution and provenance

- `prepare_binding_cohort.py --out NEW_DIRECTORY` performs source/label admission and refuses overwrites.
- `map_binding_cohort.py` performs bounded or cache-only mapping; the accepted source-adjudicated artifacts are under `results/extensions/phenotype_followups_20261007/binding/mapping/adjudicated/`.
- `qualify_binding_homology.py --mode pilot --out NEW_DIRECTORY` freezes adjudicated inputs/rules and measures CPU cost. `--mode full` requires the unchanged pilot contract and code. Legacy unadjudicated mappings are refused.

The parent reran all 55 binding/mapping/homology tests. The adjudicated pilot verified 64 actual pairs; exhaustive screening was then launched with four CPU workers under `binding/homology/screen-20261007/`. No completed homology result is claimed until its terminal receipt is verified. The small `binding-construct-adjudication.json` is a scientific admission-control table with source/note hashes and evidence pointers, not a raw dataset dump.

Public cache files remain under `data/phenotype_followups_20261007/binding-structures/`; raw labels and generated artifacts are not committed. This preparation serves the [multi-task phenotype program](phenotype-program.md), and does not authorize a model-scoring campaign before the remaining gates are resolved.
