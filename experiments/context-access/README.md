# R4 非参数化记忆的使用 — Context access

Does access to external homologous sequence change prediction or generation? This module includes mutation scoring with homologous, unrelated or absent context and historical E11's homologous-sequence-prefix generation experiment. It treats access to homology as a property of the supplied record rather than of the corpus in general.

## The question

The same target is scored under crossed controls: with homologous context, with composition-matched unrelated context, and without context. What a context arm adds is the difference between those conditions, so an improvement establishes a dependence on the supplied context and not an ability to generalise to remote homologs. Whether remote generalisation holds is a separate question, and it required a close-stratum positive control that failed; the retained record therefore leaves it non-identifiable rather than unresolved or absent.

Historical E11 asks whether an external homologous sequence prefix improves generated products. Its retained comparison establishes no clear structural-confidence benefit, with interface and power limits; this does not establish that all homolog-conditioned generation fails.

## Code

- Homolog retrieval and context construction: [`src/capability/context/homology.py`](../../src/capability/context/homology.py), [`homology_context.py`](../../src/capability/context/homology_context.py), [`scripts/capability/context/retrieve_homology_context.py`](../../scripts/capability/context/retrieve_homology_context.py)
- Crossed local, global and composition controls: [`src/capability/context/crossed_controls.py`](../../src/capability/context/crossed_controls.py), [`local_context.py`](../../src/capability/context/local_context.py), [`global_context.py`](../../src/capability/context/global_context.py), run through [`scripts/capability/context/analyse_crossed_controls.py`](../../scripts/capability/context/analyse_crossed_controls.py)
- Remote homology and retrieval bounds: [`src/capability/context/remote_homology.py`](../../src/capability/context/remote_homology.py), [`retrieval_strata.py`](../../src/capability/context/retrieval_strata.py), with [`scripts/capability/context/qualify_remote_homology_endpoint.py`](../../scripts/capability/context/qualify_remote_homology_endpoint.py) and [`refit_remote_strata.py`](../../scripts/capability/context/refit_remote_strata.py)
- Historical E11 generation and analysis: [`scripts/capability/context/generate_homolog_conditioned.py`](../../scripts/capability/context/generate_homolog_conditioned.py) (`run` and `analyse`), with structural readout in [`fold_conditioned_products.py`](../../scripts/capability/context/fold_conditioned_products.py)
- Profile increment inside the context comparison: [`src/capability/context/profile_increment.py`](../../src/capability/context/profile_increment.py), [`scripts/capability/context/analyse_profile_increment.py`](../../scripts/capability/context/analyse_profile_increment.py)

## Inputs

Local and not distributed: `data/homology_db`, `data/uniref50`, `data/swissprot`, and the staged corpus and taxonomy directories the retrieval stack reads. Hit tables and search depth are named by each endpoint's own declaration rather than set centrally.

## Evidence

Local result trees: `results/R5` holds the homologous-context campaigns, the crossed controls, the local and global context gates, the profile increment, the retrieval bound, the remoteness measurements and the remote support audit; `results/extensions/homology_context_20261008` holds the later homology-context run against `uniref90_2026_03`.

Historical E11 is stored under `results/R6/e11_completed_20261011` but belongs to current R4: its main comparison is `results/R6/e11_completed_20261011/comparison/conditioned_generation_comparison.json`, selection is `results/R6/e11_completed_20261011/selection/product_selection_novelty.json`, and the source bundle's provenance record is named `PROVENANCE.json` (its local path is not asserted here). These are local retained products, not distributed downloads.

## Shared boundaries

Retrieval bounds describe what the retained record can reach, not what a corpus contains, and the cohorts were searched deeper than the earlier mutation cohort, so a depth comparison across them is not available without a new search. Generation that varies access to an external homologous record belongs to R4. Ordinary family-label or target-own-prefix conditioning belongs to [R5](../generative-expression/README.md); sampling and evaluation tools can be shared. Sharing a module or a wild type does not imply a shared sample, resampling unit or target.
