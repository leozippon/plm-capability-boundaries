# Endogenous cellular-fitness preparation

Source-bound acquisition and identity qualification of saturation-genome-editing score sets, covering twelve candidate genes. The preparation uses no model, no fit and no GPU; it decides what a protein state *is* for an edited endogenous locus and refuses everything it cannot bind to a source.

This document was written after the fact: the preparation code and its tests were produced in an earlier session and left uncommitted and undocumented. The code itself was read, its tests re-run and its receipts checked rather than regenerated; nothing in the preparation was changed.

## What the preparation establishes

Saturation genome editing measures depletion of edited cells at the native locus, so a row is a nucleotide variant and not a protein state. The module therefore refuses four shortcuts that would quietly manufacture a protein cohort. A transcript accession alone is not a coding frame: a gene whose exact transcript lookup is unavailable stays blocked rather than being translated from a guessed frame. A clinical classifier is not a measured label, so the DDX3X aggregate score set is rejected outright. An RNA score is retained as a labelled diagnostic and never enters as an unmarked protein-only predictor. And a protein substitution reachable by several distinct nucleotide edits keeps all of its rows, with the discordance visible, instead of being averaged.

Acquired: 28 score sets over 12 candidate genes; 10 genes with usable labels; 32,711 finite label records; 23,701 distinct labelled single protein states; 26,669 validated variant records; 23,711 native protein states in the inventory. No exact or containment match to any ProteinGym target was found for the ten wild types. BAP1 and DDX3X remain blocked on source-bound transcript or coding-sequence acquisition, which needs network access this machine does not have.

The preparation's own receipt records zero source-quality-ready genes, a null final independent family count and `main_text_eligibility: false`. Those are honest statements about what the preparation did not decide, not defects: family independence and source-quality adjudication were left to the consumer.

## What the breadth programme then decided

The [breadth qualification](phenotype-breadth.md) completed the two open gates with its own instruments, and the cohort reaches nine independent homology groups, one above the eight-group floor, on 2,304 retained rows after the declared screens. Three screens did the work, and each is worth stating because each discards a lot.

A score set exports several correlated timepoint columns of one library. Those are not replicate measurements and not independent rows, so exactly one endpoint column may enter a ranking background: the score set's own `score` column where it publishes one, and otherwise its continuous aggregate over the measured timepoints. RAD51C publishes only timepoint columns and enters through the aggregate; VHL's alternative culture-condition columns are refused. 6,042 rows go this way.

PALB2 is 1,186 residues, which no frozen interface can pack inside the admitted 1,024-token budget, so all 6,496 of its states are refused before any scoring. That is the gene the cohort loses, and it is why nine groups rather than ten.

4,239 rows carry a protein substitution that appears more than once in its score set, because several nucleotide edits reach the same residue change. All copies are dropped. Choosing a summary of discordant replicates is a declaration the source has to make.

The endpoint is licensed for ranking only. Depletion scores are normalised per score set on scales that are not shared between them, so no squared-error metric may be pooled across genes.

## The irreducible limitation

An edited coding single-nucleotide variant changes the transcript as well as the protein. An endogenous growth label is therefore not a protein-only phenotype, and this cohort supports a statement about predicting endogenous cellular fitness and no statement about a protein-level mechanism. Nine groups is also one above the floor: a single gene lost to any later screen would make every interval on this cohort degenerate.

## Running it

```bash
python scripts/capability/extensions/prepare_cellular_fitness_cohort.py --offline \
  --out results/extensions/phenotype_followups_20261007/cellular-fitness/<fresh>
```

The output directory must be fresh and must stay inside the cellular-fitness scope; both are enforced. `--offline` replays the cached public payloads under `data/phenotype_followups_20261007/cellular-fitness/` instead of fetching, which is the only mode available on a machine without a route to the public APIs. The accepted outputs are `results/extensions/phenotype_followups_20261007/cellular-fitness/final/`; earlier executions are preserved beside it. Tests are `tests/extensions/test_cellular_fitness_cohort.py`.
