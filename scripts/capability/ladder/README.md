# The modification-extent ladder

Across this programme, models that predict mutation effects well do not reliably generate better proteins. One candidate explanation is a local-to-global generalization failure: a likelihood may carry real information about one residue swapped into a fixed context and little or none about a sequence the model had to construct itself. If that is what is happening, then along a ladder of increasing modification extent, built on the same natural backbones throughout, model likelihood and an independent structural evaluation should start out agreeing and, at some extent, stop. This area builds that ladder and looks for the point where they part company.

The declarations live in [`src/capability/ladder/design.py`](../../../src/capability/ladder/design.py) and are fixed before anything is measured: the rungs, the arms and their model classes, the backbone band and admission thresholds, the draw count, the seeds, the sampling settings, and the ceiling every artefact carries.

## The ladder

One axis: the number of contiguous residues the model is asked to write, at 1, 2, 5, 10, 20 and 40. One residue is a single substitution and two a double substitution, proposed by the model rather than drawn by us, so the bottom of the ladder is the same construction as the top. A seventh rung generates the whole sequence from the arm's bare prompt.

Three design points decide what may be claimed from it.

The window rungs preserve length exactly and the full-generation rung does not. Both likelihood and predicted confidence are strongly length-dependent in this project's own measurements, and in opposite directions, so a comparison across rungs of different length is a comparison of lengths. Each window rung substitutes exactly *k* residues into a fixed backbone, so the variant has the parent's length to the residue and the parent's own fold is a legitimate referent. The full-generation rung is an anchor read as a marginal distribution; its parent-fold columns are absent rather than filled with a number that would not mean the same thing. Length is reported beside every confidence number regardless.

Local regeneration is not one condition. A left-to-right decoder cannot condition on the residues after the window, so for a causal arm the rung is prefix regeneration: the model writes *k* residues given the prefix alone and the original suffix is restored afterwards. For a bidirectional or absorbing-state arm it is genuine fixed-length infilling with both flanks visible throughout. The two are reported separately and never averaged, because the asymmetry between them is part of the question. An arm that cannot express a rung leaves the cell empty.

The window is centred and nested across extents, so extent is the only thing that varies. The price is that position dependence is not estimated at all and no claim about it follows from this design.

## The arms

| arm | model class | condition |
| --- | --- | --- |
| ProtGPT2 | causal protein decoder | prefix regeneration, anchor |
| ZymCTRL | causal protein decoder, EC-conditioned | prefix regeneration, anchor |
| ProLLaMA (stage 2) | causal joint text and protein | prefix regeneration, anchor |
| ProLLaMA stage 1 | causal joint text and protein | prefix regeneration, anchor |
| ESM2-650M | bidirectional masked LM | true infilling |
| DPLM-650M | absorbing-state discrete diffusion | true infilling |
| composition shuffle | no model | extent reference |

The first three are the arms the rest of the programme already uses, so the results compose. The last needs no checkpoint: its window is drawn from the parent's own amino-acid frequencies, which fixes how much of a structural change at extent *k* comes from extent alone. Because every causal arm also scores it, it answers a second question — whether an arm's likelihood tracks structure on sequences it did not produce.

A bidirectional arm has no sequence likelihood. Its scalar is a sum of masked marginals and is labelled a pseudo-log-likelihood everywhere it appears; it is comparable within an arm, which is all the within-backbone statistic needs, and never across arms.

## The backbone set

Sixteen whole natural Swiss-Prot entries, four in each quarter of the 150–300 residue band, each already folded at full length by this project's ESMFold2 instrument at mean CA pLDDT at least 90, at least 95% of residues above 70, and pTM at least 0.80. They are selected from the generation-evaluation experiment's natural comparator arm, which folded 1,822 whole records — not length-matched fragments, which is what makes them usable here — and their parent folds are reused by literal path rather than recomputed.

Two relations are enforced between admitted members: no two may be near-duplicates under this project's own shingle relation, and no two may carry the same EC number. Every backbone carries exactly one EC number, because the EC-conditioned arm has to be expressible on all of them, so the set is EC-annotated enzymes and nothing measured on it generalises to non-enzymes.

## The two measured quantities

The model likelihood of a sequence and the ESMFold2 evaluation of that sequence are produced by different checkpoints in different stages from the same frozen string. The structure predictor never sees a likelihood and the scorer never sees a structure; that independence is what makes their rank correlation interpretable.

The structural side keeps the pairwise confidence fields and adds three parent-referenced comparisons at the residue correspondence the construction knows, since the window rungs preserve length: a TM-score-like global similarity, a superposition-free local distance difference test over the modified window, and the window's RMSD after superposing on the unmodified flanks alone. Structural confidence is not stability and not function; no stability predictor is applied anywhere here and nothing is experimentally verified.

## The statistic

Within each backbone, likelihood and structure are ranked and centred before being correlated, so every between-backbone difference — length first of all — is removed by construction. The backbone is the cluster-bootstrap unit, because two draws on one backbone share a parent, a prompt and a window. Sixteen backbones and eight draws give 128 observations and 16 resampling units per cell.

Each cell carries a marginal percentile interval; the panel also carries a sup-t simultaneous band built from the same resamples across every cell, because the ladder makes a panel-wide claim. A marginal interval is never read as a simultaneous statement.

Three readings sit side by side and none replaces another: the raw within-backbone correlation, the same correlation after linearly removing the within-backbone rank of composition distance from the parent, and the model-free extent reference. A correlation that survives only in the raw reading is a composition result, not a structural one.

The divergence is located by two declared criteria — the first rung whose interval admits zero and stays unresolved above it, and the first rung whose point estimate changes sign — and then more finely by treating extent as continuous and solving for the extent at which the fitted correlation reaches zero, inside the bootstrap. A ladder whose correlation never reaches zero within the measured range reports no crossing, and a ladder that does not resolve at all reports the number of backbones that would be needed.

## Running it

The stages, in dispatch order, all reachable through the campaign queue as bare basenames. `--device` and `--out` are injected by the runner.

| stage | resource | reads | writes |
| --- | --- | --- | --- |
| `build_ladder_backbones.py` | cpu | a `run_structure_evidence.py` index of whole natural records, the EC-labelled FASTA | `ladder_backbones.json`, `ladder_backbones.jsonl` |
| `generate_ladder_windows.py` | gpu | the backbones | `ladder_variants.json`, `ladder_variants.jsonl` |
| `infill_ladder_windows.py` | gpu | the backbones | the same pair |
| `shuffle_ladder_windows.py` | cpu | the backbones | the same pair |
| `build_ladder_fold_cohort.py` | cpu | every arm's variants | `ladder_fold_cohort.jsonl` |
| `run_structure_evidence.py` | gpu | the fold cohort | the resumable ESMFold2 tree |
| `score_ladder_likelihood.py` | gpu | the backbones and variants | `ladder_likelihood.json`, `ladder_likelihood.jsonl` |
| `compare_ladder_folds.py` | cpu | the variants, both fold trees | `ladder_structure.json`, `ladder_structure.jsonl` |
| `analyse_ladder_divergence.py` | cpu | everything above | `ladder_divergence.json`, `ladder_divergence.md` |

The folding stage is the project's existing ESMFold2 runner, used unchanged. It needs the `runtimes/esmfold2` interpreter and must not be run with `-I`.

The dispatch manifest is [`h200/campaigns/campaign_ladder_divergence_20261010.tsv`](../../../h200/campaigns/campaign_ladder_divergence_20261010.tsv).
