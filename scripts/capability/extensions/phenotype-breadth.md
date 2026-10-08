# Phenotype breadth and matched phenotypes

Two questions are answered here, and they are different questions that happen to share one scoring pass.

**Which additional biological phenotypes can benefit from mutation information in a model's likelihood?** This needs *independent* cohorts: new proteins, new assays, new phenotypes, admitted only after the same controls the anchor cohort and the nested gates demand.

**Do the effects of one mutation on different phenotypes follow shared patterns?** This needs *matched* measurements: the same protein and the same substitution measured for two phenotypes. Matching removes the sequence and support confound that makes any comparison of two separate cohorts uninterpretable, but it creates no new independent proteins, so it answers the second question and contributes nothing to the first.

## How the stages run

Four stages, in this order. Each takes `--device` and `--out`, takes every other input as an explicit flag, and writes its completion record last.

1. `qualify_phenotype_cohorts.py` (CPU) reads the prepared identity artefacts of every candidate cohort and decides what may be measured. It writes one admission record per candidate, the deduplicated state plan of the cohorts it admitted, and the query catalogue the evolutionary-profile control needs. It reads no model output and no checkpoint.
2. `score_phenotype_states.py` (GPU) scores the planned states with one or more checkpoints. The measurement is the admitted one, imported unchanged from the readout extraction path, so a likelihood produced here is the same quantity the stability likelihood-only extraction produces: each state packed in its checkpoint's native rendering, one forward per state, the native next-token negative log likelihood reduced over the interface's scored span. The stage is label-blind and refuses a plan carrying a measurement field.
3. `fit_phenotype_breadth.py` (CPU) qualifies the control blocks on each admitted cohort's own endpoint and then measures what the likelihood adds over the qualified set, separately for ranking and for quantitative prediction.
4. `analyse_matched_phenotypes.py` (CPU) analyses the matched support.

The fit stage re-derives each cohort's support from the same readers and refuses to run unless the re-derived row digest equals the one the admission recorded. A fit therefore cannot quietly run on a support the admission never saw.

All four share one output-directory contract, `prepare_output`. A missing directory is created and an existing *empty* one is accepted, because the campaign queue does its own `mkdir -p` and then injects the directory as `--out`: under the runner the output directory always exists and is always empty when a cell starts, so a guard on existence would refuse every normal cell and nothing else. What is refused is content, which is the actual evidence of prior work — a present completion record, reported as a previous run having finished there, or any other content, reported as a previous run having written there without completing. Both exit non-zero naming the path. There is no override flag: resuming is the queue's own skip-complete behaviour, which reads exactly the completion record this guard refuses to write over.

## The admission discipline

**The independent unit is a sequence-homology group**, at the frozen rule of 30% identity over 80% of both sequences with at least thirty paired residues, plus exact and containment identity. Assay files, genes, PDB complexes, score sets, conditions and exact-sequence components are not families and are never counted as such. The grouping instrument is the committed exact aligner, so every cohort's independence is measured the same way.

**The floor is eight groups**, taken from the project's percentile-interval floor rather than redeclared, and it is checked before any GPU time is spent. It is the floor that closed the DHFR candidate at one group and the TEV candidate at seven. A cohort below it is written out as `refused` with its group count and the floor's own explanation; it never reaches the state plan. A cohort that lands exactly on the floor is admitted with a recorded note that it has no margin.

**Endpoint direction is declared, never inferred from a title.** For the activity partitions the direction is a property of the named physical quantity: a larger turnover number is faster catalysis, a larger catalytic efficiency is more efficient. A Michaelis constant is an affinity and does not determine an activity direction; a substrate or enantiomer preference ratio has no phenotype direction without the objective the authors optimised. Rows under those quantities are refused and counted. A partition naming no declared quantity is refused too.

**A construct that is not verified against the measured one blocks the cohort unless the tolerance is written down.** The binding cohort carries such a tolerance, and it is a specific argument rather than a shrug: the endpoint is a within-construct difference, so a tag or a truncation elsewhere in the entity shifts the wild-type and the mutant likelihood together and cancels. What the mismatch costs is stated as a boundary — the cohort supports a claim about the likelihood difference of a substitution and no claim about the absolute likelihood of the measured construct.

**Three screens run label-blind before grouping.** A state the interface budget cannot pack is refused rather than truncated. A ranking background holding one substitution more than once has an undeclared replicate, endpoint or condition structure, so every row of that key is dropped and counted; nothing is averaged, because choosing a summary of discordant replicates is a declaration the source has to make. A background with a constant label cannot be ranked at all.

**Retention is capped at 256 variants per independent group**, which is the external-confirmation panel's own cap, drawn by a stable hash of the row identity. A group-unit interval is not limited by how many rows sit inside a group, and an uncapped cohort would spend its whole scoring budget on its three largest proteins.

## Qualified controls

The candidate control blocks are the nested gate's own: composition, mutation-local chemistry windows, and the mutation-local evolutionary profile in its raw and bounded restatements, offered in that order over a base of directed substitution identity and mutated-position geometry. A candidate is kept only if its paired reduction in group-equal held-out error over the standing set is positive at every split seed — the frozen rule, imported rather than restated. Each candidate's own contribution is reported whether or not it was kept, and so is the qualified set's contribution over the no-effect null, because an increment measured over a control that predicts nothing is uninformative.

The evolutionary-profile blocks are an input, not an option. They come from the repository's own pipeline against the retained UniRef50 index, over the catalogue and plan the qualification stage writes:

```bash
python scripts/capability/interactions/search_pairwise_homologs.py \
  --diamond external/tools/diamond/diamond \
  --database data/homology_db/uniref50_full.dmnd \
  --catalogue <qualification>/profile-catalogue.json \
  --out-dir <profiles>/search --threads 16
python scripts/capability/interactions/build_pairwise_profile_features.py \
  --plan <qualification>/profile-plan.json \
  --hits <profiles>/search/hits.tsv \
  --out <profiles>/profile_features.npz
```

Run without `--profiles`, the fit records its result as `local_controls_only` and says in terms that an evolutionary-statistics account of the increment has not been excluded. That is a weaker result honestly labelled, not an admission.

## Two endpoints, reported apart

A likelihood difference has an arbitrary scale, so the two things a model might do for a phenotype are measured as two endpoints and never combined.

*Ranking* fits the within-background standardized-rank target every frozen panel uses, with the likelihood entering as its within-background standardized rank, and reports the Spearman increment. Rank-target squared error is never called phenotype error.

*Quantitative prediction* is licensed only where the cohort's label unit is traced and shared across its backgrounds. Its predictor is fitted on the endpoint's own units using training groups only — that is the calibration, and no post-hoc rescaling is applied — and it reports the group-equal held-out squared-error reduction in those units. Where units are not shared the endpoint is refused with that reason recorded, because pooling heterogeneous units into one error metric would invent a quantity. Of the admitted cohorts only binding clears this, on the dimensionless log affinity ratio.

Inference is a simultaneous band per cohort and endpoint: the 95th percentile of the maximum absolute centered deviation across the arms under one shared resample of the independent groups, with the pointwise quantiles reported beside it. Aggregation is background within group, then equal groups; split seeds are averaged as sensitivity analyses, never as replicates. Every band carries the floor record of the group count it was computed on.

## The matched support

Thirteen anchor proteins carry two or three assays of different adjudicated phenotype classes on the identical construct — abundance beside activity, surface expression beside ion conduction, abundance beside binding. Matching their tables on exact wild-type sequence and exact substitution gives fifteen cross-class assay pairs over thirteen independent groups. Same-class pairs are excluded: they ask a replication question. Two constructs of one gene are not matched.

The likelihood difference of a substitution is one number whatever phenotype is measured on it, so a checkpoint cannot carry more information about one phenotype through a different score — only through a different alignment with each label. Three quantities on identical rows separate the cases: the rank correlation of the two labels, which is the biology and uses no model quantity; the ranking increment for each phenotype and the signed difference between them; and the rank correlation of the two phenotypes' residuals with the likelihood in the design and without it. A fall in that shared residual means the likelihood explained part of what both phenotypes miss in common, which is shared information. An unchanged shared residual beside two positive increments means it added phenotype-specific information instead.

Two design facts travel with every matched number. The per-pair fit is within one protein with mutated-position-held-out folds, because a protein-held-out partition inside a single protein would be empty; the increment therefore generalises across sites of a known protein and is not the cross-protein increment the anchor panel reports. And no phenotype pair reaches the independent-group floor on its own — six, three, three and one group — so the pooled thirteen-protein band is the only inference, and every per-pair number carries its own degenerate-floor record.

## What the qualification found

| cohort | phenotype | groups | rows | ranking | quantitative |
|---|---|---|---|---|---|
| `activity_venus` | direct catalytic activity | 89 | 1,663 | admitted | refused: units not shared |
| `binding_skempi` | protein–protein binding affinity | 140 | 3,410 | admitted | admitted on ln(Kd ratio) |
| `cellular_fitness_sge` | endogenous cellular fitness | 9 | 2,304 | admitted | refused: units not shared |
| `matched_proteingym` | matched multi-phenotype | 13 | 7,680 | admitted | refused: two phenotypes, two units |
| `membrane_multistep` | trafficking, display, secretion | 2 | — | refused: below the floor | refused |

Four further candidates are closed on recorded evidence without being read, and the reasons are carried in the stage's `declared-refusals.json`: the DHFR scanning release publishes no per-variant fitness table and collapses to one group; the TEV set gives seven groups against the floor of eight; the MGnify stability release is the existing stability endpoint rather than an additional phenotype, shares assay technology and an author with the development endpoint, and its two protease channels report systematically different quantities; the HIS3 set is one protein family and duplicates the anchor's largest endpoint class.

The membrane refusal is worth reading rather than skipping. Its endpoints are well defined and its direction is declared by the source, and it still fails: the F9 channels differ only in antibody, RHO *is* the anchor protein `OPSD_HUMAN_Wan_2019` under a new method, and SGCA supplies no source full-length protein wild type at all. Five score sets and tens of thousands of measurements are two independent proteins.

## Irreducible limitations

A 30%/80% homology group is a conservative independence unit, not an independent evolutionary or experimental history. Within-background transforms of the model feature and of the ranking target are label-blind but read every row of a background including held-out rows; that is the established convention of the frozen panels. Intervals condition on fitted out-of-fold predictions and carry no training, tuning or checkpoint-selection uncertainty. A qualified control set is a competent predictor of held-out groups, not a complete account of non-model explanations. The binding cohort scores the mutated chain alone, so it measures whether that chain's own likelihood carries binding information and not whether the model represents the interface. The endogenous-fitness cohort's label integrates transcript and protein effects for a coding edit, so no protein-only mechanism is claimed for it, and its nine groups sit one above the floor.
