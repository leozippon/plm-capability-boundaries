# Mutation–structure extensions

These are new analyses, not revisions to the frozen manuscript evidence. Run them in the order structural overlap → contact response → residue-distance prediction. A separate [likelihood-first information progression](information-progression.md) defines the subsequent Result 1 extension that adds biological feature blocks to likelihood. Outputs, fitted predictions and provenance belong in a new extension result directory; never overwrite historical inputs or infer missing native likelihood arrays from rank scores.

## Available structural support

The initial local coordinate join admits 30 of 201 anchor assays (30 of 163 families), containing 3,840 of the original 25,728 variants before strict-single selection. All admitted assays are Tsuboyama stability benchmarks: this is a selected stability subset, not representative coverage of the full mutation panel. The RSA-ready strict-single fitting intersection contains 2,067 variants across those 30 assays/families; all original outer and inner partitions remain feasible after projection.

The annotation adapter is runnable as `python -m src.capability.extensions.structure --cohort COHORT --admission ADMISSION --structures COORDINATE_DIRECTORY --out NEW_OUTPUT_DIRECTORY`, with optional repeatable `--extra-structure PATH`. Use the validated Python environment and 1–4 CPU threads. It writes every WT position, explicit missingness, exact sequence/entity mappings, recomputed isolated-chain RSA, observed mapped-WT contact degree, source hashes and exclusions. Missing neighbors, omitted assemblies/partners and fragment boundaries limit structural interpretation; no secondary structure labels are claimed.

## Structural overlap

The baseline is the manuscript's `C_P_wall`: sequence/substitution descriptors, evolutionary profile and qualified local-window controls. On an exact structure-matched strict-single-substitution cohort, compare `B`, `B+M`, `B+X`, and `B+M+X`. All four designs use identical rows, family partitions, preprocessing and nested fitting procedures. Refit on the restricted cohort; filtering historical held-out predictions is not a substitute.

The initial structural block uses actual wild-type RSA and declared RSA × substitution-property interactions. Full-coordinate contact degree is a sensitivity, not a count from the historical selected epistasis pairs. Secondary structure is not included without validated annotations. Admit exact wild-type sequence mappings to experimental structures; preserve missing coordinates and mapping exclusions explicitly. Structure choice must not use phenotype or model response.

First report whether `rho(B+M)-rho(B)` remains supported on the restricted cohort. Then report the conditional increment `rho(B+M+X)-rho(B+X)` and their paired difference. A reduced increment measures predictive overlap under these readouts, not causal mediation or a fraction of structural knowledge. An unresolved original increment prevents an affirmative attenuation interpretation; do not select models or families by this check.

### Completed RSA overlap result

The 33-model, three-split panel completed all 99 cells on the same 2,067 variants. The RSA block raises mean held-out Spearman from 0.56911 to 0.61007 (increment 0.04096; simultaneous 95% band [0.02830, 0.05363]). Six models retain a positive original-increment pointwise interval in every split: ProGen2 small/medium/xlarge, ProGen3-3B, ProteinGLM and ProtGPT2. This gate allows assessment; it does not select models out of the panel or establish attenuation.

ProGen3-3B and ProteinGLM have positive pointwise attenuation estimates of 0.00298 and 0.00341, respectively, but **all 33 simultaneous Spearman attenuation bands include zero**. Rank-MSE attenuation is likewise unresolved after simultaneous adjustment. Thus RSA is predictively useful on this selected stability subset, but the extension does not establish multiplicity-robust attenuation of model information. Intervals use 2,000 shared family draws over 33 models × two metrics × four contrasts and condition on fitted predictions. Contact degree was constructed but has not been fitted as an additional structural sensitivity.

## Contact response

The primary native-response and distance panels retain the historical exclusion of ProGen3-3B because of unresolved layout dependence; its scalar-score overlap result above does not admit it to positional interpretation. Recovery must bind each arm to its original extraction/layout provenance before analysis.

For unchanged downstream native-residue receivers, signed response is wild-type NLL minus mutant NLL. The primary endpoint is its absolute magnitude; signed disruption (negative signed response) is secondary. Primary residue-level interpretation requires strictly aligned, single-residue receiver tokens. Mutation-spanning, multi-residue, boundary and formatting tokens remain separately accounted for, not silently assigned to residues.

Construct all eligible pairs from structure before examining responses or phenotype labels. A contact uses C-beta distance below 8 Å (C-alpha for glycine); pairs separated by at most two residues are excluded. Noncontacts must have valid coordinates. Match within protein and mutation anchor on declared residue-distance strata: 3–8, 9–16, 17–32, 33–64, 65–128 and ≥129. Report distance imbalance and an adjusted-distance sensitivity, rather than claiming broad bins ensure exact balance. Evaluate receiver identity, wild-type predictability and RSA; retain explicit missingness and common-support losses.

Aggregate matched receivers to mutations, assays and biological families before inference. Pair counts do not establish independent sample size. The historical phenotype-epistasis contact qualification is a distinct endpoint and neither qualifies nor prohibits this response analysis.

## Residue-distance prediction

Candidate downstream bins are 1–8, 9–32, 33–128 and ≥129 residues from the mutation, subject to a label-blind support census. Distances are never tokenizer offsets. Compare a complete `B+own+bins+remainder` design with each drop-one-bin design on identical variants and folds. The endpoint is the held-out mutation-ranking increment conditional on the other components, not average perturbation magnitude.

Retain unallocated multi-residue and formatting contributions as an explicit remainder; native-score closure and receiver coverage must be validated. A bin with no eligible receivers has a zero contribution, not a missing value; a wholly unsupported bin does not warrant an inferential contrast. Any interval-allocation sensitivity must be labelled as an allocation convention, not recovered per-residue likelihood.

## Running the CPU extensions

Use the validated Python environment with bounded CPU threads, for example `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2`. All output paths must be new; historical artifacts are inputs only.

- `fit_structural_overlap.py --sites SITES.json.gz --out NEW_DIRECTORY` reconstructs the admitted baseline and retained model ranks, projects original partitions and runs all four designs. It emits exact rows, OOF predictions, fold audits, paired bootstrap contrasts and the original-increment gate.
- `analyse_residue_responses.py --mode prepare --input COHORT.json --structures SITES.json.gz --coverage COVERAGE.json --out NEW_PAIRS.json` constructs the geometry-only pair set. The paired coverage receipt binds experimental method, exact mapping and source provenance; the declared atom must be CB, or CA for glycine.
- `analyse_residue_responses.py --mode census --input ADMITTED_COHORT.json --out NEW_CENSUS.json` counts potential residue-distance support independently of structure. Supply the explicitly admitted cohort: the larger source cohort has different support.
- After original archives are recovered, `analyse_residue_responses.py --mode packing --arm ARM --input ARCHIVE_INPUT.json --out NEW_PACKING_INPUT.json` reconstructs and checks packing with the original tokenizer-only loader. It refuses a loaded model and performs no forward pass. Input assay rows identify their original archive, WT, ordered mutants and exact mutant sequences. ZymCTRL additionally requires the original EC-bound cohort and conditioning receipt.
- Use the resulting input with `--mode contacts` (also supplying structures/coverage) or `--mode bins`. Bin inputs require the declared baseline matrix `B` and endpoint in original archive order. Full archive identity and native closure are checked before per-variant biological exclusions; retained singles, baseline rows and labels are projected together with original indices. Corruption fails explicitly rather than being treated as an exclusion. Bin fits declare new seeded subset partitions shared across designs, not historical-fold equivalence.
- `--mode requirements --out NEW_REQUIREMENTS.json` emits the recovery contract. It is not evidence that the remote archives exist or have been recovered.

Geometry preparation currently retains 22,207 matched receiver rows across 1,607 mutations and 30 families; the residual family-weighted distance imbalance is −1.559 residues. These are potential structural pairs, not response results, and must be rematched after token admission. The admitted anchor has 20,568 strict singles in 195 assays/160 families; potential nonempty distance-bin counts are 20,430, 19,306, 15,841 and 8,551. Neither census establishes native token support. No contact-response or distance-bin predictive result is available until original arrays are validated. Contact/bin simultaneous intervals currently cover supported contrasts within each supplied model input; they are not a cross-model panel confirmation.

## Inference and provenance

Use the existing family-held-out nested ridge procedure and training-only feature standardization. Preserve historical partition membership when projecting to the subset is supported; otherwise declare new subset partitions and never claim original-fold equivalence. Each matched comparison must verify realized inner and outer membership equality.

Average split sensitivities within biological units before jointly resampling families. Retain paired contrasts and explicitly declare the model/contrast multiplicity family. Bootstrap intervals are conditional on fitted predictions; they do not include full refitting or training-history uncertainty. Report coverage, exclusions, source hashes, exact variant identities, fold membership and output receipts.

Historical scalar likelihood ranks are sufficient for overlap fitting but cannot reconstruct native token responses or additive distance components. Recover original per-token NLL vectors, state offsets, residue counts/offsets, packing metadata and exact sequence identities first. Verify strict alignment, sign, scored boundaries and native closure before analysis. Only genuinely missing inference should be scheduled after H200 returns; no restoration date is assumed verified.
