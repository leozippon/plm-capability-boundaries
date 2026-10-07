# Mutation–structure extensions

These are new analyses, not revisions to the frozen manuscript evidence. Run them in the order structural overlap → contact response → residue-distance prediction. Outputs, fitted predictions and provenance belong in a new extension result directory; never overwrite historical inputs or infer missing native likelihood arrays from rank scores.

## Structural overlap

The baseline is the manuscript's `C_P_wall`: sequence/substitution descriptors, evolutionary profile and qualified local-window controls. On an exact structure-matched strict-single-substitution cohort, compare `B`, `B+M`, `B+X`, and `B+M+X`. All four designs use identical rows, family partitions, preprocessing and nested fitting procedures. Refit on the restricted cohort; filtering historical held-out predictions is not a substitute.

The initial structural block uses actual wild-type RSA and declared RSA × substitution-property interactions. Full-coordinate contact degree is a sensitivity, not a count from the historical selected epistasis pairs. Secondary structure is not included without validated annotations. Admit exact wild-type sequence mappings to experimental structures; preserve missing coordinates and mapping exclusions explicitly. Structure choice must not use phenotype or model response.

First report whether `rho(B+M)-rho(B)` remains supported on the restricted cohort. Then report the conditional increment `rho(B+M+X)-rho(B+X)` and their paired difference. A reduced increment measures predictive overlap under these readouts, not causal mediation or a fraction of structural knowledge. An unresolved original increment prevents an affirmative attenuation interpretation; do not select models or families by this check.

## Contact response

For unchanged downstream native-residue receivers, signed response is wild-type NLL minus mutant NLL. The primary endpoint is its absolute magnitude; signed disruption (negative signed response) is secondary. Primary residue-level interpretation requires strictly aligned, single-residue receiver tokens. Mutation-spanning, multi-residue, boundary and formatting tokens remain separately accounted for, not silently assigned to residues.

Construct all eligible pairs from structure before examining responses or phenotype labels. A contact uses C-beta distance below 8 Å (C-alpha for glycine); pairs separated by at most two residues are excluded. Noncontacts must have valid coordinates. Match within protein and mutation anchor on declared residue-distance strata: 3–8, 9–16, 17–32, 33–64, 65–128 and ≥129. Report distance imbalance and an adjusted-distance sensitivity, rather than claiming broad bins ensure exact balance. Evaluate receiver identity, wild-type predictability and RSA; retain explicit missingness and common-support losses.

Aggregate matched receivers to mutations, assays and biological families before inference. Pair counts do not establish independent sample size. The historical phenotype-epistasis contact qualification is a distinct endpoint and neither qualifies nor prohibits this response analysis.

## Residue-distance prediction

Candidate downstream bins are 1–8, 9–32, 33–128 and ≥129 residues from the mutation, subject to a label-blind support census. Distances are never tokenizer offsets. Compare a complete `B+own+bins+remainder` design with each drop-one-bin design on identical variants and folds. The endpoint is the held-out mutation-ranking increment conditional on the other components, not average perturbation magnitude.

Retain unallocated multi-residue and formatting contributions as an explicit remainder; native-score closure and receiver coverage must be validated. A bin with no eligible receivers has a zero contribution, not a missing value; a wholly unsupported bin does not warrant an inferential contrast. Any interval-allocation sensitivity must be labelled as an allocation convention, not recovered per-residue likelihood.

## Inference and provenance

Use the existing family-held-out nested ridge procedure and training-only feature standardization. Preserve historical partition membership when projecting to the subset is supported; otherwise declare new subset partitions and never claim original-fold equivalence. Each matched comparison must verify realized inner and outer membership equality.

Average split sensitivities within biological units before jointly resampling families. Retain paired contrasts and explicitly declare the model/contrast multiplicity family. Bootstrap intervals are conditional on fitted predictions; they do not include full refitting or training-history uncertainty. Report coverage, exclusions, source hashes, exact variant identities, fold membership and output receipts.

Historical scalar likelihood ranks are sufficient for overlap fitting but cannot reconstruct native token responses or additive distance components. Recover original per-token NLL vectors, state offsets, residue counts/offsets, packing metadata and exact sequence identities first. Verify strict alignment, sign, scored boundaries and native closure before analysis. Only genuinely missing inference should be scheduled after H200 returns; no restoration date is assumed verified.
