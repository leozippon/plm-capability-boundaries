# E19: forcing an early residue and reading its prescribed partner

E02 asked whether a generative protein model's conditional at position *i* already
favours the residue it will later contact. That reading is not identifiable as an
intervention for a causal model — the conditional at *i* is a function of the
prefix, so a later partner is outside the conditioning set — and the predictive
quantity E02 measured instead turned out to depend on the partner only through its
residue identity, which makes it a statement about the composition of contacting
partners rather than about position-specific anticipation.

In generation the prefix *is* the conditioning set, so forcing the residue at *i*
is a genuine intervention. This lane runs that intervention and reads what the
model emits afterwards.

The design, the frozen constants and everything the experiment cannot establish
live in `src/capability/forcing/forcing_design.py`. Read that module before
changing anything here; the numbers in it are the pre-registration.

## The three stages, and where each one has to run

**Build the cohort — Compute host only.** `build_forcing_cohort.py` selects the
backbone panel from the staged AlphaFold release, chooses each backbone's
prescribed anchor/partner pairs with their matched non-contacting controls, and
pairs units reciprocally so that the two conditions force the same multiset of
residues by construction. It needs the CATH superfamily table and the DIAMOND
binary, neither of which the H200 pod holds, so the cohort is built here and
pushed to GPFS as a frozen artefact. Every later stage reads that file and
re-derives nothing.

**Band the reference-database coverage — Compute host only.**
`screen_forcing_coverage.py` searches the panel against the staged UniRef90
release and bands each backbone by its best hit over the query. The 88 GB index
lives on this host; the pod's `TRANSFER_DIAMOND_DIR` and `TRANSFER_DIAMOND_DB`
are dead declarations. The band is distance to a published reference release and
is not pretraining exposure. Without this sidecar the gate analysis reports no
coverage stratum and says so.

**Generate and analyse — pod.** `run_forcing_generation.py` runs one arm, forcing
each unit's anchor under both conditions and generating only as far as the
furthest read position, which is what keeps the gate cheap. It also takes the
teacher-forced reading, which costs one forward pass per cell because every
intervening residue is then known. Cells are individually resumable.
`analyse_forcing_gate.py` then forms the endpoint, its variants, its strata, its
permutation calibration and the panel-wide band.

The campaign manifest is `h200/campaigns/campaign_forcing_gate_20261010.tsv`.

## Two things that bite

The sampler here is explicit rather than `generate`. That is not a preference:
ProGen2's released modelling code carries a cache current Transformers no longer
builds, RITA's does not inherit `generate` at all and its `forward` rejects
`cache_position`, and the teacher-forced mode needs a conditional `generate` never
produces — so with the library in the loop the two modes decode under two
implementations of one policy. One sampler removes the question.

Inference is float32. RITA's attention refuses bfloat16 outright, and a
reduced-precision logit row ties enough logits at the nucleus boundary to move the
sampling distribution by more than this experiment's claimable effect can afford.
