# R5 参数化记忆的生成能力 — Generative expression

Given fixed weights, what does a checkpoint express? This module holds free generation, native family-label or target-own-prefix conditional sampling, Profile guidance, forced-anchor generation and screening. Natural fragments and family-specific explicit statistical samplers provide the controls; access to an external homologous record is instead the question of R4.

## The question

Generation is compared with real fragments of matched length because length and repetitiveness can strongly affect these readouts. Recognition against the natural record does not establish folding, stability or function. Guidance is evaluated against its matched unguided baseline; class-conditioned interfaces keep their declared matched/mismatched-class controls. Removing a required class tag is not automatically a valid unconditioned control.

## Code

- Unconditional and family-conditioned generation: [`src/capability/generation/unconditional_generation.py`](../../src/capability/generation/unconditional_generation.py), [`conditioned_generation.py`](../../src/capability/generation/conditioned_generation.py), [`conditional_contrast.py`](../../src/capability/generation/conditional_contrast.py)
- Generation controls, replication and evidence: [`src/capability/generation/generation_controls.py`](../../src/capability/generation/generation_controls.py), [`generation_replication.py`](../../src/capability/generation/generation_replication.py), [`generation_evidence.py`](../../src/capability/generation/generation_evidence.py)
- Guidance and decoding sweeps: [`src/capability/decoding/guided_decoding.py`](../../src/capability/decoding/guided_decoding.py), [`guidance_strength.py`](../../src/capability/decoding/guidance_strength.py), [`decoding_sweep.py`](../../src/capability/decoding/decoding_sweep.py)
- Forced-anchor generation and the modification-extent ladder: [`src/capability/forcing/forced_completion.py`](../../src/capability/forcing/forced_completion.py), [`src/capability/ladder/design.py`](../../src/capability/ladder/design.py), with the protocol in [`scripts/capability/ladder/README.md`](../../scripts/capability/ladder/README.md)
- The generated side of evaluation: [`src/capability/evaluation/likelihood_ranking.py`](../../src/capability/evaluation/likelihood_ranking.py), [`generated_phenotype.py`](../../src/capability/evaluation/generated_phenotype.py), [`domain_stability.py`](../../src/capability/evaluation/domain_stability.py)
- Shared tools: [`family_oracle.py`](../../src/capability/generation/family_oracle.py) for family recognition and [`near_duplicates.py`](../../src/capability/generation/near_duplicates.py) for length matching

## Inputs

Local and not distributed: `data/omg_prot50_20261010`, `data/kmer_background`, `data/uniref50` and `data/homology_db` for identity, and the per-campaign generated pools under `data/e20_*`. The replication campaign is declared in [`configs/generation_replication_manifest.json`](../../configs/generation_replication_manifest.json), whose oracle keys are digest-bound host paths kept as written and expected not to resolve elsewhere.

## Evidence

Local result trees, all under `results/R6/`: `generation_protocol_20260905`, `generation_inputs_20260905`, `generation_evidence_20260905`, `generation_evaluation_20261008`, `generation_replication_20260927`, `generation_followup_20260923`, `generation_novelty_20261008`, `generation_length_sensitivity_20260926`, `conditioned_generation_20260826`, `decoding_sweep_20261010`, `decoding_frontier_20261010`, `decoding_novelty_20261010`, `forcing_*`, `unconditional_diversity_*`, the guided-generation campaign `e20_*` with its confirmatory analysis, and the modification-extent ladder `ladder_scoped_20261011` and `ladder_validated_20261011`.

## Shared boundaries

Two checkpoint families appear here and in post-training adaptation: this module owns what a model expresses, that module owns the before/after contrast between a base and its tuned sibling. The evaluation instruments are shared, so a recognition or selectivity number formed on tuned arms is read in post-training adaptation even though its tooling lives here. Structural confidence is an instrument with its own length and novelty dependence and never evidence of a measured fold. Historical E11's external homologous-sequence-prefix experiment belongs to [R4](../context-access/README.md), despite its physical storage under `results/R6`; ordinary family-label and target-own-prefix sampling remain here, with tools shared as needed.

[Deferred cross-family-coherence diagnostic](cross-family-interference.md): design and timing only; no additional experiment is commissioned in this round. It compares the same frozen ProGen2-medium under BOS versus native target-own amino-acid prefixes, without external homolog context. Cross-family interference is an explanatory hypothesis; this proposed R5 phenotype diagnostic would not by itself establish parameter competition.
