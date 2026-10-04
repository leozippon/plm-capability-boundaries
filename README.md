# Capability measurements for protein, text and joint generative models

Code accompanying a manuscript that compares released protein, text and joint language–protein checkpoints on what each can be measured to do. Every measurement states its endpoint, its interface and its controls explicitly, and reports a quantity with its unit, its support and an interval.

The design point is that these are different quantities rather than different scores on one quantity. A likelihood difference ranks supplied variants, a readout fitted on frozen hidden states predicts labels, and a generation procedure produces new sequences. A checkpoint can rank well and generate unremarkably without contradiction, so the code keeps the three apart instead of aggregating them into a single score.

## Results structure

The results are organised into six sections, not a single capability ranking:

| Result | Evidence-led conclusion |
|---|---|
| Protein-specialized or adapted models show mutation-effect gains beyond local and evolutionary controls | Selected released checkpoints retain predictive increments beyond the qualified position-profile and window controls; this is not a universal contrast between protein-exposed and unexposed models. |
| Mutation-site and downstream likelihood terms carry complementary predictive signals | Mutation-spanning token terms (potentially spanning multiple residues) and downstream terms carry complementary predictive signals in some checkpoints, without identifying a training-objective mechanism. |
| Predictive gains extend beyond mutation ranking to abundance and, more selectively, stability | Abundance and single-substitution stability provide support, with gains depending on the endpoint, metric and qualified baseline. |
| Single-mutation signals do not reliably resolve double-mutant interactions | Raw interaction signals are limited, and the cleaned, measured-singles-adjusted panel establishes no positive increment. |
| Homologous context improves prediction but does not establish remote generalization | Context improves selected interfaces, but remote generalization remains unconfirmed because the required close-stratum positive control failed. |
| Predictive gains do not guarantee generative advantage over length-matched protein fragments | Generation must be compared directly with length-matched real fragments, and scoring gains can coexist with lower absolute generation yields. |

These are checkpoint- and protocol-specific findings, not causal effects of protein exposure or model size. An unresolved interval does not establish equivalence or absent information; a failed endpoint qualification is not a model failure. Sequence recognition and predicted fold confidence do not establish measured folding or function.

## Measurement lanes

`R1`–`R6` are legacy storage and measurement labels retained for provenance and operational references, not the numbering of the six scientific sections. Frozen readouts (`R2`) support the mutation-prediction question; abundance (`R1`) and stability (`R4`) share the cross-phenotype question; contact qualification stays with interactions (`R3`). The code packages and retained result paths keep their existing names.

| Legacy lane | Quantity |
|---|---|
| `R1` | Held-family rank correlation between native likelihood and measured assay values, a decomposition of that likelihood over sequence positions, and external abundance confirmation |
| `R2` | Predictive accuracy of readouts fitted on frozen hidden states, compared across readout classes and layer depths against specified sequence controls |
| `R3` | Pairwise interaction increments against a nested, measured-singles-adjusted residual target, plus a contact-structure qualification |
| `R4` | Single-substitution stability association, keeping the support that qualifies an endpoint separate from the support that fits it |
| `R5` | Contrasts between supplied homologous context and composition-matched controls, retrieval bounds, and remote-homology qualification |
| `R6` | Recognition of generated sequences against length-matched corpus fragments, and structure prediction used as an instrument with its own calibration |

Each lane keeps its endpoint, support, controls and interval procedure explicit. Raw-label, adjusted-target and generation comparisons remain separate rather than being pooled into one score.

## Repository contents

| Path | Contents |
|---|---|
| `src/capability/core/` | Shared numbers, paths, sequences and records |
| `src/capability/models/` | Checkpoint scoring |
| `src/capability/` | One package per measurement: mutation, position, readouts, interactions, stability, context, generation |
| `scripts/capability/` | Entry points in the same groups, plus `data/` (dataset acquisition and checksum registry), `stages/`, `gates/` and `reporting/` (derived quantities, figure code and plotted source tables) |
| `scripts/ops/` | Dataset preparation utilities for the composition-matched fold set, the EC-labelled Swiss-Prot corpus and the OpenWebText subset |
| `configs/` | Scientific contract for the generation replication endpoint |
| `tests/` | Test suite, grouped like the library |
| `requirements.txt` | Direct Python dependencies |

## Requirements

Python 3.11 with the pinned direct dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.9.1` without a CUDA build tag. Install the CUDA-enabled build that matches the execution host; the CUDA runtime itself is host-provisioned. Scoring and generation entry points need a GPU. The estimators, panel contracts, campaign recovery and reporting entry points run on CPU.

## Running a measurement

Run every entry point from the repository root and read its declared inputs from `--help`:

```bash
python scripts/capability/context/analyse_crossed_controls.py --help
```

Three files under `scripts/capability/reporting/` are not entry points and do not answer `--help`. `current_manuscript_quantities.py` and `followup_quantities.py` are modules that `build_derived_numbers.py` imports, and `build_ncs_figures.py` runs to completion with an optional `--from-source-data`. Everything else under `scripts/` takes `--help`.

Three entry points are not category-specific:

- `scripts/capability/entrypoints.py` resolves current and historical numbered stage names to their files.
- `scripts/capability/panel_contract.py` derives the foundational panel from `src/capability/core/arms.py`. `--json` prints the resolved contract, `--emit` writes the equivalent shell contract for the launch layer, which is host-local and not distributed, and `--verify` fails if that shell contract is stale.
- `scripts/capability/campaigns.py` recovers an exact historical recipe from the host-local recipe archive and refuses to replace an existing output. That archive records queue slots and their cluster paths, so it is not distributed.

Endpoint-specific campaigns declare their own support and checkpoints, which are not the same panel as that foundational stage.

Checkpoint weights, model directories and dataset paths are environment variables. `src/capability/core/arms.py` reads the model and corpus locations, and the modules and entry points that need other inputs (ProteinGym, InterPro, CATH, Swiss-Prot XML, MegaScale, AlphaFold structures, DIAMOND) read their own. All of them are named in `.env.example`, with a comment saying what each points at. Copy that file to `.env.local` and fill in this machine's paths. `.env.local` is ignored. A variable already exported in the shell is left unchanged, and an unset or empty one keeps the default in the code that reads it. A missing input stops the run with the variable's name. On the cluster, the launch script assigns the path variables before Python starts. Never commit `.env.local`.

## Figures

`scripts/capability/reporting/figure_data/` holds the plotted source tables for the manuscript figures, with the rendered SVG and PNG previews: `main/` for the five main figures and `ncs/` for the supplementary ones. `render_figures.py` redraws the main figures from `main/`, and `render_ncs_figures.py` redraws the supplementary figures from `ncs/`. Neither rescores a checkpoint. Both write their PDFs under `manuscript/figures/`, which they create if it is absent. `build_ncs_figures.py` extracts the supplementary source tables from retained run outputs that are not distributed, so on a clone only `build_ncs_figures.py --from-source-data`, which hands over to the renderer, runs. `validate_manuscript.py` checks the manuscript sources and figure provenance against `figure_data/` and needs the manuscript directory, which is not distributed.

## Tests

```bash
python -m pytest -q
```

`pytest.ini` sets `testpaths = tests`. The suite covers estimator behaviour, contract checks and negative paths, and `ruff` is pinned alongside it for linting.

## Data, results and models

Datasets, checkpoint weights, run outputs, per-cell receipts and runtime logs are not distributed, and the manuscript evidence derived from them is distributed only as the figure source tables described above. Datasets are staged locally from their upstream releases, and their size and licences do not permit redistribution from this repository. Each measurement entry point declares the inputs it expects in `--help`, and each retained output records the support it was computed on, so a run can be checked against what it declares rather than against a number restated in prose.

A few retained records name absolute paths on the authoring host. The oracle inputs in `configs/generation_replication_manifest.json` are the clearest case: those keys are the identity the completed replication was digested against, so they are kept as written and are not expected to resolve on another machine.

## Licence

The code in this repository is released under the MIT licence; see `LICENSE.md`.

## Citation

Citation details for the accompanying manuscript will be added on publication.
