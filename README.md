# Capability measurements for protein, text and joint generative models

Code accompanying a manuscript that compares released protein, text and joint language–protein checkpoints on what each can be measured to do. Every measurement states its endpoint, its interface and its controls explicitly, and reports a quantity with its unit, its support and an interval.

The design point is that these are different quantities rather than different scores on one quantity. A likelihood difference ranks supplied variants, a readout fitted on frozen hidden states predicts labels, and a generation procedure produces new sequences. A checkpoint can rank well and generate unremarkably without contradiction, so the code keeps the three apart instead of aggregating them into a single score.

The current synthesis is that evolutionary memory carries genuine phenotype-discriminative signal, while its likelihood-based predictive value overlaps strongly with explicit evolutionary statistics. The statistical ladder measures conditional gains in both directions; unequal readout capacity limits an information-containment interpretation but does not invalidate those comparisons. Fine-tuning can produce useful local generative bias, consistent with redistribution within a narrower part of the evolutionary record, without yet establishing general innovative protein-design capability. Family-statistical relearning is a plausible explanation, not an identified mechanism.

## Results structure

The results are organised by question, not into a single capability ranking. [`experiments/`](experiments/README.md) defines the current scientific modules R1–R6 — statistical memory, quantitative readouts, interaction readouts, context access, generative expression and post-training adaptation — with their code entry points, inputs and retained results. The modules are documentation only. No dataset, result tree, schema or campaign identifier moved, and no two experiments inherit a sample, an endpoint or an estimator from being filed together; the grouping is where a reader starts, never the unit a number is read in.

These are checkpoint- and protocol-specific findings, not causal effects of protein exposure or model size. An unresolved interval does not establish equivalence or absent information; a failed endpoint qualification is not a model failure. Sequence recognition and predicted fold confidence do not establish measured folding or function. Small conditional increments and their upper bounds, rather than non-significance alone, support the statistical-overlap interpretation.

## Legacy storage lanes

Historical storage `results/R1/`–`results/R6/` and historical measurement or campaign labels retain their original identifiers for provenance and operational references. They are separate namespaces from the current scientific modules R1–R6 in [`experiments/`](experiments/README.md), not a numbering of the paper's Results sections. In the historical storage namespace, frozen readouts (`R2`) support current R1, abundance (`R1`) and stability (`R4`) support current R2, and contact qualification (`R3`) is auxiliary to current R3. The table below describes historical storage only; the crosswalk in [`experiments/README.md`](experiments/README.md) changes no code package, path or retained identifier.

| Historical storage lane | Quantity |
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
| `experiments/` | Navigation modules that group the measurements by question, each naming its code, input and result entry points |
| `src/capability/core/` | Shared numbers, paths, sequences and records |
| `src/capability/models/` | Checkpoint scoring |
| `src/capability/` | One package per measurement: mutation, position, readouts, interactions, stability, context, generation, forcing, decoding, evaluation; `baselines/` holds the explicit statistical controls (profile and Potts scoring, cohort descriptors), `ladder/` the modification-extent generation ladder, `extensions/` the later estimand declarations, `general_plm/`, `esm3_ext/` and `memory_theory/` the exploratory multi-paradigm, multimodal and associative-memory suites, `unified/` the cross-paradigm 84-checkpoint (65 core + 19 expanded `DPLM-2`, `ProteinMPNN`, and `DiMA`) unified yardstick spanning R1–R6, and `benchmark/` the reusable external capability benchmark protocol, validator, adapters and CLI runner |
| `scripts/capability/` | Entry points in the same groups (including `general_plm/`, `esm3_ext/`, `memory_theory/`, `unified/` and `unified/run_capability_benchmark.py`), plus `data/` (dataset acquisition and checksum registry), `stages/`, `gates/` and `reporting/` (derived quantities, figure code and plotted source tables) |
| `scripts/ops/` | Dataset preparation utilities for the composition-matched fold set, the EC-labelled Swiss-Prot corpus and the OpenWebText subset |
| `configs/` | Declarations read by the entry points: the generation replication manifest and the matched-discrimination manifest, the explicit-statistics cohort descriptors, and the family-conditionals, family-distribution and objective-tuned-prediction contracts |
| `tests/` | Test suite, grouped like the library (including `general_plm/`, `esm3_ext/`, `memory_theory/` and `unified/`) |
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

Run measurement entry points from the repository root. For runners with an argument parser, start with `--help`:

```bash
python scripts/capability/context/analyse_crossed_controls.py --help
```

Reporting helpers, including `current_manuscript_quantities.py`, `followup_quantities.py` and the shared palette definitions, are imported modules rather than commands. Dated analysis recipes are retained producers, not general command-line interfaces: inspect their source and recorded invocation before running them, because `--help` may not be implemented. The historical `build_ncs_figures.py` extractor likewise runs to completion, with an optional `--from-source-data`, rather than providing `--help`.

Three entry points are not category-specific:

- `scripts/capability/entrypoints.py` resolves current and historical numbered stage names to their files.
- `scripts/capability/panel_contract.py` derives the foundational panel from `src/capability/core/arms.py`. `--json` prints the resolved contract, `--emit` writes the equivalent shell contract for the launch layer, which is host-local and not distributed, and `--verify` fails if that shell contract is stale.
- `scripts/capability/campaigns.py` recovers an exact historical recipe from the host-local recipe archive and refuses to replace an existing output. That archive records queue slots and their cluster paths, so it is not distributed.

Endpoint-specific campaigns declare their own support and checkpoints, which are not the same panel as that foundational stage.

Checkpoint weights, model directories and dataset paths are environment variables. `src/capability/core/arms.py` reads the model and corpus locations, and the modules and entry points that need other inputs (ProteinGym, InterPro, CATH, Swiss-Prot XML, MegaScale, AlphaFold structures, DIAMOND) read their own. The dataset and checkpoint variables are named in `.env.example`, with a comment saying what each points at. Copy that file to `.env.local` and fill in this machine's paths. Three variables are read by code and deliberately absent from it: `TRANSFER_PROJECT_ROOT` and `TRANSFER_PACKAGE_ROOT`, which the host-local cluster launch layer (`h200/h200_env.sh`) exports before Python starts, and `PYMOL_EXECUTABLE`, which only names an existing PyMOL binary for `render_framework_structures.py` and has `--pymol` as its equivalent. `.env.local` is ignored. A nonempty shell value takes precedence over `.env.local`; an unset shell variable allows the local file and then the code default. An explicitly empty shell value selects the code default without consulting the local file. A missing input stops the run with the variable's name. On the cluster, the launch script assigns the path variables before Python starts. Never commit `.env.local`.

## Figures

`scripts/capability/reporting/figure_data/` holds the source tables and the previews drawn from them: `main/` for six main-figure previews, `Fig1` through `Fig6`, and `ncs/` for four supplementary figures. The full quantitative renderer redraws the five quantitative main figures, `Fig2` through `Fig6`, and all four supplementary figures from distributed tables and bundled illustrative inputs, without opening raw results, rescoring checkpoints or resampling observations:

```bash
python scripts/capability/reporting/render_figures.py
```

Every output of those nine — PDF, SVG, PNG and plotted table — is written beside the source tables under `figure_data/`, and none of it reaches the submission: two tests assert that a full render leaves `manuscript/submission/figures/` untouched. They are previews of the distributed evidence and of the receipt that binds it, not submission artwork. The submitted figures are the author's Figma exports, inserted byte for byte with their original fonts retained, as `manuscript/package-manifest.json` records under `current_figure_master`; what this repository distributes is their source tables rather than the code that drew them. `publication_paths.py` accordingly maps exactly one figure ID to a submission filename, Figure 1, which `render_framework.py` assembles; the historical `Fig2`–`Fig6` and supplementary IDs stay fixed where the retained tables carry them, under `figure_data/`. Editable handoff exports remain under `manuscript/figures/`. Short checkpoint labels share one full-name key, exported as `figure_data/checkpoint-name-key.csv` and a supplementary LaTeX table. Verified illustrative WT insets are bundled under `figure_data/structure-insets/` with image digests and coordinate provenance. `render_ncs_figures.py` remains a supplementary-only entry point; run the full renderer first after changing rendering code or shared styling. Source extraction is separate: `render_figures.py --refresh-source-tables` explicitly refreshes the clean-interaction and position-support tables from local retained receipts. `build_ncs_figures.py` extracts other supplementary sources from non-distributed run outputs; its `--from-source-data` option instead hands over to the retained-source renderer.

Fig. 1 is assembled separately by `render_framework.py`: it embeds the author's current `manuscript/assets/framework-original.png` directly and adds a results-summary band. The supplied PNG is an input, never an output destination. `render_framework_structures.py` separately produces source-linked illustrations with CPU PyMOL; regenerating these requires local coordinates and retained records. Structure selection, exact residue mapping and unchanged-WT overlays are recorded with source hashes. Bundled derived insets allow the quantitative figures to render without those private inputs.

Shared palette definitions keep repeated visual roles consistent across generated graphics, and signed heat maps map zero exactly to white. The supplied framework retains its original colors. Generated text uses editable 7 pt body labels and 8 pt bold panel letters at manuscript insertion size; the original framework's burned-in text is outside that certificate. Its native raster resolution is reported separately from the 450 dpi composite preview. Render receipts bind source inputs, renderer dependencies and output files. `validate_manuscript.py` checks current manuscript structure, provenance, generated-text typography and layout while reporting the supplied-raster limitations explicitly. It requires the non-distributed manuscript directory and does not certify scientific completeness or author approval.

## Tests

```bash
python -m pytest -q
```

`pytest.ini` sets `testpaths = tests`. The suite covers estimator behaviour, contract checks and negative paths, and `ruff` is pinned alongside it for linting. Code-only and synthetic tests run in a clean checkout. Integration tests requiring non-distributed data, campaign receipts, model weights or manuscript assets explicitly skip when their prerequisites are absent. A skip reports unavailable inputs, not a successful validation; supplied inputs remain subject to the tests' format and provenance checks. A public-checkout pass is therefore not a reproduction of the scientific measurements. Run the retained-material checks on a suitably staged host as a separate validation.

## Data, results and models

Datasets, checkpoint weights, run outputs, per-cell receipts and runtime logs are not distributed, and the manuscript evidence derived from them is distributed only as the figure source tables described above. Datasets are staged locally from their upstream releases, and their size and licences do not permit redistribution from this repository. Input declarations and retained outputs identify the support used by each measurement, so a run can be checked against those declarations rather than against a number restated in prose.

A few retained records and digest-bound historical producers name absolute paths on the authoring host. The oracle inputs in `configs/generation_replication_manifest.json` are one example: those keys are the identity the completed replication was digested against, so they are kept as written and are not expected to resolve on another machine. These historical paths are provenance constraints, not portable defaults; do not silently redirect them to a different input or checkout.

## Licence

The code in this repository is released under the MIT licence; see `LICENSE.md`.

## Citation

Citation details for the accompanying manuscript will be added on publication.
