"""What each gate retained, and whether its representation cells can be refitted.

Every representation-level verdict in the capability map is provisional until the
readout reassessment completes, after which the representation analyses at every
gate are revisited and, where necessary, recomputed
(:doc:`archive/docs/shared/capability-map-plan.md`). A recomputation is cheap only where the
gate retained enough to refit without another forward pass, and that is a
property of what each gate's own extraction hooked rather than of how much disk
it used.

Two selection kinds are distinguished because they have different retention
requirements, and conflating them is the mistake this module exists to prevent:

``readout_class``
    A different function of the **two admitted extraction depths**' four pooled
    summaries -- a wider projection, the uncompressed features, a nonlinear
    head, or one depth-and-pooling block alone. A gate can serve this from its
    own retained full-width block outputs, with no forward pass anywhere.

``extraction_depth``
    Any block other than the two the admitted extraction hooked, or a
    position-resolved summary. No retained array holds those. A gate can serve
    this only if a depth-resolved extraction exists **on that gate's own
    cohort**; otherwise the recomputation needs a fresh extraction, which is a
    measurement campaign rather than a refit.

The declaration below is the single source for both the inventory report and the
recomputation driver's artefact resolution, so a location cannot drift between
the two. It records locations as the gate records state them, and a probe reports
which are present under which root: the substantive artefacts live on the remote
cluster, so "absent from this host" and "missing" are different findings and are
reported as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

#: The two selection kinds a readout reassessment can return, in the order of
#: increasing retention requirement.
SELECTION_KINDS = ('readout_class', 'extraction_depth')

#: Where a retained artefact lives. ``repository`` is a path under the working
#: tree on either host; ``cluster`` is a path on the remote allocation's shared
#: filesystem, reachable from a pod and not from the workstation.
STORES = ('repository', 'cluster')

#: Roles a retained artefact plays in a recomputation. Named rather than free
#: text, because the driver refuses a gate whose declaration lacks one of the
#: roles a refit consumes.
ROLES = (
    'cohort',                # the frozen support and its digest
    'projected_states',      # the states as the published fit read them
    'full_width_states',     # the same states before any projection
    'depth_states',          # every block's state, from a depth-resolved extraction
    'fold_map',              # outer and inner held-unit membership per split seed
    'penalty_grid',          # the selected penalty of every design in every fold
    'declared_designs',      # the design/control-set declaration and its digest
    'published_increments',  # the increments this recomputation reports beside
    'upstream_increments',   # per-unit increments a re-aggregating gate consumes
)

#: Roles a refit of a representation cell consumes at either selection kind. A
#: gate declaring representation cells without all of these cannot be refitted on
#: identical support, folds, seeds, weighting and label budget.
REQUIRED_ROLES = ('cohort', 'fold_map', 'penalty_grid', 'declared_designs',
                  'published_increments')


@dataclass(frozen=True)
class RetainedArtifact:
    """One retained artefact, its role, and where its own record says it is."""

    role: str
    location: str
    store: str
    note: str = ''

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f'undeclared retention role {self.role!r}')
        if self.store not in STORES:
            raise ValueError(f'undeclared store {self.store!r}')
        if not self.location:
            raise ValueError(f'{self.role}: a retained artefact needs a location')


    def fixed_prefix(self) -> str:
        """The leading path of this location that carries no placeholder.

        Locations are declared with placeholders -- ``<arm>``, ``<wave>`` -- because
        one entry covers a panel. The part before the first placeholder is still a
        concrete path, and it is the part a build-time check can verify: a location
        whose fixed prefix does not exist names nothing, whatever the placeholders
        would have expanded to.
        """
        parts = []
        for part in self.location.split('/'):
            if '<' in part or '*' in part:
                break
            parts.append(part)
        return '/'.join(parts)

@dataclass(frozen=True)
class GateRetention:
    """One gate's representation cells and what it retained for their refit.

    ``representation_cells`` is zero where the gate fitted no representation
    design at all. That is a different state from a gate that fitted one and
    found nothing, and the two are never merged: a gate with no cells has
    nothing to recompute, and reporting it as recomputable would imply a
    measurement it never made.
    """

    gate: str
    kind: str
    question: str
    record: str
    cohort: str
    unit: str
    representation_cells: int
    cell_shape: str
    published_metrics: tuple[str, ...]
    artifacts: tuple[RetainedArtifact, ...]
    supported_selections: frozenset[str]
    missing_for: dict[str, tuple[str, ...]] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    no_cells_reason: str = ''

    def __post_init__(self):
        if self.kind not in ('mechanism_gate', 'control_qualification', 'upstream_study'):
            raise ValueError(f'{self.gate}: undeclared entry kind {self.kind!r}')
        unknown = self.supported_selections - set(SELECTION_KINDS)
        if unknown:
            raise ValueError(f'{self.gate}: undeclared selection kinds {sorted(unknown)}')
        if set(self.missing_for) - set(SELECTION_KINDS):
            raise ValueError(f'{self.gate}: missing_for names an undeclared selection kind')
        if self.supported_selections & set(self.missing_for):
            raise ValueError(f'{self.gate}: a selection kind is both supported and missing')
        if self.representation_cells < 0:
            raise ValueError(f'{self.gate}: cell counts are non-negative')
        if self.representation_cells == 0:
            if not self.no_cells_reason:
                raise ValueError(f'{self.gate}: a gate with no representation cells must say why')
            if self.published_metrics:
                raise ValueError(f'{self.gate}: no representation cells but published metrics')
        else:
            if self.no_cells_reason:
                raise ValueError(f'{self.gate}: {self.representation_cells} cells and a no-cells reason')
            if not self.published_metrics:
                raise ValueError(f'{self.gate}: representation cells with no published metric to '
                                 'report the recomputed increment beside')
        for kind in SELECTION_KINDS:
            if kind not in self.supported_selections and kind not in self.missing_for:
                raise ValueError(f'{self.gate}: {kind} is neither supported nor accounted for')

    def roles(self) -> frozenset[str]:
        return frozenset(artifact.role for artifact in self.artifacts)

    def role_gaps(self) -> tuple[str, ...]:
        """Required roles this gate declares no artefact for."""
        if not self.representation_cells:
            return ()
        return tuple(role for role in REQUIRED_ROLES if role not in self.roles())


_ANCHOR = ('the admitted Readout anchor: 201 assays, 163 wild-type clusters and 25,728 '
           'variants, cohort SHA256 4093ac34368cd9e7be02e5c84a22655ce78cb9bb4e4d58d626f281868dbe7992')
_CLUSTER_UNIT = 'wild-type family at 50% identity'

#: The depth-resolved re-extraction covers the admitted Readout cohort's 17
#: panels for all 34 arms, so every gate fitted on a Readout panel can resolve
#: depth from those archives without an extraction of its own. A gate fitted on
#: any other cohort cannot, whatever it retained.
_DEPTH_FROM_READOUT_CAMPAIGN = (
    'depth-resolved recomputation consumes the readout depth campaign\'s archives on this '
    'same cohort and arms and needs no extraction of its own; it waits on that campaign '
    'completing, including the two arms re-extracting after their container digests drifted')

#: Every gate of the capability map, the two activities that are not gates, and
#: the three upstream studies whose retained arrays the gates' own cells are
#: built from. The upstream entries are here because two gates read them rather
#: than their own extraction, so an inventory that omitted them would report a
#: gate as recomputable while the arrays it consumes were unaccounted for.
GATE_RETENTION: dict[str, GateRetention] = {
    'local_context': GateRetention(
        gate='local_context',
        kind='mechanism_gate',
        question='local chemistry and local context beyond controls built from the same '
                 'local information',
        record='archive/docs/R1/gate-local-context.md',
        cohort=_ANCHOR,
        unit=_CLUSTER_UNIT,
        representation_cells=99,
        cell_shape='33 unconditioned arms at three split seeds on the anchor panel',
        published_metrics=('increment_R_C', 'increment_R_C_wall', 'increment_R_C_P',
                           'increment_R_C_P_wall', 'increment_R_C_P_rf3',
                           'increment_R_after_M_C_P_wall'),
        artifacts=(
            RetainedArtifact('cohort', 'data/context_mutation_rescue/cohort.json', 'cluster',
                             'the anchor cohort as the allocation holds it, digest '
                             '4093ac34368cd9e7be02e5c84a22655ce78cb9bb4e4d58d626f281868dbe7992, '
                             'verified in-pod; archive/logs/R2/readout_20260923/cohort.json is the '
                             'workstation copy of the same bytes'),
            RetainedArtifact('full_width_states',
                             'results/shared/external_baseline/<readout extraction>/<arm>/', 'cluster',
                             'the admitted Readout extractions\' per-state block outputs at full '
                             'hidden width, retained beside the projections'),
            RetainedArtifact('depth_states',
                             'results/shared/external_baseline/<wave>/depth_extract_<arm>/', 'cluster',
                             'the depth campaign\'s per-assay archives on this same cohort, with '
                             'the sharded arms merged under depth_merge_<arm>/; located in-pod '
                             'on 2026-09-24, 104 such directories'),
            RetainedArtifact('fold_map',
                             'results/R5/local_context_20260923/20260923223301_305478e124be/lcgm_*/', 'cluster',
                             '99 cell reports, each carrying outer and inner group membership'),
            RetainedArtifact('penalty_grid',
                             'results/R5/local_context_20260923/20260923223301_305478e124be/lcgm_*/', 'cluster',
                             'the selected ridge penalty per fold and design, in the same reports'),
            RetainedArtifact('declared_designs', 'src/capability/context/local_context.py', 'repository',
                             'the frozen block declaration, content digest '
                             '255b7e3d1e42893e9c05cd201e2ad9685015a24bc14b5aeefbdee102e5aa25c1'),
            RetainedArtifact('published_increments',
                             'results/R5/local_context_20260923/20260923233257_e429ce7f31e4/lcgp_*/', 'cluster',
                             'the 24 extension arms\' cell reports, bound by the 99-cell admission '
                             '6d18d2f40c99e6c969121d0806794c8dd4017af4ef3bc344093e7014b160447e'),
            RetainedArtifact('published_increments', 'archive/logs/R1/local_context_gate_20260923', 'repository',
                             'the local pull of the panel and its dispatch record'),
        ),
        supported_selections=frozenset(SELECTION_KINDS),
        conditions=(_DEPTH_FROM_READOUT_CAMPAIGN,
                    'the qualification receipt e0fa349e3efc9ad7e21a637143103581fb5becd603509b9a7b24a67347931dd4 '
                    'is reused and never re-qualified: re-qualifying against a wider panel would '
                    'lose the outcome-blindness that makes the gate usable',
                    'the control side, the folds, the seeds and the label budget are fixed by these '
                    'artefacts, so a revisit changes only the representation block'),
    ),
    'folding_stability': GateRetention(
        gate='folding_stability',
        kind='mechanism_gate',
        question='information about how a single substitution changes folding stability, beyond '
                 'qualified sequence, chemistry, profile and nonlinear-additive controls',
        record='archive/docs/R4/gate-stability.md',
        cohort='101 MegaScale natural family groups, one background each, 25,856 single '
               'substitutions over 5,664 mutated sites, cohort SHA256 '
               '1e42c3cc1d11d479fcbcce23ffcc37b5799fcf7fc79c65780b1a521fd953e9de',
        unit='MegaScale natural family group',
        representation_cells=99,
        cell_shape='33 arms at three split seeds, each reported on two control sets, two metrics '
                   'and two remote-purge thresholds',
        published_metrics=('representation_mse_increment_primary',
                           'representation_spearman_increment_secondary',
                           'representation_mse_increment_secondary',
                           'representation_spearman_increment_primary'),
        artifacts=(
            RetainedArtifact('cohort', 'results/gate_stability_20260924/cohort.json', 'cluster',
                             'cohort 1e42c3cc…, extraction plan content digest 283b4503…'),
            RetainedArtifact('projected_states',
                             'results/shared/external_baseline/20260924000148_fbfa7ddc1f5f/<arm>/', 'cluster',
                             'one NPZ per background with the projected blocks, the likelihood, the '
                             'token records and the repeat-check quantities; Galactica-30B and '
                             'Qwen2.5-32B under …/20260923232409_9ab5f150f2ac/'),
            RetainedArtifact('full_width_states',
                             'results/shared/external_baseline/20260924024010_7caaa5ac6216/full-<arm>/',
                             'cluster',
                             'a separate retention pass over the identical cohort, plan, sequences '
                             'and batch-one setting, verified complete in-pod on 2026-09-24 for '
                             'all 33 arms: every manifest reads status complete at 101 of 101 '
                             'backgrounds and 25,957 sequences scored, with 101 projected and 101 '
                             'full-width archives per arm'),
            RetainedArtifact('depth_states',
                             'results/shared/external_baseline/<depth3-fs-<arm> wave>/depth3-fs-<arm>/', 'cluster',
                             'the every-block extraction of this gate\'s own 101-background single-mutant stability cohort, dispatched 2026-09-24 and 2026-09-25 in runs 20260924211249_dfb821c212da and 20260924234500_d832d46b7fb0: 32 of 32 cells reconciled against disk, 6,464 archives, the 8 of the first dispatch and the 24 of the rerun. Its one-arm gate compared the admitted block summaries against the retained two-depth archives of the same backgrounds and found them byte-identical at a maximum absolute difference of 0.0'),
            RetainedArtifact('fold_map',
                             'results/shared/external_baseline/20260924024010_7caaa5ac6216/fit-<arm>/'
                             'fit_<arm>.json', 'cluster',
                             'held-out groups of every outer fold at every split seed in the '
                             'nuisance block, with the purged training groups of each remote fit'),
            RetainedArtifact('penalty_grid',
                             'results/gate_stability_20260924/fits/fit_<arm>.json', 'cluster',
                             'the selected ridge penalty for every design in every fold, with the '
                             'design names and their column dimensions'),
            RetainedArtifact('declared_designs', 'src/capability/stability/stability_gate.py', 'repository',
                             'the control ladder, the qualification rule and the two carried '
                             'control sets; controls_qualification.json f4ed91fa… on the cluster'),
            RetainedArtifact('published_increments',
                             'results/gate_stability_20260924/panel/panel.json', 'cluster',
                             'the panel report the recomputed increments are reported beside'),
            RetainedArtifact('published_increments', 'results/R4/gate_stability_20260924', 'repository',
                             'the local pull, including panel.json and the fits directory'),
        ),
        supported_selections=frozenset(SELECTION_KINDS),
        conditions=('the full-width retention pass is complete: verified in-pod for all 33 arms '
                    'at 101 of 101 backgrounds and 25,957 sequences scored each, so the class-level '
                    'recomputation of this gate has no outstanding retention prerequisite',
                    'the two ProGen3 arms are bfloat16 against float32 elsewhere, which is an '
                    'interface constraint of their mixture-of-experts block and travels with any '
                    'refit of their cells',
                    'no arm\'s tokenisation descriptors qualified, so the baseline carries no '
                    'arm-specific column and one baseline per seed serves all 33 arms'),
    ),
    'remote_homology': GateRetention(
        gate='remote_homology',
        kind='mechanism_gate',
        question='whether the single-substitution stability finding holds on a support that '
                 'populates the remote-identity bands every earlier cohort left empty',
        record='archive/docs/R5/gate-remote-homology.md',
        cohort='179 MGnify-derived family groups, 6,291 single substitutions over 6,289 mutated '
               'sites and 305 backgrounds, 6,596 sequences scored per arm',
        unit='family group',
        representation_cells=99,
        cell_shape='33 arms at three split seeds',
        published_metrics=('representation_mse_increment', 'representation_spearman_increment'),
        artifacts=(
            RetainedArtifact('cohort', 'results/R5/remote_homology_20260924/cohort.json', 'cluster',
                             'with endpoint_qualification.json, endpoint_records.json, '
                             'support_declaration.json and controls_qualification.json beside it'),
            RetainedArtifact('full_width_states',
                             'results/shared/external_baseline/20260924081450_45cbf26d2945/rh-<arm>/'
                             'full_<arm>_<background>.npz', 'cluster',
                             'every cell carried --keep-full-features; each background writes one '
                             'full-width archive holding a single `features` array of (rows, 4, '
                             'hidden width) float32 and its metadata, 10,065 of them over 33 arms, '
                             'verified in-pod'),
            RetainedArtifact('projected_states',
                             'results/shared/external_baseline/20260924081450_45cbf26d2945/rh-<arm>/'
                             '<arm>_<background>.npz', 'cluster',
                             'the second archive each background writes, and the one a class refit '
                             'also needs: `projected` at (rows, 4, 256) float32 beside the '
                             '`likelihood` column and the pooled, scored and packed token counts, '
                             'the token ids and offsets, the variant states and positions, the '
                             'hidden width and the two repeat controls. A loader reading only the '
                             'full-width archives would have no likelihood column at all'),
            RetainedArtifact('depth_states',
                             'results/shared/external_baseline/<depth3-rh-<arm> wave>/depth3-rh-<arm>/', 'cluster',
                             'the every-block extraction of this gate\'s own 305-background MGnify-derived cohort, dispatched 2026-09-24 and 2026-09-25 in runs 20260924211249_dfb821c212da: 32 of 32 cells reconciled against disk, 19,520 archives. Its one-arm gate compared the admitted block summaries against the retained two-depth archives of the same backgrounds and found them byte-identical at a maximum absolute difference of 0.0'),
            RetainedArtifact('fold_map', 'results/R5/remote_homology_20260924/fits/fit_<arm>.json',
                             'repository',
                             'each carrying its complete fold map and the first-stage diagnostics '
                             'of the nonlinear response'),
            RetainedArtifact('penalty_grid', 'results/R5/remote_homology_20260924/fits/fit_<arm>.json',
                             'repository',
                             'the selected ridge penalty of every design in every fold, with the '
                             'declared design names and their column dimensions'),
            RetainedArtifact('declared_designs', 'src/capability/stability/stability_gate.py', 'repository',
                             'every feature block is imported from the stability gate rather than '
                             'reimplemented'),
            RetainedArtifact('published_increments',
                             'results/R5/remote_homology_20260924/panel/panel.json', 'repository',
                             'with a digest of every fit it reads'),
        ),
        supported_selections=frozenset(SELECTION_KINDS),
        conditions=('the extraction shares extract_stability_singles.py with the '
                    'folding-and-stability gate, so the repeatable depth option already reaches '
                    'this cohort and only its manifests are missing',
                    'its endpoint has about half the development endpoint\'s headroom, at a '
                    'shared-to-discordance ratio of 1.566 against 3.86, which bounds any '
                    'recomputed increment on it as much as the readout class does'),
    ),
    'external_confirmation': GateRetention(
        gate='external_confirmation',
        kind='mechanism_gate',
        question='whether the residue-level likelihood finding reproduces on a third source with '
                 'a third assay, on domains screened against both development sources',
        record='archive/docs/R1/external-confirmation.md',
        cohort='96 family groups over 428 Domainome domains at a 256-substitution cap each, '
               '109,996 sequences and 6,289,818 residues scored per arm',
        unit='family group',
        representation_cells=99,
        cell_shape='33 arms at three split seeds',
        published_metrics=('representation_mse_increment', 'representation_spearman_increment'),
        artifacts=(
            RetainedArtifact('cohort', 'results/R1/external_confirmation_20260924/cohort.json',
                             'cluster',
                             'with endpoint_qualification.json, extraction_plan.json, '
                             'support_declaration.json, profile_features.npz and the alignment '
                             'screen\'s own query and subject FASTA under screen/'),
            RetainedArtifact('full_width_states',
                             'results/shared/external_baseline/20260924085613_5f8d32cf0368/ec-<arm>/'
                             'full_<arm>_<background>.npz', 'cluster',
                             'every cell carries --keep-full-features; one full-width archive per '
                             'background holding a single `features` array of (rows, 4, hidden '
                             'width) float32, 14,124 of them over 33 arm directories, measured '
                             'in-pod at 139.37 GiB for the whole run directory'),
            RetainedArtifact('projected_states',
                             'results/shared/external_baseline/20260924085613_5f8d32cf0368/ec-<arm>/'
                             '<arm>_<background>.npz', 'cluster',
                             'the second archive each background writes, and the one a class refit '
                             'also needs: `projected` at (rows, 4, 256) float32 beside the '
                             '`likelihood` column and the token descriptors. The two kinds sit in '
                             'one directory here and in two separate runs for the '
                             'folding-and-stability gate, so a loader cannot assume either layout'),
            RetainedArtifact('depth_states',
                             'results/shared/external_baseline/<depth3-ec-<arm> wave>/depth3-ec-<arm>/', 'cluster',
                             'the every-block extraction of this gate\'s own 428-background Domainome cohort, run 20260924234500_d832d46b7fb0, reconciled against disk from the reader\'s side in-pod on 2026-09-26: 33 of 33 lanes report status complete at 428 backgrounds and 109,996 sequences scored each, identical on every arm; every one of the 14,124 archives the manifests declare is present, each carries its full_<arm>_<background>.npz companion holding one features_d<NNN> array of (rows, 2, hidden width) per hooked block, 28,248 archives in all and no .npz of any other kind; cohort digest fca2a53a03e4b7cc8b69a2f2348848d59f3ee073e1c70ee0e4599a854fc474d9 on all 33, equal to the sha256 of results/R1/external_confirmation_20260924/cohort.json; feature_blocks exactly twice block_indices on all 33; 45.300 GPU-hours of summed elapsed time against 57.535 measured on this cohort\'s own admitted two-depth extraction, so hooking every block cost no extra forward pass. Its one-arm gate compared the admitted block summaries against the retained two-depth archives of the same 428 backgrounds and found them byte-identical at a maximum absolute difference of 0.0'),
            RetainedArtifact('fold_map', 'results/R1/external_confirmation_20260924/fits/', 'cluster',
                             '33 per-arm fit records; the entry points are the ones that '
                             'produced the remote-homology gate\'s completed panel'),
            RetainedArtifact('penalty_grid', 'results/R1/external_confirmation_20260924/fits/',
                             'cluster', 'as for the sibling gate, in the same records'),
            RetainedArtifact('declared_designs', 'src/capability/stability/stability_gate.py', 'repository',
                             'the control side of this confirmation is the same code the '
                             'development measurement fitted'),
            RetainedArtifact('published_increments',
                             'results/R1/external_confirmation_20260924/panel/panel.json', 'cluster',
                             'the panel report the recomputed increments are reported beside, '
                             'present on the allocation with this gate\'s 33 fit records'),
        ),
        supported_selections=frozenset(SELECTION_KINDS),
        conditions=('the extraction shares extract_stability_singles.py with the '
                    'folding-and-stability gate, so the repeatable depth option reached this '
                    'cohort as written and its depth-all manifests have since been run',
                    'the two manifest groupings of the depth-all wave are a recording artefact '
                    'and not two code versions: 19 lanes record 7 code files and 14 record 11, '
                    'the four extra naming the literal-text rendering path the other 19 never '
                    'execute, all 7 shared files agree digest for digest, and the last commit '
                    'touching any of the four predates the wave\'s earliest manifest',
                    'there is one assay here, so blocking the designs by assay is degenerate: one '
                    'block and nothing to block on'),
    ),
    'residue_interactions': GateRetention(
        gate='residue_interactions',
        kind='mechanism_gate',
        question='compensatory nonadditivity on a measured four-state double-mutant cycle beyond '
                 'additive, nonlinear-additive, sequence/profile, tokenisation and fitted '
                 'pairwise-coupling controls',
        record='archive/docs/R3/pairwise-epistasis-results.md',
        cohort='64 held family groups of MegaScale wild-type-centred stability cycles, 8,192 '
               'cycles and 217 site pairs at a Kish effective 129.2, cohort SHA256 '
               '8133463ec30293013864685a426a99b69c51c1bac6774b37ebd036915670040f',
        unit='held family group, with the site pair carrying the weight inside a group',
        representation_cells=99,
        cell_shape='33 arms at three split seeds, on the 64-group support and the two further '
                   'declared supports',
        published_metrics=('representation_interaction_contrast_kcal2',),
        artifacts=(
            RetainedArtifact('cohort', 'archive/logs/R3/pairwise_cohort_20260924/cohort.json', 'repository',
                             'the frozen cohort, digest 8133463e…'),
            RetainedArtifact('full_width_states',
                             'results/pairwise_epistasis_20260924/extraction/<arm>/', 'cluster',
                             'per-state block outputs at full hidden width, beside the projected '
                             'ones, so a different projection replays without model inference'),
            RetainedArtifact('depth_states',
                             'results/shared/external_baseline/<depth3-ri-<arm> wave>/depth3-ri-<arm>/', 'cluster',
                             'the every-block extraction of this gate\'s own 64-background double-mutant cycle cohort, dispatched 2026-09-24 and 2026-09-25 in runs 20260924211249_dfb821c212da: 32 of 32 cells reconciled against disk, 4,096 archives. Its one-arm gate compared the admitted block summaries against the retained two-depth archives of the same backgrounds and found them byte-identical at a maximum absolute difference of 0.0'),
            RetainedArtifact('fold_map', 'results/pairwise_epistasis_20260924/fits/', 'cluster',
                             '33 per-arm fit records with held-group membership per support and '
                             'seed, and the row-identity and fold-identity digests'),
            RetainedArtifact('penalty_grid', 'results/pairwise_epistasis_20260924/fits/', 'cluster',
                             'the selected penalty of every design in every fold, in the same records'),
            RetainedArtifact('declared_designs', 'src/capability/interactions/pairwise_epistasis.py', 'repository',
                             'the feature blocks, the nuisance control and the nested comparison'),
            RetainedArtifact('published_increments', 'archive/logs/R3/pairwise_epistasis_20260924/panel',
                             'repository', 'the aggregated panel reports'),
        ),
        supported_selections=frozenset(SELECTION_KINDS),
        conditions=('the row-identity digest renders each target through Python float before repr, '
                    'so a refit reproduces the admitted digest under both NumPy 1.26 and NumPy 2; a '
                    'NumPy-scalar rendering hashes the same rows to different values',
                    'the published representation contrast resolved above zero for no arm of 33, so '
                    'a recomputation reports a recomputed null beside a published null unless the '
                    'selected class changes it'),
    ),
    'higher_order': GateRetention(
        gate='higher_order',
        kind='mechanism_gate',
        question='multi-residue or background-dependent effects not reducible to pairwise terms, '
                 'on the pinned MegaScale bytes',
        record='archive/docs/R3/gate-higher-order.md',
        cohort='60 exact third-order cycles on 2 site pairs in 2 family groups',
        unit='site pair',
        representation_cells=0,
        cell_shape='',
        published_metrics=(),
        artifacts=(
            RetainedArtifact('cohort', 'archive/logs/shared/gate_higher_order_20260923/inventory.json',
                             'repository', 'the declared support, digest 3b0b883e…'),
        ),
        supported_selections=frozenset(),
        missing_for={kind: ('the gate stopped at endpoint qualification and fitted no readout, so '
                            'there is no representation cell to recompute',)
                     for kind in SELECTION_KINDS},
        no_cells_reason='the support is two independent units and the endpoint does not clear its '
                        'own measurement-noise floor, so no model quantity was read at all; a '
                        'fold-identity or leakage condition on a fitted readout is recorded as '
                        'absent rather than satisfied',
    ),
    'higher_order_extended': GateRetention(
        gate='higher_order_extended',
        kind='mechanism_gate',
        question='the same question on ProteinGym, where measured combinatorial states exist',
        record='archive/docs/R3/gate-higher-order-extended.md',
        cohort='2,971 order-3 cubes on 598 site triples across 76 positions at a Kish effective '
               '119.6 site triples',
        unit='site triple',
        representation_cells=0,
        cell_shape='',
        published_metrics=(),
        artifacts=(
            RetainedArtifact('cohort', 'archive/logs/shared/gate_higher_order_extended_20260923/support.json',
                             'repository', 'the declared support, digest ddc71784…'),
        ),
        supported_selections=frozenset(),
        missing_for={kind: ('no readout was fitted, so there is no representation cell',)
                     for kind in SELECTION_KINDS},
        no_cells_reason='the order-3 endpoint fails its own replicate-channel floor on both assays '
                        'where a floor can be measured, so the 33-arm roster was not run and no '
                        'model quantity was read; the roster stays unconditioned on any outcome',
    ),
    'structural_constraints': GateRetention(
        gate='structural_constraints',
        kind='mechanism_gate',
        question='whether model-side signals concentrate in contacting residue pairs relative to '
                 'matched non-contact pairs',
        record='archive/docs/R3/gate-structure-contact.md',
        cohort='111 retained contacts and 53 matched controls at a Kish effective 25.1 control '
               'site pairs, over 63 held groups',
        unit='held group, with the site pair drawn inside it',
        representation_cells=0,
        cell_shape='',
        published_metrics=(),
        artifacts=(
            RetainedArtifact('cohort',
                             'archive/logs/shared/gate_structure_contact_20260924/contact_annotation.json',
                             'repository', 'the label-free annotation, digest 3d07129b…'),
        ),
        supported_selections=frozenset(),
        missing_for={kind: ('part 2 was not run, so no model-side contact number exists for any '
                            'arm and there is no representation cell to recompute',)
                     for kind in SELECTION_KINDS},
        no_cells_reason='part 1 is not passed -- the measured contact-minus-matched-control '
                        'difference does not resolve on the raw endpoint and is bounded near zero '
                        'on the pair-specific one -- so part 2 was not run rather than run and '
                        'reported as uninterpretable',
    ),
    'evolution_adaptation': GateRetention(
        gate='evolution_adaptation',
        kind='mechanism_gate',
        question='information beyond profile statistics, coevolution estimates, homolog retrieval '
                 'and memorization, under six declared stratifications',
        record='archive/docs/R5/gate-retrieval-memorization.md',
        cohort=_ANCHOR + ' for the ranking side; the pairwise stability cohort for the other',
        unit=_CLUSTER_UNIT,
        representation_cells=99,
        cell_shape='33 arms at three split seeds, each decomposed over six stratifications and '
                   '20 populated bands',
        published_metrics=('increment_R_C_P',),
        artifacts=(
            RetainedArtifact('cohort',
                             'archive/logs/R5/gate_retrieval_20260923/strata_declaration.json', 'repository',
                             'all six stratifications with per-unit band assignments and per-band '
                             'support counts, content digest cb56f5fe…'),
            RetainedArtifact('upstream_increments',
                             'archive/logs/R5/gate_retrieval_20260923/readout_panel', 'repository',
                             'the crossed-control cell reports whose per-assay increments the '
                             'ranking strata re-aggregate'),
            RetainedArtifact('fold_map', 'archive/logs/shared/crossed_controls_20260923/reports', 'repository',
                             'the upstream cells\' own fold maps, one held-family map per seed '
                             'shared by all 33 arms'),
            RetainedArtifact('penalty_grid', 'archive/logs/shared/crossed_controls_20260923/reports',
                             'repository', 'the upstream cells\' selected penalties'),
            RetainedArtifact('declared_designs', 'src/capability/context/retrieval_strata.py', 'repository',
                             'the band declarations, the partition check, the two Kish conventions '
                             'and the decomposition identity'),
            RetainedArtifact('published_increments',
                             'archive/logs/R5/gate_retrieval_20260923/retrieval_strata_report.json',
                             'repository',
                             'every interval, support count and effective count of the record'),
        ),
        supported_selections=frozenset(SELECTION_KINDS),
        depends_on=('crossed_controls',),
        conditions=('its published record does not carry a flat summaries map: a stratum estimate '
                    'is addressed as readout_anchor[arm][contrasts][increment_R_C_P][split seed]'
                    '[stratum], so this gate is compared report against report through its own '
                    'stage rather than metric by metric through the driver\'s comparison',
                    'this gate refits nothing: a stratum estimate is the same paired statistic over '
                    'the same fitted held-out predictions on the units the stratum holds, so its '
                    'recomputation is a re-aggregation that consumes the upstream cells\' '
                    'recomputed per-assay increments',
                    'the stratum assignments, the unit counts and the aggregation are unchanged by '
                    'any readout selection, and the decomposition identity must still hold: the '
                    'whole-support estimate is the unit-count-weighted mean of its strata',
                    _DEPTH_FROM_READOUT_CAMPAIGN),
    ),
    'generation_control': GateRetention(
        gate='generation_control',
        kind='mechanism_gate',
        question='whether sampled sequences satisfy a biological requirement relative to '
                 'generators carrying only declared sequence statistics',
        record='archive/docs/R6/gate-generative-control.md',
        cohort='20 cells and 16,000 attempts, of which 14,467 carry a canonical searchable sequence',
        unit='near-duplicate sequence group of the attempt ledger',
        representation_cells=0,
        cell_shape='',
        published_metrics=(),
        artifacts=(
            RetainedArtifact('cohort',
                             'results/R6/generative_control_20260924/gate_endpoints.json',
                             'repository', 'the endpoint record, digest 7e3d34d7…'),
        ),
        supported_selections=frozenset(),
        missing_for={kind: ('the endpoints are properties of emitted sequences; reading a frozen '
                            'representation against them would need an instrument this gate does '
                            'not build',)
                     for kind in SELECTION_KINDS},
        no_cells_reason='no representation readout is applicable at this gate: its endpoints are '
                        'properties of emitted sequences rather than of a fitted readout',
    ),
    'conformational_dynamics': GateRetention(
        gate='conformational_dynamics',
        kind='mechanism_gate',
        question='state transitions and long-range regulatory coupling',
        record='archive/docs/shared/capability-map-plan.md',
        cohort='none: no endpoint is qualified',
        unit='undeclared',
        representation_cells=0,
        cell_shape='',
        published_metrics=(),
        artifacts=(),
        supported_selections=frozenset(),
        missing_for={kind: ('the gate is not launched and no record exists',)
                     for kind in SELECTION_KINDS},
        no_cells_reason='not launched: the gate is opened early only if existing data make a '
                        'qualified endpoint immediately tractable',
    ),
    'molecular_recognition': GateRetention(
        gate='molecular_recognition',
        kind='mechanism_gate',
        question='binding, catalysis and substrate specificity beyond family membership and '
                 'retrieval support',
        record='archive/docs/shared/capability-map-plan.md',
        cohort='none: no endpoint is qualified',
        unit='undeclared',
        representation_cells=0,
        cell_shape='',
        published_metrics=(),
        artifacts=(),
        supported_selections=frozenset(),
        missing_for={kind: ('the gate is not launched and no record exists',)
                     for kind in SELECTION_KINDS},
        no_cells_reason='not launched, and a functional endpoint must be a measurement rather than '
                        'an annotation',
    ),
    'cellular_regulation': GateRetention(
        gate='cellular_regulation',
        kind='mechanism_gate',
        question='expression, localisation, degradation, modification and membrane context',
        record='archive/docs/shared/capability-map-plan.md',
        cohort='none: no endpoint is qualified',
        unit='undeclared',
        representation_cells=0,
        cell_shape='',
        published_metrics=(),
        artifacts=(),
        supported_selections=frozenset(),
        missing_for={kind: ('the gate is not launched and no record exists',)
                     for kind in SELECTION_KINDS},
        no_cells_reason='not launched',
    ),
    'global_additive_context': GateRetention(
        gate='global_additive_context',
        kind='control_qualification',
        question='whether the residual model-side likelihood information is longer-range but '
                 'still additive sequence context',
        record='archive/docs/R5/gate-global-context.md',
        cohort='the residue-interaction cohort, on three declared supports',
        unit='held family group, with the site pair carrying the weight inside a group',
        representation_cells=0,
        cell_shape='',
        published_metrics=(),
        artifacts=(
            RetainedArtifact('declared_designs', 'src/capability/context/global_context.py', 'repository',
                             'the two blocks and their content digest b3687416…, rebuildable from '
                             'the extraction plan alone'),
            RetainedArtifact('full_width_states',
                             'results/pairwise_epistasis_20260924/extraction/<arm>/', 'cluster',
                             'the flagship\'s full-width block outputs, retained beside the '
                             'projected ones for a later representation-side revisit'),
            RetainedArtifact('fold_map', 'results/gate_global_context_20260924/fits/', 'cluster',
                             '33 per-arm records with per-support per-seed held-group membership'),
        ),
        supported_selections=frozenset(),
        missing_for={kind: ('the gate is declared over model likelihood only: GATE_ADDITIONS holds '
                            'the likelihood interaction and nothing else, and no design reads the '
                            'projected representation blocks, so the representation contrast is '
                            'not measured rather than measured and reported',)
                     for kind in SELECTION_KINDS},
        no_cells_reason='declared over model likelihood only, which is why its contrasts are '
                        'reported as final while a representation-level verdict would not be',
    ),
    'readout_panel': GateRetention(
        gate='readout_panel',
        kind='upstream_study',
        question='whether a supervised predictor of frozen representations improves prediction of '
                 'measured mutation effects beyond equally supervised controls',
        record='archive/docs/R2/readout-results.md',
        cohort=_ANCHOR + ', with 16 further admitted panels',
        unit=_CLUSTER_UNIT,
        representation_cells=132,
        cell_shape='34 arms over 17 admitted panels at every split seed each panel declares, as 54 '
                   'fitted (arm, panel) cells covering 132 (arm, panel, seed) fit cells',
        published_metrics=('delta_spearman', 'delta_rank_mse', 'R_minus_raw_M_spearman',
                           'B_R_minus_raw_M_spearman'),
        artifacts=(
            RetainedArtifact('cohort', 'archive/logs/R2/readout_20260923/cohort.json', 'repository',
                             'the frozen cohort, digest 4093ac34…'),
            RetainedArtifact('projected_states',
                             'results/R2/readout_20260923/complete/results/external_baseline/'
                             '20260923103322_004fdfab72bf', 'repository',
                             'the five original arms\' extraction manifests; the expansion arms\' '
                             'manifests and every NPZ live on the cluster'),
            RetainedArtifact('full_width_states',
                             'results/shared/external_baseline/<readout extraction>/<arm>/', 'cluster',
                             'full-dimensional mutant-minus-wild-type block vectors, retained so '
                             'deterministic compression replays without model inference'),
            RetainedArtifact('depth_states',
                             'results/shared/external_baseline/<wave>/depth_extract_<arm>/', 'cluster',
                             'every block\'s state under the two pooled rules for all 34 arms, and '
                             'two position-resolved summaries for the 19 one-token-per-residue '
                             'arms, with the sharded arms merged under depth_merge_<arm>/'),
            RetainedArtifact('fold_map',
                             'results/R2/readout_20260923/complete/results/external_baseline/'
                             '20260923105347_9e1188dbfcda', 'repository',
                             'the admitted analysis reports, each carrying outer and inner fold '
                             'membership, the selected penalties and every per-variant prediction'),
            RetainedArtifact('penalty_grid',
                             'results/R2/readout_20260923/complete/results/external_baseline/'
                             '20260923105347_9e1188dbfcda', 'repository',
                             'the selected ridge penalty per fold and design, in the same reports'),
            RetainedArtifact('declared_designs', 'src/capability/readouts/readout_analysis.py', 'repository',
                             'the eight admitted designs, the projection contract and the fitting '
                             'recipe, called unchanged by both sweeps'),
            RetainedArtifact('published_increments',
                             'results/R2/readout_20260923/final_admission.json', 'repository',
                             'the five-arm admission receipt, whose baseline_tolerance is the '
                             '1e-8 a pipeline-identity control is bound at; staged to the '
                             'allocation at the same path on 2026-09-24 and digest-verified, '
                             'sha256 79f6d2a02d52e14b79b5b041eba541d652c34fb573039c35b29c0ef8c7b4d9a3'),
            RetainedArtifact('fold_map',
                             'results/shared/external_baseline/20260923105347_9e1188dbfcda/'
                             'readout_analysis_<arm>_<panel>/', 'cluster',
                             'the admitted analysis reports as the allocation holds them, 18 '
                             'directories located in-pod, which are the published cells a replay '
                             'is read against'),
        ),
        supported_selections=frozenset(SELECTION_KINDS),
        conditions=('the admitted design is formed one assay at a time; forming it over a whole '
                    'panel departs by up to 7.629e-06 in absolute value at every thread count, and '
                    'that reached 3.483e-08 in the held-out predictions, above the 1e-8 the '
                    'pipeline-identity control is bound at',
                    'the compressed block is a float32 product whose reduction order follows the '
                    'BLAS thread count; the admitted analysis ran it at four threads'),
    ),
    'crossed_controls': GateRetention(
        gate='crossed_controls',
        kind='upstream_study',
        question='whether model-derived features add stable predictive information once '
                 'composition, local-pattern and profile controls are fitted',
        record='archive/docs/shared/hierarchy-crossed-controls.md',
        cohort=_ANCHOR,
        unit=_CLUSTER_UNIT,
        representation_cells=99,
        cell_shape='33 arms at three split seeds on the anchor panel',
        published_metrics=('increment_R_C', 'increment_R_C_P', 'increment_R_C_L_P',
                           'rank_mse_reduction_R_C_L_P'),
        artifacts=(
            RetainedArtifact('cohort', 'archive/logs/R2/readout_20260923/cohort.json', 'repository',
                             'the same anchor cohort, digest 4093ac34…'),
            RetainedArtifact('full_width_states',
                             'results/shared/external_baseline/<readout extraction>/<arm>/', 'cluster',
                             'the admitted Readout extractions, read rather than re-extracted'),
            RetainedArtifact('depth_states',
                             'results/shared/external_baseline/<wave>/depth_extract_<arm>/', 'cluster',
                             'the depth campaign\'s archives on this same cohort'),
            RetainedArtifact('fold_map', 'archive/logs/shared/crossed_controls_20260923/reports', 'repository',
                             'cell reports with the realised group membership at all three seeds'),
            RetainedArtifact('penalty_grid', 'archive/logs/shared/crossed_controls_20260923/reports',
                             'repository', 'the selected penalties, in the same reports'),
            RetainedArtifact('declared_designs', 'src/capability/context/crossed_controls.py', 'repository',
                             'the four nested control sets and the tokenisation block'),
            RetainedArtifact('published_increments', 'archive/logs/shared/crossed_controls_20260923/tables.md',
                             'repository', 'the rendered panel the record\'s tables are cut from'),
        ),
        supported_selections=frozenset(SELECTION_KINDS),
        conditions=(_DEPTH_FROM_READOUT_CAMPAIGN,
                    'the evolution-and-adaptation gate re-aggregates these cells\' per-assay '
                    'increments, so this study is recomputed before that gate is',),
    ),
}


def declaration_digest() -> str:
    """Content digest of the retention declaration, bound into every report."""
    payload = {
        gate: dict(kind=entry.kind, record=entry.record, cohort=entry.cohort, unit=entry.unit,
                   representation_cells=entry.representation_cells,
                   cell_shape=entry.cell_shape,
                   published_metrics=list(entry.published_metrics),
                   artifacts=[dict(role=a.role, location=a.location, store=a.store, note=a.note)
                              for a in entry.artifacts],
                   supported_selections=sorted(entry.supported_selections),
                   missing_for={k: list(v) for k, v in sorted(entry.missing_for.items())},
                   depends_on=list(entry.depends_on), conditions=list(entry.conditions),
                   no_cells_reason=entry.no_cells_reason)
        for gate, entry in sorted(GATE_RETENTION.items())
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def recomputability(gate: str, kind: str) -> dict:
    """Whether one gate's representation cells are refittable at one selection kind.

    Three verdicts, never merged. ``not_applicable`` is a gate that fitted no
    representation design, so there is nothing to recompute and the reason is
    carried. ``requires_re_extraction`` is a gate that fitted one and cannot
    serve this selection from what it retained, with the missing item named --
    the finding this inventory exists to surface. ``recomputable`` is a gate that
    can, with any condition it depends on carried beside it.
    """
    if kind not in SELECTION_KINDS:
        raise ValueError(f'undeclared selection kind {kind!r}')
    entry = GATE_RETENTION[gate]
    gaps = entry.role_gaps()
    if not entry.representation_cells:
        verdict, missing = 'not_applicable', entry.missing_for.get(kind, ())
    elif gaps:
        verdict = 'requires_re_extraction'
        missing = tuple(f'no retained artefact declared for the required role {role}'
                        for role in gaps)
    elif kind in entry.supported_selections:
        verdict, missing = 'recomputable', ()
    else:
        verdict, missing = 'requires_re_extraction', entry.missing_for[kind]
    return dict(gate=gate, selection_kind=kind, verdict=verdict,
                representation_cells=entry.representation_cells,
                cell_shape=entry.cell_shape, unit=entry.unit,
                published_metrics=list(entry.published_metrics),
                missing=list(missing), depends_on=list(entry.depends_on),
                conditions=list(entry.conditions),
                no_cells_reason=entry.no_cells_reason or None)


def probe_artifacts(gate: str, roots: dict[str, Path | None]) -> list[dict]:
    """Presence of one gate's declared artefacts under the given per-store roots.

    A location containing ``<`` is a pattern over arms or cells rather than one
    path, so its expansion is not claimed here -- that would need the arm roster,
    which belongs to the driver's plan. Its **fixed prefix** is claimed, because
    that part carries no placeholder and is a concrete path: a pattern whose prefix
    is absent names nothing whatever it would have expanded to. Exempting patterns
    entirely is how six declared locations reached a driver instead of a build --
    `results/R5/remote_homology_20260924/fits/fit_<arm>.json` among them, whose
    directory does not exist -- so an absent prefix is now ``absent`` and counts
    toward the non-zero exit the inventory script already returns for one.

    A store with no root is ``unreachable_from_this_host``, which is deliberately
    not ``absent``: the substantive artefacts live on the remote allocation, and
    reporting them missing from the workstation would turn a location into a
    defect. The corollary is that a clean run proves nothing about a store whose
    root was not given, so the inventory must be run on the allocation with
    ``--cluster-root`` for its cluster locations to be checked at all.
    """
    records = []
    for artifact in GATE_RETENTION[gate].artifacts:
        root = roots.get(artifact.store)
        pattern = '<' in artifact.location or '*' in artifact.location
        if root is None:
            status = 'unreachable_from_this_host'
            detail = f'no root given for the {artifact.store} store'
        elif pattern:
            prefix = artifact.fixed_prefix()
            if not prefix:
                status = 'pattern_not_probed'
                detail = 'the location begins with a placeholder, so it has no fixed prefix'
            elif (Path(root) / prefix).exists():
                status = 'pattern_prefix_present'
                detail = (f'fixed prefix {prefix} exists; the per-arm expansion is the '
                          "driver's to check against its plan")
            else:
                status = 'absent'
                detail = f'the fixed prefix {prefix} does not exist under this root'
        else:
            path = Path(root) / artifact.location
            if path.exists():
                status = 'present'
                detail = 'directory' if path.is_dir() else f'{path.stat().st_size} bytes'
            else:
                status = 'absent'
                detail = 'declared by the gate record and not found under this root'
        records.append(dict(role=artifact.role, location=artifact.location, store=artifact.store,
                            note=artifact.note, status=status, detail=detail))
    return records


def inventory(roots: dict[str, Path | None]) -> dict:
    """The whole retention inventory, one record per gate and per selection kind."""
    gates = []
    for gate, entry in sorted(GATE_RETENTION.items()):
        gates.append(dict(
            gate=gate, kind=entry.kind, question=entry.question, record=entry.record,
            cohort=entry.cohort, unit=entry.unit,
            representation_cells=entry.representation_cells, cell_shape=entry.cell_shape,
            published_metrics=list(entry.published_metrics),
            roles=sorted(entry.roles()), role_gaps=list(entry.role_gaps()),
            artifacts=probe_artifacts(gate, roots),
            recomputability={kind: recomputability(gate, kind) for kind in SELECTION_KINDS}))
    blocked = {
        kind: sorted(record['gate'] for record in gates
                     if record['recomputability'][kind]['verdict'] == 'requires_re_extraction')
        for kind in SELECTION_KINDS}
    return dict(schema='d1_gate_retention_v1', declaration_sha256=declaration_digest(),
                selection_kinds=list(SELECTION_KINDS), required_roles=list(REQUIRED_ROLES),
                roots={store: (None if root is None else str(root))
                       for store, root in sorted(roots.items())},
                gates=gates,
                counts=dict(
                    entries=len(gates),
                    with_representation_cells=sum(1 for g in gates if g['representation_cells']),
                    representation_cells=sum(g['representation_cells'] for g in gates)),
                requires_re_extraction=blocked)
