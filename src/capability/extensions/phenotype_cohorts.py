"""Candidate phenotype cohorts: where each one comes from and how it is read.

One place names every cohort the breadth programme considered, so that a refusal
is as traceable as an admission. Each entry declares the phenotype, the endpoint
with its unit and direction, the prepared artefacts it is read from, and -- for a
candidate refused before any reading -- the evidence for the refusal.

The readers do no preparation. Every cohort here was acquired, identity-checked
and screened by its own committed preparation stage, and these functions only
project those artefacts onto the one row shape the breadth fits consume
(:class:`..phenotype_breadth.PhenotypeRow`). A reader never repairs a source, and
a row whose declared substitution disagrees with its own sequences raises instead
of being dropped.

Direction is declared, never inferred from a title. For the activity partitions
the direction is a property of the named physical quantity
(:data:`..phenotype_breadth.ACTIVITY_DIRECTIONS`); for binding it is the sign of
a free-energy change; for the membrane and endogenous-fitness cohorts it is the
direction the source's own export records.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .phenotype_breadth import ACTIVITY_DIRECTIONS, Cohort, PhenotypeRow, sha_text

#: Measurement directory of the prepared phenotype payloads and results.
MEASUREMENT = 'phenotype_followups_20261007'

#: Candidates closed on recorded evidence before any row is read. They are
#: listed because "we looked and it fails" is a result; each carries the
#: artefact or audit that closed it.
DECLARED_REFUSALS: tuple[dict[str, Any], ...] = (
    {'cohort': 'dhfr_bms_romanowicz2025', 'phenotype': 'antifolate-resistance growth',
     'status': 'refused',
     'blockers': ['the release publishes no per-variant fitness table at all, only barcode counts '
                  'and an external analysis document',
                  'one family group against the floor, recorded in the remote-acquisition audit'],
     'evidence': 'results/R5/remote_acquisition_20260925/remote_acquisition_audit.json'},
    {'cohort': 'groqseq_tev_align2026', 'phenotype': 'protease function in a split reporter',
     'status': 'refused',
     'blockers': ['11,413 sequences collapse to 7 family groups at the frozen 30%/80% edge rule, '
                  'one group below the floor'],
     'evidence': 'results/R5/remote_acquisition_20260925/remote_acquisition_audit.json'},
    {'cohort': 'mgnify_stability_cho2026', 'phenotype': 'folding stability (recalibrated dG)',
     'status': 'refused',
     'blockers': ['stability is not an additional phenotype: it is the existing Result 3 endpoint',
                  'the release shares assay technology and an author with the development '
                  'stability endpoint, so it cannot carry an independent-replication reading',
                  'its two protease channels report systematically different quantities, which '
                  'closed it as an endpoint "on a mechanism"'],
     'evidence': 'results/R5/remote_acquisition_20260925/aggregation_endpoint_audit.json'},
    {'cohort': 'his3_pokusaeva_2019', 'phenotype': 'organismal fitness (HIS3 complementation)',
     'status': 'refused',
     'blockers': ['one protein family, so one independent group against the floor',
                  'organismal fitness is already the largest endpoint class of the anchor, so a '
                  'single-family set adds no phenotype breadth',
                  'no licence record; treated as all-rights-reserved'],
     'evidence': 'data/dataset_registry.json'},
)


def _float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{label}: label is not a number') from error


def activity_quantity(assay: str) -> tuple[str, dict[str, Any]]:
    """The measured-quantity class of a VenusMutHub activity partition.

    The partition identifier is ``<accession>_<quantity descriptors>``, and every
    quantity keyword appears as a whole descriptor token. Matching is therefore
    on tokens only -- a substring test would read ``kcatkm`` as ``km`` -- in the
    declaration order of :data:`..phenotype_breadth.ACTIVITY_DIRECTIONS`, so a
    preference ratio is recognised before the efficiency it is a ratio of. A
    partition naming no declared quantity is refused, never guessed at.
    """

    tail = assay.split('_', 1)[1].lower() if '_' in assay else assay.lower()
    tokens = set(re.split(r'[^a-z0-9]+', tail))
    for keyword, record in ACTIVITY_DIRECTIONS.items():
        if keyword in tokens:
            return keyword, record
    return 'unrecognised', {'direction': None,
                            'reason': 'the partition names no declared measured quantity'}


def read_activity(root: Path) -> Cohort:
    """VenusMutHub single-mutant activity partitions, pinned at their revision."""
    registry = (root / 'results/extensions' / MEASUREMENT /
                'activity/final-20261007/admissible-identity-registry.jsonl')
    cohort = Cohort(
        key='activity_venus', phenotype='direct molecular activity',
        endpoint=('per-partition catalytic activity in the partition\'s own reported quantity '
                  '(turnover number, catalytic efficiency, maximal velocity or reported activity); '
                  'larger is more active'),
        source=f'VenusMutHub single_mutant/activity, pinned revision, via {registry.name}',
        quantitative_unit=None,
        notes=['The 130 partitions report four different quantities on scales that are not shared '
               'between partitions, and only two partitions carry a source unit trace, so the '
               'quantitative endpoint is refused for the whole cohort rather than pooled.',
               'A partition identifier is an assay partition, not a verified experimental '
               'condition; it is used as the ranking background and nothing more.'])
    refusals: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    for line in registry.open(encoding='utf-8'):
        record = json.loads(line)
        if not record.get('admissible_identity_single') or not record.get('finite_label'):
            refusals['not_admissible_single_or_nonfinite'] += 1
            continue
        quantity, declaration = activity_quantity(record['assay'])
        classes[quantity] += 1
        if declaration['direction'] is None:
            refusals[f'undeclared_direction_{quantity}'] += 1
            continue
        mutation = record['mutation']
        cohort.rows.append(PhenotypeRow(
            cohort=cohort.key, row_id=record['source_row_id'], background=record['assay'],
            unit=record['wt_sha256'], wildtype=record['wt_sequence'],
            mutant_sequence=record['original']['mutated_sequence'],
            position=int(mutation['position']), wt_aa=mutation['wt_aa'], mt_aa=mutation['mt_aa'],
            label=_float(record['source_score'], record['source_row_id']),
            direction=int(declaration['direction']), label_unit=None,
            source_sha256=record['source_sha256']))
    cohort.refusals = {k: int(v) for k, v in refusals.items()}
    cohort.notes.append('measured-quantity classes read from the partition identifiers: '
                        + json.dumps(dict(sorted(classes.items())), sort_keys=True))
    return cohort


def read_binding(root: Path) -> Cohort:
    """SKEMPI 2.0 single substitutions joined to their operational PDB chain states."""
    base = root / 'results/extensions' / MEASUREMENT / 'binding'
    labels: dict[str, dict[str, Any]] = {}
    for line in (base / 'admission.jsonl').open(encoding='utf-8'):
        record = json.loads(line)
        if not record.get('admitted') or not record.get('single_substitution'):
            continue
        labels[record['source_row_id']] = record
    cohort = Cohort(
        key='binding_skempi', phenotype='physical protein-protein binding affinity',
        endpoint=('ln(Kd_mutant / Kd_wildtype) of a protein-protein complex, dimensionless; '
                  'positive weakens binding, so the oriented phenotype is its negative'),
        source='SKEMPI 2.0 affinities with PDB entity chain states, via binding/mapping',
        quantitative_unit='dimensionless ln(Kd_mutant/Kd_wildtype)',
        construct_status=('unverified: the deposited PDB entity may be engineered or truncated '
                          'relative to the construct whose affinity was measured'),
        construct_tolerated=(
            'the endpoint is a within-construct difference. Both scored states carry the same '
            'construct, so a tag, a truncation or an engineered residue elsewhere in the entity '
            'shifts the wild-type and the mutant likelihood together and cancels in their '
            'difference. The cohort therefore supports a claim about the likelihood difference of '
            'a substitution and no claim about the absolute likelihood of the measured construct, '
            'and that boundary is what the construct mismatch costs'),
        notes=['The scored state is the mutated chain alone. A single-sequence likelihood cannot '
               'see the partner, so this measures whether the chain\'s own likelihood carries '
               'binding information, not whether the model represents the interface.',
               'The ranking background is the complex, the mutated chain and the source condition '
               'group (reference, method, temperature and wild-type affinity) together. Neither '
               'of the first two may be dropped: one publication reporting several complexes at '
               'one temperature lands in a single condition group, and within one complex the '
               'source reports mutations on either partner, whose sequences are unrelated. '
               'Ranking across either boundary would not be a rank. The composite is still a '
               'provisional proxy and not proof of matched biochemical conditions.',
               'Repeated measurements of one complex and mutation by several publications are '
               'retained as separate rows; they are not averaged and they are not independent.'])
    refusals: Counter[str] = Counter()
    for line in (base / 'mapping/row-mapping.jsonl').open(encoding='utf-8'):
        record = json.loads(line)
        if record.get('blockers') or record.get('status') != 'validated_operational_PDB_reference':
            refusals['mapping_blocked_or_unvalidated'] += 1
            continue
        label = labels.get(record['source_row_id'])
        if label is None:
            refusals['no_admitted_affinity_label'] += 1
            continue
        mutations = label['mutations']
        if len(mutations) != 1:
            refusals['not_single_substitution_label'] += 1
            continue
        cohort.rows.append(PhenotypeRow(
            cohort=cohort.key, row_id=record['source_row_id'],
            background=('{}|{}|{}'.format(record['complex_id'], record['mutated_auth_chain'],
                                          record['condition_group'])),
            unit=record['wt_sha256'], wildtype=record['wt_sequence'],
            mutant_sequence=record['mutant_sequence'],
            position=int(record['sequence_index_1based']),
            wt_aa=mutations[0]['wt_aa'], mt_aa=mutations[0]['mt_aa'],
            label=_float(label['ln_kd_ratio'], record['source_row_id']),
            direction=-1, label_unit='ln(Kd_mutant/Kd_wildtype)',
            source_sha256=sha_text(record['source_row_id'] + record['mutant_sha256'])))
    cohort.refusals = {k: int(v) for k, v in refusals.items()}
    return cohort


def read_membrane(root: Path) -> Cohort:
    """RHO, F9 and SGCA surface-display and secretion channels."""
    base = root / 'results/extensions' / MEASUREMENT / 'membrane'
    manifest = {record['channel']: record for record in json.loads(
        (base / 'channel-manifest.json').read_text(encoding='utf-8'))}
    wildtypes = json.loads((base / 'canonical-WT-manifest.json').read_text(encoding='utf-8'))
    states: dict[str, dict[str, Any]] = {}
    for line in (base / 'canonical-mutant-states.jsonl').open(encoding='utf-8'):
        record = json.loads(line)
        states[record['state_id']] = record
    cohort = Cohort(
        key='membrane_multistep', phenotype='membrane trafficking, surface display and secretion',
        endpoint=('per-channel reporter signal as the source export normalises it; larger is more '
                  'surface or secreted protein'),
        source='MaveDB MultiSTEP F9 channels, RHO surface and proximity methods, SGCA surface',
        quantitative_unit=None,
        notes=['RHO is the anchor protein OPSD_HUMAN_Wan_2019: a new measurement on an old '
               'protein, not a new independent protein.',
               'SGCA supplies no source full-length protein wild type, so its rows cannot be '
               'joined to a protein state at all.',
               'The F9 channels differ in the antibody they use, so they are channels of one '
               'protein and not independent proteins.'])
    refusals: Counter[str] = Counter()
    for line in (base / 'channel-rows.jsonl').open(encoding='utf-8'):
        record = json.loads(line)
        if not record.get('accepted') or record.get('score_status') != 'finite':
            refusals['not_accepted_or_nonfinite'] += 1
            continue
        state = states.get(record.get('state_id') or '')
        if state is None:
            refusals['no_canonical_protein_state'] += 1
            continue
        wildtype = wildtypes.get(state['protein'], {}).get('wt')
        if not wildtype:
            refusals['no_source_full_length_wildtype'] += 1
            continue
        channel = manifest[record['channel']]
        cohort.rows.append(PhenotypeRow(
            cohort=cohort.key, row_id=record['source_row_id'], background=record['channel'],
            unit=state['wt_sha256'], wildtype=wildtype,
            mutant_sequence=state['mutated_sequence'], position=int(state['position']),
            wt_aa=state['wt_aa'], mt_aa=state['mutant_aa'],
            label=_float(record['score'], record['source_row_id']), direction=1,
            label_unit=None, source_sha256=record['source_sha256']))
        if not channel.get('direction'):
            raise ValueError(f"{record['channel']}: the manifest declares no direction")
    cohort.refusals = {k: int(v) for k, v in refusals.items()}
    return cohort


#: One endpoint column per saturation-genome-editing score set, in preference
#: order. A score set exports several correlated timepoint columns of the same
#: library, which are not replicate measurements and are not independent rows, so
#: exactly one column may enter a ranking background. ``score`` is the score set's
#: own primary column; where a set publishes none, its continuous aggregate over
#: the measured timepoints is the declared substitute. A set offering neither is
#: refused rather than read under a per-timepoint column.
SGE_ENDPOINT_PREFERENCE = ('score', 'processed_adj_lfc_continuous')


def read_cellular_fitness(root: Path) -> Cohort:
    """MaveDB saturation-genome-editing endogenous growth fitness."""
    base = root / 'results/extensions' / MEASUREMENT / 'cellular-fitness/final'
    inventory = json.loads((base / 'state-inventory.json').read_text(encoding='utf-8'))['states']
    wildtypes = {record['gene']: record['sequence'] for record in inventory
                 if record['role'] == 'WT'}
    mutants = {(record['gene'], sha_text(record['sequence'])): record['sequence']
               for record in inventory if record['role'] != 'WT'}
    cohort = Cohort(
        key='cellular_fitness_sge', phenotype='endogenous cellular fitness (saturation genome editing)',
        endpoint=('score-set primary depletion score of an edited endogenous locus; larger is '
                  'more growth'),
        source='MaveDB saturation-genome-editing score sets, via cellular-fitness/final',
        quantitative_unit=None,
        notes=['Exactly one endpoint column enters each score set: ' +
               ' then '.join(SGE_ENDPOINT_PREFERENCE) + '. The other exported columns are '
               'correlated timepoints or alternative culture conditions of the same library, not '
               'replicate measurements, and are refused rather than stacked into one background.',
               'An edited coding SNV changes the transcript as well as the protein, so an '
               'endogenous growth label is not a protein-only phenotype; the retained RNA score '
               'is an explicitly label-assisted diagnostic and no protein-only mechanism is '
               'claimed for this endpoint.',
               'Depletion scores are normalised per score set on scales that are not shared, so '
               'the quantitative endpoint is refused for the cohort.'])
    refusals: Counter[str] = Counter()
    chosen: dict[str, str] = {}
    available: dict[str, set[str]] = {}
    for line in (base / 'observations.jsonl').open(encoding='utf-8'):
        record = json.loads(line)
        available.setdefault(record['score_set'], set()).add(record['endpoint'])
    for score_set, endpoints in sorted(available.items()):
        declared = next((name for name in SGE_ENDPOINT_PREFERENCE if name in endpoints), None)
        if declared is None:
            refusals[f'no_declared_endpoint_column_{score_set}'] = 1
            continue
        chosen[score_set] = declared
    cohort.notes.append('endpoint column chosen per score set: '
                        + json.dumps(dict(sorted(chosen.items())), sort_keys=True))
    for line in (base / 'observations.jsonl').open(encoding='utf-8'):
        record = json.loads(line)
        if record['endpoint'] != chosen.get(record['score_set']):
            refusals['endpoint_column_not_declared_primary'] += 1
            continue
        wildtype = wildtypes.get(record['gene'])
        mutant = mutants.get((record['gene'], record['mutant_sha256']))
        if wildtype is None or mutant is None:
            refusals['no_validated_protein_state'] += 1
            continue
        match = re.fullmatch(r'([A-Z])(\d+)([A-Z])', record.get('mutation') or '')
        if match is None:
            refusals['unparsed_protein_substitution'] += 1
            continue
        label = record.get('label')
        if label is None or not isinstance(label, (int, float)):
            refusals['nonfinite_or_absent_label'] += 1
            continue
        cohort.rows.append(PhenotypeRow(
            cohort=cohort.key, row_id=record['record'], background=record['score_set'],
            unit=sha_text(wildtype), wildtype=wildtype, mutant_sequence=mutant,
            position=int(match.group(2)), wt_aa=match.group(1), mt_aa=match.group(3),
            label=_float(label, record['record']), direction=1, label_unit=None,
            source_sha256=sha_text(record['score_set'] + record['record'])))
    cohort.refusals = {k: int(v) for k, v in refusals.items()}
    return cohort


#: The readers, in the order the qualification stage reports them.
READERS: dict[str, Callable[[Path], Cohort]] = {
    'activity_venus': read_activity,
    'binding_skempi': read_binding,
    'membrane_multistep': read_membrane,
    'cellular_fitness_sge': read_cellular_fitness,
}
