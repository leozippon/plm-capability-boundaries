#!/usr/bin/env python3
"""Development-only counterfactual response diagnostic against measured stability.

The frozen protocol ``data/r6_constraint_response_20261013/diagnostic_protocol.json``
fixes four development domains, eight evaluation domains, two native models, the
exact 256 measured single substitutions per development domain, and the two
evolutionary contextual comparators. This stage reads that protocol, builds
outcome-blind full-grid reference response arrays for every one of the twelve
domains from the completed missing-state campaign, and only then reads the
measured unfolding ddG of the four development domains.

What it computes, per development domain:

* primary: within-domain Spearman association between the native FULLSUM
  ``score_mut - score_WT`` response and the measured ddG, separately per model;
* contextual comparators: the same association for the full-natural
  independent-site profile log-score difference at pseudocount 0.5, and for the
  pinned PLMC positive unnormalized log-weight difference;
* secondary: mean measured ddG of the larger-score top 26 of 256, minus the
  whole-panel mean, with exact score ties broken by ascending sequence SHA-256.

What it never does: read an evaluation-domain phenotype, restrict or retune a
reference using an outcome, pool across domains, combine comparators, form an
interval, tune a threshold, or claim convergence beyond the recorded flags.

Reference construction is outcome-blind. The profile response is the difference
of ``log(count + 0.5)`` at the mutant and wild-type residues, where the canonical
weighted residue count is ``frequency x column_weight``; the site normalisation
cancels and no ad-hoc floor or softmax probability is introduced. The Potts
response is the direct single-substitution delta of ``fields + wild-type-background
couplings``, verified against ``PlmcParameters.score(mutant) - score(WT)`` on a
bounded deterministic sample. Every declared input and every written reference is
hashed; a missing, unknown, non-finite or digest-mismatched input refuses the run
rather than being salvaged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.amino_acids import AA20
from src.capability.core.io import sha256_file, write_json
from src.capability.stability.pairwise_stability import read_plmc

PROTOCOL_SCHEMA = 'r6_counterfactual_diagnostic_protocol_v1'
REPORT_SCHEMA = 'r6_counterfactual_diagnostic_v1'
REFERENCE_SCHEMA = 'r6_counterfactual_reference_v1'
MERGE_SCHEMA = 'stability_substitution_grid_merge_v1'
COHORT_SCHEMA = 'stability_singles_cohort_v1'
ALLOWLIST_SCHEMA = 'measured_selection_allowlist_v1'
PROVENANCE_SCHEMA = 'pairwise_reference_provenance_v1'
PROFILE_SCHEMA = 'pairwise_profile_features_v1'
PRIOR_SCHEMA = 'pairwise_sequence_baseline_v1'
PSEUDOCOUNT = 0.5
SINGLES_PER_DOMAIN = 256
TOP_K = 26
POTTS_VERIFICATION_SAMPLE = 32
DEFAULT_PROTOCOL = 'data/r6_constraint_response_20261013/diagnostic_protocol.json'
DEFAULT_POTTS_ROOT = 'archive/logs/R3/pairwise_homologs_20260924/baseline'
DEFAULT_OUT = 'results/R6/r6_counterfactual_diagnostic_20261013'
#: Completion artefact name fixed for the operations caller and its verifier.
REPORT_FILENAME = 'counterfactual_responses.json'

GRID_ARRAYS = ('score', 'response', 'position', 'mutant_index', 'state_sequence_sha256',
               'metadata')

#: The declared reduction and the response identity, restated from the merge stage
#: so a downstream reader cannot mistake the response for a probability.
RESPONSE_SEMANTICS = ('CFSCORE R_M,i(a) = S_native(s_i -> a) - S_native(s); raw native '
                      'packed residue-span summed log likelihood in nats; exactly zero at '
                      'the wild type; not a probability')
#: Reference semantics are declared once here and used by both the reference
#: archives and the report. The two references are evolutionary summaries, and
#: neither is the native FULLSUM quantity or a biological conditional probability.
PROFILE_REFERENCE_SEMANTICS = (
    'independent-site profile log-score difference log(count_mut + 0.5) - '
    'log(count_WT + 0.5) over canonical weighted residue counts frequency x '
    'column_weight; the site normalisation cancels; not a native FULLSUM quantity '
    'and not a probability')
POTTS_REFERENCE_SEMANTICS = (
    'pinned PLMC positive unnormalized log-weight difference sum h + sum J at the '
    'mutant minus the wild type; not a native FULLSUM quantity and not a probability')
REFERENCE_RESPONSE_SEMANTICS = (
    'reference arrays only: profile_response is ' + PROFILE_REFERENCE_SEMANTICS
    + '; potts_response is ' + POTTS_REFERENCE_SEMANTICS)
DISCLOSURE = ('the native response is the model-induced full-native-sum quantity; the '
              'profile and Potts reference responses are distinct summaries of shared evolutionary information, '
              'not native FULLSUM and not biological conditional probabilities; a '
              'token-segmentation difference is not a biological error')
POTTS_DISCLOSURE = ('positive unnormalized log weight sum h + sum J, selected on held-out '
                    'homolog sequence prediction, not phenotype; historical final_converged is '
                    'false for nine of the twelve fits and is retained as a flag')


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('ascii')).hexdigest()


def _resolve(base: Path, value) -> Path:
    """A declared path: absolute as given, otherwise relative to the staged root."""

    path = Path(value)
    return path if path.is_absolute() else base / path


def _read_json(path: Path, label: str, schema: str | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'{label} is absent: {path}')
    try:
        document = json.loads(path.read_text())
    except ValueError as error:
        raise ValueError(f'{label} at {path} is not readable JSON: {error}') from error
    if schema is not None and document.get('schema') != schema:
        raise ValueError(f'{label} at {path} has schema {document.get("schema")!r}, '
                         f'expected {schema!r}')
    return document


def verify_declared(path: Path, declared: str, label: str) -> str:
    """Re-hash one declared input and refuse any mismatch; never adopt silently."""

    if not path.is_file():
        raise FileNotFoundError(f'{label} is absent: {path}')
    observed = sha256_file(path)
    if observed != declared:
        raise ValueError(f'{label} at {path} has sha256 {observed}, declared {declared}')
    return observed


def bind_inputs(protocol: dict, root: Path) -> dict:
    """Resolve and digest-bind every input the frozen protocol declares."""

    bound = {}
    for key, entry in protocol['inputs'].items():
        path = _resolve(root, entry['path'])
        observed = verify_declared(path, entry['sha256'], f'protocol input {key}')
        bound[key] = {'path': str(path), 'declared_sha256': entry['sha256'],
                      'sha256': observed}
    return bound


# --------------------------------------------------------------------------- #
# Grids
# --------------------------------------------------------------------------- #

def verify_grid_mapping(wildtype: str, position, mutant_index, state_sha, score, response):
    """Bind every grid state to the wild type through the exact sequence hash.

    The mutation is reconstructed from the wild type, the 1-based position and the
    ``AA20`` mutant index, and its SHA-256 must equal the archived state hash. The
    state set must be exactly the 19 non-wild-type residues at every site, so an
    unknown, duplicated or misaligned state refuses rather than being reindexed.
    """

    length = len(wildtype)
    total = len(score)
    if total != 1 + 19 * length:
        raise ValueError(f'the grid carries {total} states, not {1 + 19 * length}')
    if not np.isfinite(score).all() or not np.isfinite(response).all():
        raise ValueError('the grid carries a non-finite native score or response')
    if not np.array_equal(response, score - score[0]):
        raise ValueError('the grid response is not exactly score - score_WT')
    if state_sha[0] != sha256_text(wildtype):
        raise ValueError('grid state zero is not the declared wild type')
    if int(position[0]) != 0 or int(mutant_index[0]) != -1:
        raise ValueError('grid state zero carries a substitution')
    seen = set()
    for index in range(1, total):
        site, code = int(position[index]), int(mutant_index[index])
        if not 1 <= site <= length or not 0 <= code < len(AA20):
            raise ValueError(f'grid state {index} carries an out-of-range site or mutant')
        mutant = AA20[code]
        if mutant == wildtype[site - 1]:
            raise ValueError(f'grid state {index} names the wild-type residue as its mutant')
        if _resolve_sha(wildtype, site, mutant) != str(state_sha[index]):
            raise ValueError(f'grid state {index} does not hash to its declared sequence')
        if (site, code) in seen:
            raise ValueError(f'grid state {index} repeats the substitution {site}{mutant}')
        seen.add((site, code))
    expected = {(site, code) for site in range(1, length + 1)
                for code in range(len(AA20)) if AA20[code] != wildtype[site - 1]}
    if seen != expected:
        raise ValueError('the grid is not exactly the 19 non-wild-type residues per site')


def _resolve_sha(wildtype: str, site_1based: int, mutant: str) -> str:
    return sha256_text(wildtype[:site_1based - 1] + mutant + wildtype[site_1based:])


def load_merged_grid(path: Path, arm: str, name: str, entry: dict, wildtype: str) -> dict:
    """Load and fully verify one merged full-grid archive against the manifest row."""

    observed = verify_declared(path, entry['sha256'], f'{arm}/{name} merged grid')
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in GRID_ARRAYS}
    except (OSError, ValueError) as error:
        raise ValueError(f'{arm}/{name}: unreadable merged grid: {error}') from error
    try:
        metadata = json.loads(str(arrays['metadata']))
    except ValueError as error:
        raise ValueError(f'{arm}/{name}: unreadable merged grid metadata: {error}') from error
    if (metadata.get('schema'), metadata.get('arm'), metadata.get('background')) != (
            MERGE_SCHEMA, arm, name):
        raise ValueError(f'{arm}/{name}: merged grid metadata does not name this grid')
    if metadata.get('full_states') != int(entry['full_states']):
        raise ValueError(f'{arm}/{name}: merged grid declares a different state count')
    if metadata.get('wildtype_sha256') != sha256_text(wildtype):
        raise ValueError(f'{arm}/{name}: merged grid wild type differs from the cohort')
    if entry.get('wildtype_sha256') != sha256_text(wildtype):
        raise ValueError(f'{arm}/{name}: manifest wild type differs from the cohort')
    score = np.asarray(arrays['score'], dtype=np.float64)
    response = np.asarray(arrays['response'], dtype=np.float64)
    if score.shape != (int(entry['full_states']),) or response.shape != score.shape:
        raise ValueError(f'{arm}/{name}: merged grid arrays have the wrong shape')
    verify_grid_mapping(wildtype, arrays['position'], arrays['mutant_index'],
                        arrays['state_sequence_sha256'], score, response)
    return {'path': str(path), 'sha256': observed, 'score': score, 'response': response,
            'position': np.asarray(arrays['position']), 'mutant_index': np.asarray(
                arrays['mutant_index']), 'state_sequence_sha256': np.asarray(
                    arrays['state_sequence_sha256']), 'group': entry.get('group'),
            'full_states': int(entry['full_states'])}


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #

def profile_single_response(frequencies, column_weight, wildtype: str, position,
                            mutant_index) -> tuple[np.ndarray, dict]:
    """Full-grid independent-site response from cached weighted column counts."""

    length = len(wildtype)
    frequencies = np.asarray(frequencies)
    column_weight = np.asarray(column_weight)
    if frequencies.shape != (length, len(AA20)) or column_weight.shape != (length,):
        raise ValueError('cached profile arrays do not match the wild-type length')
    if (not np.isfinite(frequencies).all() or not np.isfinite(column_weight).all()
            or np.any(frequencies < 0) or np.any(column_weight < 0)):
        raise ValueError('cached profile carries a non-finite or negative entry')
    positive = column_weight > 0
    if not positive.any():
        raise ValueError('cached profile has no supported column')
    if not np.allclose(frequencies[positive].sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError('cached profile frequencies are not normalised per supported column')
    counts = frequencies * column_weight[:, None]
    log_counts = np.log(counts + PSEUDOCOUNT)
    wild_codes = np.array([AA20.index(residue) for residue in wildtype])
    delta = np.zeros(len(position), dtype=np.float64)
    for index in range(1, len(position)):
        site, code = int(position[index]) - 1, int(mutant_index[index])
        delta[index] = log_counts[site, code] - log_counts[site, wild_codes[site]]
    reconstruction = {
        'canonical_weighted_counts': 'frequency x column_weight',
        'pseudocount': PSEUDOCOUNT,
        'supported_columns': int(positive.sum()),
        'frequencies_rows_normalised': True,
        'column_weight_nonnegative': True,
    }
    return delta, reconstruction


def potts_marginal_fields(params, wildtype: str) -> tuple[np.ndarray, list[int], dict]:
    """Marginal ``fields + wild-type-background couplings`` for every site and residue."""

    if params.target != wildtype:
        raise ValueError('the plmc target sequence is not the cohort wild type')
    length = len(wildtype)
    if not np.array_equal(np.asarray(params.positions), np.arange(1, length + 1)):
        raise ValueError('the plmc focus coordinates are not the ordered complete wild-type positions')
    if not np.isfinite(params.fields).all() or not np.isfinite(params.couplings).all():
        raise ValueError('plmc parameters carry a non-finite entry')
    codes = [params.alphabet.index(residue) for residue in wildtype]
    marginals = params.fields.astype(np.float64).copy()
    pair = 0
    for first in range(length):
        for second in range(first + 1, length):
            block = params.couplings[pair].astype(np.float64)
            marginals[first] += block[:, codes[second]]
            marginals[second] += block[codes[first], :]
            pair += 1
    return marginals, codes, {'alphabet': params.alphabet, 'positions_complete': True}


def potts_single_response(params, wildtype: str, position, mutant_index, *,
                          sample: int = POTTS_VERIFICATION_SAMPLE) -> tuple[np.ndarray, dict]:
    """Full-grid Potts single-substitution delta, verified against the brute score."""

    marginals, codes, binding = potts_marginal_fields(params, wildtype)
    delta = np.zeros(len(position), dtype=np.float64)
    for index in range(1, len(position)):
        site, code = int(position[index]) - 1, int(mutant_index[index])
        allele = params.alphabet.index(AA20[code])
        delta[index] = marginals[site, allele] - marginals[site, codes[site]]
    total = len(position)
    chosen = np.unique(np.linspace(0, total - 1, num=min(sample, total)).astype(int))
    max_difference, checked = 0.0, 0
    for index in chosen:
        index = int(index)
        if index == 0:
            sequence = wildtype
        else:
            sequence = _resolve_sequence(wildtype, int(position[index]), AA20[int(mutant_index[index])])
        difference = abs(delta[index] - (params.score(sequence) - params.score(wildtype)))
        max_difference = max(max_difference, difference)
        checked += 1
    if not np.isfinite(max_difference) or max_difference > 1e-4:
        raise ValueError(f'the direct Potts delta disagrees with PlmcParameters.score by '
                         f'{max_difference:.3e} over {checked} sampled states')
    binding.update({'brute_force_checked': checked, 'brute_force_max_absolute_difference':
                    float(max_difference)})
    return delta, binding


def _resolve_sequence(wildtype: str, site_1based: int, mutant: str) -> str:
    return wildtype[:site_1based - 1] + mutant + wildtype[site_1based:]


def load_profile_arrays(path: Path, name: str, wildtype: str) -> tuple[np.ndarray, np.ndarray]:
    """Cached profile frequencies and column weight for one background."""

    with np.load(path, allow_pickle=False) as data:
        try:
            frequencies = np.asarray(data[f'{name}|frequencies'])
            column_weight = np.asarray(data[f'{name}|column_weight'])
        except KeyError as error:
            raise ValueError(f'{path}: no cached profile arrays for {name}') from error
    if frequencies.shape != (len(wildtype), len(AA20)):
        raise ValueError(f'{path}: {name} frequencies are not ({len(wildtype)}, 20)')
    return frequencies, column_weight


def build_reference(domain: str, wildtype: str, grid: dict, frequencies, column_weight,
                    params, prior_record: dict) -> dict:
    """The two full-grid reference responses plus their provenance for one domain."""

    profile, reconstruction = profile_single_response(
        frequencies, column_weight, wildtype, grid['position'], grid['mutant_index'])
    potts, potts_binding = potts_single_response(
        params, wildtype, grid['position'], grid['mutant_index'])
    if not np.isfinite(profile).all() or not np.isfinite(potts).all():
        raise ValueError(f'{domain}: a reference response is non-finite')
    if profile[0] != 0.0 or potts[0] != 0.0:
        raise ValueError(f'{domain}: a reference response is not zero at the wild type')
    return {
        'domain': domain, 'wildtype_sha256': sha256_text(wildtype),
        'full_states': len(grid['position']),
        'profile_response': profile, 'potts_response': potts,
        'position': np.asarray(grid['position']), 'mutant_index': np.asarray(grid['mutant_index']),
        'state_sequence_sha256': np.asarray(grid['state_sequence_sha256']),
        'profile_reconstruction': reconstruction,
        'potts_binding': potts_binding,
        'potts_flags': {
            'final_converged': prior_record.get('final_converged'),
            'selected_at_grid_edge': prior_record.get('selected_at_grid_edge'),
            'retrieval_ceiling_reached': prior_record.get('retrieval_ceiling_reached'),
            'selected_lambda_e': prior_record.get('selected_lambda_e'),
        },
    }


def write_reference(path: Path, reference: dict) -> str:
    """Write one reference archive and return its digest; never overwrite silently."""

    if path.exists():
        raise FileExistsError(f'{path}: refusing to overwrite an existing reference archive')
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({
        'schema': REFERENCE_SCHEMA, 'domain': reference['domain'],
        'wildtype_sha256': reference['wildtype_sha256'], 'full_states': reference['full_states'],
        'profile_reconstruction': reference['profile_reconstruction'],
        'potts_binding': reference['potts_binding'], 'potts_flags': reference['potts_flags'],
        'response_semantics': REFERENCE_RESPONSE_SEMANTICS, 'disclosure': DISCLOSURE,
        'potts_disclosure': POTTS_DISCLOSURE,
    }, sort_keys=True)
    payload = {
        'profile_response': reference['profile_response'],
        'potts_response': reference['potts_response'],
        'position': reference['position'], 'mutant_index': reference['mutant_index'],
        'state_sequence_sha256': reference['state_sequence_sha256'],
        'metadata': metadata,
    }
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **payload)
    temp.replace(path)
    return sha256_file(path)


# --------------------------------------------------------------------------- #
# Measured support and selection
# --------------------------------------------------------------------------- #

def development_variants(cohort: dict, roles: dict) -> dict:
    """The 256 frozen measured singles for each development domain, and no other.

    The eight evaluation domains are never read, so no evaluation phenotype can
    leak into this stage. Variants are ordered by ascending sequence SHA-256.
    """

    development = list(roles['development'])
    evaluation = list(roles['evaluation'])
    if set(development) & set(evaluation):
        raise ValueError('a domain is assigned to development and evaluation')
    backgrounds = {row['name']: row for row in cohort['backgrounds']}
    if len(backgrounds) != len(cohort['backgrounds']):
        raise ValueError('the cohort repeats a background')
    out = {}
    for name in development:
        row = backgrounds.get(name)
        if row is None:
            raise ValueError(f'{name}: development domain is absent from the cohort')
        if row.get('group') is None:
            raise ValueError(f'{name}: development domain carries no group')
        wildtype = row.get('wildtype')
        if not wildtype or any(residue not in AA20 for residue in wildtype):
            raise ValueError(f'{name}: development wild type is not canonical residues')
        variants = row.get('variants') or []
        if len(variants) != SINGLES_PER_DOMAIN:
            raise ValueError(f'{name}: carries {len(variants)} measured singles, not '
                             f'{SINGLES_PER_DOMAIN}')
        records = []
        for variant in variants:
            sequence, site, mutant = variant.get('sequence'), variant.get('position'), variant.get('mutant')
            if not sequence or len(sequence) != len(wildtype):
                raise ValueError(f'{name}: a measured variant does not match the wild-type length')
            changed = [i for i in range(len(wildtype)) if sequence[i] != wildtype[i]]
            if len(changed) != 1 or changed[0] + 1 != site or sequence[changed[0]] != mutant:
                raise ValueError(f'{name}: a measured variant is not the single substitution it declares')
            ddg = float(variant['ddg'])
            if not np.isfinite(ddg):
                raise ValueError(f'{name}: a measured ddG is not finite')
            records.append({'sha256': sha256_text(sequence), 'ddg': ddg,
                            'position': int(site), 'mutant': mutant, 'sequence': sequence})
        if len({record['sha256'] for record in records}) != SINGLES_PER_DOMAIN:
            raise ValueError(f'{name}: the measured singles are not distinct sequences')
        if sha256_text(wildtype) in {record['sha256'] for record in records}:
            raise ValueError(f'{name}: the wild type is present as a measured mutant row')
        records.sort(key=lambda record: record['sha256'])
        out[name] = {'group': row['group'], 'wildtype': wildtype,
                     'wildtype_sha256': sha256_text(wildtype), 'records': records}
    return out


def top_selection(score, shas, k: int = TOP_K) -> np.ndarray:
    """Indices of the ``k`` larger scores; exact ties break by ascending SHA-256."""

    score = np.asarray(score, dtype=np.float64)
    shas = np.asarray(shas)
    if score.ndim != 1 or shas.shape != score.shape or k > len(score):
        raise ValueError('selection needs aligned one-dimensional scores and hashes')
    if not np.isfinite(score).all():
        raise ValueError('selection received a non-finite score')
    order = np.lexsort((shas, -score))
    return order[:k]


def selection_digest(shas) -> str:
    return sha256_text('\n'.join(sorted(str(value) for value in shas)))


def spearman_association(response, ddg) -> dict:
    """Within-domain Spearman association; a degenerate or non-finite value refuses."""

    response = np.asarray(response, dtype=np.float64)
    ddg = np.asarray(ddg, dtype=np.float64)
    if len(response) != len(ddg) or len(response) < 3:
        raise ValueError('a Spearman association needs aligned vectors of at least three values')
    if not np.isfinite(response).all() or not np.isfinite(ddg).all():
        raise ValueError('a Spearman association received a non-finite value')
    if np.ptp(response) == 0 or np.ptp(ddg) == 0:
        raise ValueError('a Spearman association received a constant vector')
    rho = float(cast(float, spearmanr(response, ddg)[0]))
    if not np.isfinite(rho):
        raise ValueError('the Spearman association is not finite')
    return {'rho': rho, 'n': int(len(response))}


def secondary_selection(response, ddg, shas) -> dict:
    """Mean measured ddG of the larger-score top 26 minus the whole-panel mean."""

    ddg = np.asarray(ddg, dtype=np.float64)
    shas = np.asarray(shas)
    chosen = top_selection(response, shas)
    panel_mean = float(ddg.mean())
    selected_mean = float(ddg[chosen].mean())
    return {'k': int(TOP_K), 'panel_n': int(len(ddg)),
            'selected_mean_ddg': selected_mean, 'panel_mean_ddg': panel_mean,
            'selected_minus_panel': selected_mean - panel_mean,
            'tie_rule': 'larger score, exact ties by ascending sequence SHA-256',
            'selected_sha256_digest': selection_digest(shas[chosen])}


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def _prior_by_name(prior: dict) -> dict:
    records = {row['name']: row for row in prior['backgrounds']}
    if len(records) != len(prior['backgrounds']):
        raise ValueError('the prior Potts manifest repeats a background')
    return records


def _potts_params_path(potts_root: Path, name: str, record: dict) -> Path:
    declared = record.get('params') or {}
    recorded = Path(declared.get('path', ''))
    if (recorded.name != 'selected.params' or recorded.parent.name != name
            or recorded.parent.parent.name != 'work'):
        raise ValueError(f'{name}: the recorded parameter path is not work/{name}/selected.params')
    path = potts_root / 'work' / name / 'selected.params'
    if not path.is_file():
        raise FileNotFoundError(f'{name}: relocated plmc parameters are absent: {path}')
    if declared.get('sha256') is not None and sha256_file(path) != declared['sha256']:
        raise ValueError(f'{name}: relocated plmc parameters differ from the declared digest')
    if declared.get('bytes') is not None and path.stat().st_size != int(declared['bytes']):
        raise ValueError(f'{name}: relocated plmc parameters differ from the declared size')
    return path


def run(protocol_path: Path, root: Path, potts_root: Path, out: Path) -> dict:
    protocol = _read_json(protocol_path, 'frozen protocol', PROTOCOL_SCHEMA)
    if protocol.get('frozen_date') is None or protocol.get('decision_boundary') is None:
        raise ValueError('the protocol is not the frozen counterfactual diagnostic protocol')
    inputs = bind_inputs(protocol, root)

    development, evaluation = list(protocol['roles']['development']), list(protocol['roles']['evaluation'])
    models = list(protocol['models'])
    all_domains = development + evaluation
    if len(all_domains) != 12 or len(set(all_domains)) != 12:
        raise ValueError('the protocol does not fix twelve distinct domains')

    cohort = _read_json(Path(inputs['cohort']['path']), 'frozen cohort', COHORT_SCHEMA)
    allowlist = _read_json(Path(inputs['allowlist']['path']), 'reviewed allowlist', ALLOWLIST_SCHEMA)
    provenance = _read_json(Path(inputs['reference_provenance']['path']),
                            'reference provenance', PROVENANCE_SCHEMA)
    prior = _read_json(Path(inputs['prior_potts_manifest']['path']), 'prior Potts manifest',
                       PRIOR_SCHEMA)

    allowlist_names = {row['name'] for row in allowlist['backgrounds']}
    if allowlist_names != set(all_domains):
        raise ValueError('the reviewed allowlist does not match the twelve protocol domains')
    assets = provenance['assets']
    for key, asset in (('cohort', 'cohort'), ('profile_features', 'profile_features'),
                       ('prior_potts_manifest', 'prior_baseline'), ('allowlist', 'allowlist')):
        if assets.get(asset, {}).get('sha256') != inputs[key]['sha256']:
            raise ValueError(f'the reference provenance disagrees about the {asset} digest')

    merged_manifest = _read_json(Path(inputs['merged_manifest']['path']),
                                 'merged grid manifest', MERGE_SCHEMA)
    if merged_manifest.get('status') != 'complete':
        raise ValueError('the merged grid manifest is not complete')
    if sorted(merged_manifest['arms']) != sorted(models):
        raise ValueError('the merged grid manifest does not carry exactly the protocol models')

    prior_by_name = _prior_by_name(prior)
    profile_path = Path(inputs['profile_features']['path'])
    merged_dir = Path(inputs['merged_manifest']['path']).parent

    cohort_by_name = {row['name']: row for row in cohort['backgrounds']}
    references, native, grid_lookup = {}, {}, {}
    for name in all_domains:
        cohort_row = cohort_by_name.get(name)
        if cohort_row is None:
            raise ValueError(f'{name}: protocol domain is absent from the cohort')
        wildtype = cohort_row['wildtype']
        if cohort_row.get('wildtype_sha256') not in (None, sha256_text(wildtype)):
            raise ValueError(f'{name}: the cohort wild-type digest does not match its sequence')
        frequencies, column_weight = load_profile_arrays(profile_path, name, wildtype)
        prior_record = prior_by_name.get(name)
        if prior_record is None or prior_record.get('status') != 'fitted':
            raise ValueError(f'{name}: no fitted pinned PLMC reference')
        params_path = _potts_params_path(potts_root, name, prior_record)
        params = read_plmc(params_path)
        grids = {}
        for arm in models:
            entry = merged_manifest['arms'][arm]['backgrounds'].get(name)
            if entry is None:
                raise ValueError(f'{arm}/{name}: the merged manifest has no grid row')
            grids[arm] = load_merged_grid(merged_dir / arm / f'{name}.npz', arm, name, entry,
                                          wildtype)
        first = grids[models[0]]
        for arm in models[1:]:
            for key in ('position', 'mutant_index', 'state_sequence_sha256'):
                if not np.array_equal(first[key], grids[arm][key]):
                    raise ValueError(f'{arm}/{name}: models disagree about the grid mapping')
        references[name] = build_reference(name, wildtype, first, frequencies, column_weight,
                                           params, prior_record)
        references[name]['potts_params'] = {'path': str(params_path),
                                            'sha256': sha256_file(params_path)}
        native[name] = {arm: grids[arm]['response'] for arm in models}
        grid_lookup[name] = {str(value): index
                             for index, value in enumerate(first['state_sequence_sha256'])}

    # All twelve outcome-blind references are complete before any measured label is read.
    measured = development_variants(cohort, protocol['roles'])
    for name in measured:
        if references[name]['wildtype_sha256'] != measured[name]['wildtype_sha256']:
            raise ValueError(f'{name}: the reference wild type differs from the measured panel')
        for record in measured[name]['records']:
            if record['sha256'] not in grid_lookup[name]:
                raise ValueError(f'{name}: measured single {record["sha256"]} is outside the grid')

    out.mkdir(parents=True, exist_ok=True)
    if (out / REPORT_FILENAME).exists():
        raise FileExistsError(f'{out}/{REPORT_FILENAME}: refusing to overwrite an existing report')

    reference_outputs = {}
    for name in all_domains:
        path = out / 'references' / f'{name}.npz'
        digest = write_reference(path, references[name])
        reference_outputs[name] = {'path': str(path.relative_to(out)), 'sha256': digest,
                                   'full_states': references[name]['full_states']}

    primary, profile_comparator, potts_comparator, secondary = {}, {}, {}, {}
    measured_identity = {name: {'wildtype_sha256': panel['wildtype_sha256'],
                                'group': panel['group'], 'singles': len(panel['records']),
                                'sequence_sha256_digest': selection_digest(
                                    record['sha256'] for record in panel['records'])}
                         for name, panel in measured.items()}
    for name in development:
        records = measured[name]['records']
        index = np.array([grid_lookup[name][record['sha256']] for record in records])
        shas = np.array([record['sha256'] for record in records])
        ddg = np.array([record['ddg'] for record in records], dtype=np.float64)
        primary[name] = {arm: spearman_association(native[name][arm][index], ddg)
                         for arm in models}
        profile = references[name]['profile_response'][index]
        potts = references[name]['potts_response'][index]
        profile_comparator[name] = spearman_association(profile, ddg)
        potts_comparator[name] = spearman_association(potts, ddg)
        secondary[name] = {arm: secondary_selection(native[name][arm][index], ddg, shas)
                           for arm in models}
        secondary[name]['independent_site_profile'] = secondary_selection(profile, ddg, shas)
        secondary[name]['pinned_plmc'] = secondary_selection(potts, ddg, shas)

    flags = {name: references[name]['potts_flags'] for name in all_domains}
    report = {
        'schema': REPORT_SCHEMA,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'producer': {'script': str(Path(__file__).resolve()), 'sha256': sha256_file(Path(__file__))},
        'protocol': {'path': str(protocol_path), 'sha256': sha256_file(protocol_path),
                     'schema': PROTOCOL_SCHEMA, 'frozen_date': protocol['frozen_date']},
        'inputs': inputs,
        'roles': {
            'development': development, 'evaluation': evaluation, 'models': models,
            'development_screen_units': protocol['roles']['development_screen_units'],
            'assignment_rule': protocol['roles']['assignment_rule'],
            'limitations': protocol['roles']['limitations'],
        },
        'response_semantics': RESPONSE_SEMANTICS,
        'reference_response_semantics': REFERENCE_RESPONSE_SEMANTICS,
        'disclosure': DISCLOSURE,
        'potts_disclosure': POTTS_DISCLOSURE,
        'measurement': protocol['measurement'],
        'reference_limits': protocol['reference_limits'],
        'decision_boundary': protocol['decision_boundary'],
        'identity': {
            'development_domains': len(development), 'evaluation_domains': len(evaluation),
            'models': len(models), 'singles_per_domain': SINGLES_PER_DOMAIN,
            'development_singles_total': len(development) * SINGLES_PER_DOMAIN,
            'model_domain_records': len(development) * len(models),
            'comparator_records': 2 * len(development),
            'secondary_records': len(development) * (len(models) + 2),
            'measured_panel': measured_identity,
        },
        'references': {name: {'output': reference_outputs[name],
                              'profile_reconstruction': references[name]['profile_reconstruction'],
                              'potts_params': references[name]['potts_params'],
                              'potts_binding': references[name]['potts_binding'],
                              'potts_flags': references[name]['potts_flags']}
                       for name in all_domains},
        'reference_flags': {
            'final_converged': sorted(name for name in all_domains
                                      if flags[name]['final_converged'] is True),
            'final_converged_false': sorted(name for name in all_domains
                                            if flags[name]['final_converged'] is False),
            'selected_at_grid_edge': sorted(name for name in all_domains
                                            if flags[name]['selected_at_grid_edge']),
            'retrieval_ceiling_reached': sorted(name for name in all_domains
                                                if flags[name]['retrieval_ceiling_reached']),
            'reading': protocol['reference_limits']['convergence'],
        },
        'primary': primary,
        'contextual_comparators': {'independent_site_profile': profile_comparator,
                                   'pinned_plmc': potts_comparator},
        'secondary': secondary,
        'leakage_guard': {
            'evaluation_phenotypes_computed': False,
            'evaluation_domains_read_for_measured_outcome': [],
            'evidence': 'measured ddG is read only for the protocol development roles',
        },
        'uncertainty': ('descriptive per-domain values only; no confidence interval, pooling, '
                        'comparator combination, adaptive orientation or threshold selection '
                        'was formed'),
        'outputs': reference_outputs,
    }
    write_json(out / REPORT_FILENAME, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu',), default='cpu',
                        help='CPU-only execution; explicit resource contract for the campaign queue')
    parser.add_argument('--protocol', type=Path, default=None)
    parser.add_argument('--root', type=Path, default=ROOT,
                        help='staged repository root the repo-relative protocol paths resolve against')
    parser.add_argument('--potts-root', type=Path, default=None,
                        help='explicit root of the relocated selected.params files')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    protocol_path = _resolve(root, args.protocol or DEFAULT_PROTOCOL)
    potts_root = _resolve(root, args.potts_root or DEFAULT_POTTS_ROOT)
    out = _resolve(root, args.out or DEFAULT_OUT)
    report = run(protocol_path, root, potts_root, out)
    print(json.dumps({'report': str(out / REPORT_FILENAME),
                      'report_sha256': sha256_file(out / REPORT_FILENAME),
                      'development': report['roles']['development'],
                      'models': report['roles']['models'],
                      'primary': report['primary']}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())