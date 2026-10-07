"""CPU-only retained-token response analyses, without inference or tokenization.

Coordinates are zero-based sequence indices; spans are half-open packed-token
indices. State metadata must be exported by the original packer, not guessed
from counts. Structures supply every mapped site, not selected epistasis pairs.
"""
from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import re

import numpy as np
from scipy.stats import rankdata

from ..core.amino_acids import AA20
from ..position.position_terms import alignment, residue_bounds, partition_masks, PARTITION_RESIDUAL_RELATIVE
from ..context.profile_increment import correlation, standardized_rank
from ..context.local_context import fold_membership
from ..readouts.readout_analysis import nested_predict

DISTANCE_EDGES = (8, 16, 32, 64, 128)
DISTANCE_NAMES = ('3-8', '9-16', '17-32', '33-64', '65-128', '129+')
BIN_NAMES = ('1-8', '9-32', '33-128', '129+')


def simultaneous_bands(values, *, draws=2000, seed=20261006):
    """Use the existing numerical helper, not a second bootstrap implementation."""
    path = Path(__file__).resolve().parents[3] / 'scripts/capability/position/position_simultaneous.py'
    spec = importlib.util.spec_from_file_location('_response_bands', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('existing simultaneous-band helper is unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.bands(values, draws=draws, seed=seed)


def _integers(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in 'iu' or array.ndim != 1:
        raise ValueError(f'{name} must be a one-dimensional integer array')
    return array.astype(np.int64)


class RetainedResponses:
    """Adapter for original ragged position archives plus exact packing sidecar.

    ``states`` is an ordered list with sequence, ids, span, counts, offset.
    State zero is WT; subsequent states follow ``mutants`` exactly. ``identity``
    must contain wildtype, mutants and sequences, in that same order. Native
    likelihood is the original signed delta, never a rank or ``raw_M`` column.
    No tokenizer/model is loaded to recover missing metadata.
    """

    def __init__(self, data, states: list[dict], identity: dict):
        required = ('position_nats', 'position_offsets', 'position_residue_counts',
                    'position_residue_offsets', 'position_sum_check_nats',
                    'mutants', 'likelihood', 'wt_likelihood')
        if any(key not in data for key in required):
            raise ValueError('missing original retained NLL arrays or native identities')
        self.terms = np.asarray(data['position_nats'])
        self.offsets = _integers(data['position_offsets'], 'position_offsets')
        self.counts = _integers(data['position_residue_counts'], 'position_residue_counts')
        self.residue_offsets = _integers(data['position_residue_offsets'], 'position_residue_offsets')
        if (self.terms.dtype != np.float32 or self.terms.ndim != 1
                or not np.isfinite(self.terms).all() or np.any(self.terms < 0)
                or self.counts.shape != self.terms.shape or np.any(self.counts < 0)
                or len(self.offsets) != len(self.residue_offsets) + 1
                or not len(self.residue_offsets) or self.offsets[0] != 0
                or self.offsets[-1] != len(self.terms) or np.any(np.diff(self.offsets) <= 0)
                or np.any(self.residue_offsets < 0)):
            raise ValueError('invalid float32 retained NLL arrays or ragged mapping')
        if float(data['position_sum_check_nats']) != 0:
            raise ValueError('original retention check is not exactly zero')
        self.mutants = np.asarray(data['mutants']).tolist()
        self.native = np.asarray(data['likelihood'], dtype=np.float64)
        self.wt_native = float(data['wt_likelihood'])
        self.states = states
        if (self.mutants != identity['mutants'] or len(set(self.mutants)) != len(self.mutants)
                or len(states) != len(self.mutants) + 1 or len(states) != len(self.residue_offsets)
                or len(identity['sequences']) != len(self.mutants)
                or self.native.shape != (len(self.mutants),) or not np.isfinite(self.native).all()
                or not np.isfinite(self.wt_native)
                or [s['sequence'] for s in states] != [identity['wildtype'], *identity['sequences']]):
            raise ValueError('WT/mutant ordered identity mismatch')
        wild = identity['wildtype']
        if not wild or set(wild) - set(AA20):
            raise ValueError('invalid WT sequence')
        self.sites = []
        for mutation, sequence in zip(self.mutants, identity['sequences']):
            changed = list(wild)
            sites = []
            for token in mutation.split(':'):
                match = re.fullmatch(r'([A-Z])(\d+)([A-Z])', token)
                if match is None:
                    raise ValueError('invalid archived substitution identity')
                before, number, after = match.groups()
                site = int(number)-1
                if (not 0 <= site < len(wild) or wild[site] != before or after not in AA20
                        or after == before or site in sites):
                    raise ValueError('mutation disagrees with exact WT/mutant sequence')
                changed[site] = after
                sites.append(site)
            if sequence != ''.join(changed):
                raise ValueError('mutation disagrees with exact WT/mutant sequence')
            self.sites.append(sites)
        for index, state in enumerate(states):
            ids = _integers(state['ids'], 'ids')
            span = _integers(state['span'], 'span')
            low, high = self.offsets[index:index+2]
            if (np.any(ids < 0) or len(span) != 2 or not 1 <= span[0] < span[1] <= len(ids)
                    or span[1]-span[0] != high-low
                    or not np.array_equal(_integers(state['counts'], 'counts'), self.counts[low:high])
                    or state['offset'] != self.residue_offsets[index]
                    or state['offset'] + int(self.counts[low:high].sum()) != len(state['sequence'])):
                raise ValueError('packing sidecar disagrees with retained residue mapping')
        # Absolute state totals also close: a wrong WT scalar must not cancel.
        totals = [self.state_terms(i).sum(dtype=np.float64) for i in range(len(states))]
        for total, native in zip(totals, [-self.wt_native, *(-self.wt_native-self.native)]):
            if abs(total-native) > PARTITION_RESIDUAL_RELATIVE * abs(total):
                raise ValueError('native state NLL closure failed')
        # Validate every native delta before any biological support exclusion.
        for index in range(len(self.mutants)):
            residual = self.native[index]-(totals[0]-totals[index+1])
            if abs(residual) > PARTITION_RESIDUAL_RELATIVE*(totals[0]+totals[index+1]):
                raise ValueError('native mutation closure failed')
        self.selected_indices = []
        self.exclusions = []
        for index, sites in enumerate(self.sites):
            reason = None
            if len(sites) != 1:
                reason = 'multiple_substitutions'
            elif not all(partition_masks(s['counts'],s['offset'],sites[0])['own'].any()
                         for s in (states[0],states[index+1])):
                reason = 'unscored_mutation'
            elif not alignment(states[0],states[index+1],sites[0])['aligned']:
                reason = 'grid_misalignment'
            if reason is None:
                self.selected_indices.append(index)
            else:
                self.exclusions.append(dict(original_variant_index=index,original_state_index=index+1,
                                            mutation=self.mutants[index],reason=reason))
        self.selected_state_indices = [0,*[i+1 for i in self.selected_indices]]

    def project(self, row: dict) -> dict:
        """One indexed view for identities, baseline and labels; no ragged copying."""
        if row['mutants'] != self.mutants or row['sequences'] != [s['sequence'] for s in self.states[1:]]:
            raise ValueError('projection row identities differ from validated archive')
        result = dict(row)
        index = self.selected_indices
        for key in ('mutants','sequences','B','measured'):
            if key not in row:
                continue
            if len(row[key]) != len(self.mutants):
                raise ValueError(f'full archive projection has unaligned {key}')
            result[key] = [row[key][i] for i in index]
        result['responses'] = [self.response(i) for i in index]
        result['original_variant_indices'] = list(index)
        result['original_state_indices'] = list(self.selected_state_indices)
        result['exclusions'] = self.exclusions
        return result

    def state_terms(self, index):
        low, high = self.offsets[index:index+2]
        return self.terms[low:high]

    def response(self, index: int) -> dict:
        """Native-signed tokens, exclusive receiver census, and exact bin sums."""
        if index not in self.selected_indices:
            raise ValueError('variant is biologically excluded; use selected_indices')
        site = self.sites[index][0]
        wild = self.states[0]
        wt = self.state_terms(0).astype(np.float64)
        mt = self.state_terms(index+1).astype(np.float64)
        delta = wt - mt
        starts, ends = residue_bounds(wild['counts'], wild['offset'])
        widths = ends-starts
        masks = partition_masks(wild['counts'], wild['offset'], site)
        own = masks['own']
        primary = (widths == 1) & (starts > site) & ~own
        special = widths == 0
        multi = (widths > 1) & ~own
        upstream = (widths == 1) & (starts < site) & ~own
        bins = np.zeros(4)
        support = np.zeros(4, dtype=int)
        receivers = []
        for token in np.flatnonzero(primary):
            distance = int(starts[token]-site)
            bucket = int(np.searchsorted((8, 32, 128), distance, side='left'))
            bins[bucket] += delta[token]
            support[bucket] += 1
            j = int(starts[token])
            receivers.append(dict(j=j, distance=distance, identity=wild['sequence'][j],
                                  response=float(delta[token]), wt_nll=float(wt[token])))
        own_sum = float(delta[own].sum())
        separate = dict(special=float(delta[special & ~own].sum()),
                        multiresidue=float(delta[multi].sum()), upstream=float(delta[upstream].sum()))
        # Native reduction residual remains nuisance, never allocated to a bin.
        remainder = float(self.native[index]-own_sum-bins.sum())
        residual = float(self.native[index]-delta.sum())
        bound = PARTITION_RESIDUAL_RELATIVE * float(wt.sum()+mt.sum())
        if abs(residual) > bound:
            raise ValueError('native mutation closure failed')
        return dict(mutation=self.mutants[index], original_variant_index=index,
                    original_state_index=index+1, i=site, native=float(self.native[index]),
                    own=own_sum, bins=bins.tolist(), bin_support=support.tolist(),
                    remainder=remainder, separate=separate,
                    token_census=dict(primary=int(primary.sum()), own=int(own.sum()),
                                      special=int((special & ~own).sum()), multiresidue=int(multi.sum()),
                                      upstream=int(upstream.sum())), closure_nats=residual,
                    closure_bound_nats=bound, receivers=receivers)


def mapped_pairs(response: dict, sites: list[dict], wildtype: str) -> list[dict]:
    """Explicit structural contract: j, identity, atom, xyz, rsa (null allowed).

    The caller supplies *all* exact mapped residues for the protein. Cbeta is
    required except Gly Calpha. No epistasis-selected pair list is accepted.
    Missing sites stay out of support; missing RSA is counted, not imputed.
    """
    mapped = {}
    for site in sites:
        j = site['j']
        if (isinstance(j, bool) or not isinstance(j, (int, np.integer))
                or not 0 <= j < len(wildtype) or j in mapped
                or site['identity'] != wildtype[j]
                or site['atom'] != ('CA' if wildtype[j] == 'G' else 'CB')):
            raise ValueError('invalid/duplicate exact structure residue mapping')
        xyz = np.asarray(site['xyz'], float)
        rsa = site['rsa']
        if xyz.shape != (3,) or not np.isfinite(xyz).all() or (rsa is not None and (not np.isfinite(rsa) or rsa < 0)):
            raise ValueError('invalid structural coordinate or RSA')
        mapped[j] = site
    anchor = mapped.get(response['i'])
    if anchor is None:
        return []
    result = []
    for receiver in response['receivers']:
        j = receiver['j']
        if j not in mapped or receiver['distance'] <= 2:
            continue
        distance = float(np.linalg.norm(np.asarray(mapped[j]['xyz'])-np.asarray(anchor['xyz'])))
        result.append(dict(receiver, contact=bool(distance < 8), structure_distance=distance,
                           rsa=mapped[j]['rsa'], stratum=DISTANCE_NAMES[int(np.searchsorted(DISTANCE_EDGES, receiver['distance'], side='left'))]))
    return result


def structure_sites(site_rows: list[dict], *, assay: str, family: str, wildtype: str,
                    coverage: dict) -> tuple[list[dict], dict]:
    """Bridge the structural developer's all-WT-site table, preserving exclusions.

    ``wt_position`` is one-based in that API and converted once here. Every WT
    site, including excluded coordinates, must be present exactly once. Only
    exact mapping rows with actual CB/GlyCA coordinates enter pair eligibility.
    """
    from .structure import EXPERIMENTAL_METHODS
    assignments = [r for r in coverage['assays'] if r['assay']==assay]
    if len(assignments) != 1 or assignments[0]['cluster'] != family:
        raise ValueError('structure coverage assay/family binding mismatch')
    assigned = assignments[0]
    sources = [s for s in coverage['sources'] if s.get('sha256')==assigned.get('source_sha256')
               and s.get('path')==assigned.get('source_path')]
    if assigned['status']=='admitted':
        if (len(sources) != 1 or sources[0]['status'] != 'parsed'
                or not re.fullmatch(r'[0-9a-f]{64}',str(assigned.get('source_sha256','')))
                or not isinstance(assigned.get('source_path'),str) or not assigned['source_path'].strip()
                or assigned.get('method') not in EXPERIMENTAL_METHODS
                or sources[0].get('method') != assigned['method']):
            raise ValueError('admitted structure lacks bound experimental source provenance')
    selected = [r for r in site_rows if r['assay_id']==assay]
    digest = hashlib.sha256(wildtype.encode()).hexdigest()
    if len(selected) != len(wildtype):
        raise ValueError('structure table must contain every WT site including exclusions')
    seen, usable, excluded = set(), [], []
    for row in selected:
        position = row['wt_position']
        if not isinstance(position,int) or isinstance(position,bool):
            raise ValueError('structure WT position must be an integer')
        j = position-1
        if (j in seen or not 0 <= j < len(wildtype) or row['residue'] != wildtype[j]
                or row.get('wt_sequence', row.get('wildtype')) != wildtype
                or row.get('wt_sha256', row.get('wildtype_sha256')) != digest or row['cluster'] != family):
            raise ValueError('structure table exact WT/family identity mismatch')
        seen.add(j)
        coordinates = row.get('contact_coordinates', row.get('contact_coordinate'))
        atom = 'CA' if wildtype[j]=='G' else 'CB'
        if row.get('contact_atom') != atom:
            raise ValueError('declared structure contact atom is not CB/GlyCA')
        if row['status'] != assigned['status']:
            raise ValueError('site status disagrees with paired structure coverage')
        if row['status'] != 'admitted':
            if coordinates is not None or row['contact_atom_present'] or row['coordinate_present']:
                raise ValueError('excluded structure carries purported usable coordinates')
            excluded.append(dict(j=j,status=row['status'],mapping_kind=row['mapping_kind']))
            continue
        for key in ('source_sha256','source_path','entry_id','entity_id','chain','auth_chain','model','mapping_kind'):
            if row.get(key) != assigned.get(key):
                raise ValueError(f'structure site/coverage provenance mismatch: {key}')
        if row.get('method',assigned['method']) != assigned['method']:
            raise ValueError('structure site method disagrees with experimental coverage')
        if row['mapping_kind'] not in ('full_entity','full_wt_fragment'):
            raise ValueError('non-exact structural mapping cannot enter response pairs')
        if row['contact_atom_present']:
            if not row['coordinate_present'] or coordinates is None:
                raise ValueError('structure contact atom has inconsistent coordinate flags')
            xyz = np.asarray(coordinates,float)
            rsa = row['rsa']
            if xyz.shape != (3,) or not np.isfinite(xyz).all() or (rsa is not None and (not np.isfinite(rsa) or rsa < 0)):
                raise ValueError('invalid structural coordinates/RSA')
            usable.append(dict(j=j,identity=wildtype[j],atom=row['contact_atom'],xyz=coordinates,rsa=rsa))
        else:
            if coordinates is not None:
                raise ValueError('absent contact atom carries coordinates')
            excluded.append(dict(j=j,status=row['status'],mapping_kind=row['mapping_kind']))
    return usable, dict(excluded=excluded,source_sha256=sorted({r['source_sha256'] for r in selected if r['source_sha256'] is not None}),
                        source_paths=sorted({r['source_path'] for r in selected if r['source_path'] is not None}))


def _geometry_support(pairs, *, identity=False, complete_rsa=False):
    groups = {}
    missing = 0
    for pair in pairs:
        if complete_rsa and pair['rsa'] is None:
            missing += 1
            continue
        key = (pair['stratum'],pair['identity']) if identity else (pair['stratum'],)
        groups.setdefault(key,[]).append(pair)
    census = []
    for key, rows in sorted(groups.items()):
        c = [r for r in rows if r['contact']]
        n = [r for r in rows if not r['contact']]
        matched = bool(c and n)
        census.append(dict(stratum=list(key),contacts=len(c),noncontacts=len(n),matched=matched,
                           distance_imbalance=float(np.mean([r['distance'] for r in c])-np.mean([r['distance'] for r in n])) if matched else None))
    matched = [s for s in census if s['matched']]
    return dict(strata=census,matched_strata=len(matched),
                matched_receivers=sum(s['contacts']+s['noncontacts'] for s in matched),
                missing_rsa_excluded=missing,
                distance_imbalance=float(np.mean([s['distance_imbalance'] for s in matched])) if matched else None)


def prepare_structure_pairs(proteins: list[dict]) -> dict:
    """Executable response/phenotype-blind candidate census before NLL recovery.

    Protein rows require protein, assay, family, wildtype, sites, mutations.
    This is structural eligibility only, never a token-response support claim.
    All downstream mapped residues are emitted, with matching membership and
    deterministic geometry-only weights; unmatched strata remain visible.
    """
    units = []
    seen = set()
    assay_family = {}
    for protein in proteins:
        wild = protein['wildtype']
        if not wild or set(wild)-set(AA20):
            raise ValueError('invalid structure WT sequence')
        assay, family = protein['assay'], protein['family']
        if assay in assay_family and assay_family[assay] != family:
            raise ValueError('assay crosses families')
        assay_family[assay] = family
        for mutation in protein['mutations']:
            match = re.fullmatch(r'([A-Z])(\d+)([A-Z])', mutation)
            if match is None:
                raise ValueError('pair preparation requires strict single substitutions')
            before, position, after = match.groups()
            i = int(position)-1
            if not 0 <= i < len(wild) or wild[i] != before or after not in AA20 or after == before:
                raise ValueError('pair mutation identity disagrees with WT')
            key = (assay, protein['protein'], mutation)
            if key in seen:
                raise ValueError('duplicate structural mutation')
            seen.add(key)
            candidates = [dict(j=j,distance=j-i,identity=wild[j]) for j in range(i+3,len(wild))]
            pairs = mapped_pairs(dict(i=i,receivers=candidates),protein['sites'],wild)
            strata = []
            overlapping = []
            for name in DISTANCE_NAMES:
                rows = [p for p in pairs if p['stratum']==name]
                contact = [p for p in rows if p['contact']]
                noncontact = [p for p in rows if not p['contact']]
                matched = bool(contact and noncontact)
                if matched:
                    overlapping.append(name)
                strata.append(dict(stratum=name,contacts=len(contact),noncontacts=len(noncontact),matched=matched,
                                   distance_imbalance=float(np.mean([p['distance'] for p in contact])-np.mean([p['distance'] for p in noncontact])) if matched else None,
                                   log_distance_imbalance=float(np.mean([np.log(p['distance']) for p in contact])-np.mean([np.log(p['distance']) for p in noncontact])) if matched else None))
            for pair in pairs:
                group = [p for p in pairs if p['stratum']==pair['stratum'] and p['contact']==pair['contact']]
                pair['base_weight'] = 1/len(pairs)
                pair['matched_weight'] = 1/(2*len(overlapping)*len(group)) if pair['stratum'] in overlapping else 0.0
            mapped_j = {site['j'] for site in protein['sites']}
            units.append(dict(assay=assay,family=family,protein=protein['protein'],mutation=mutation,i=i,
                              anchor_mapped=i in mapped_j,candidate_receivers=len(candidates),
                              eligible_receivers=len(pairs),missing_receiver_coordinates=sum(p['j'] not in mapped_j for p in candidates),
                              missing_rsa=sum(p['rsa'] is None for p in pairs),matched_receivers=sum(p['matched_weight']>0 for p in pairs),
                              strata=strata,identity_sensitivity=_geometry_support(pairs,identity=True),
                              rsa_sensitivity=_geometry_support(pairs,complete_rsa=True),pairs=pairs))
    return dict(schema='structure_response_pairs_v1',analysis=4,stage='prepared_without_NLL',units=units,
                mutations=len(units),families=len({r['family'] for r in units}),
                eligible_receivers=sum(r['eligible_receivers'] for r in units),
                weights='uniform base weights over geometry-eligible receivers; matched weights normalize contact classes and overlapping distance strata, then equal mutations/assays/families',
                response_status='not evaluated; token grid and single-residue support require retained arrays; matching must be recomputed on that admitted support',
                endpoint_blind=True, selected_epistasis_pairs_used=False)


def _matched_contrast(pairs: list[dict], *, outcome='absolute', identity=False, covariates=()):
    """Uniform pre-outcome receiver weights; overlap strata have equal weight.

    All receivers are retained in overlapping strata (no nearest-pair selection).
    Each contact class is normalized within stratum. Common covariate slopes
    are fitted after stratum/contact demeaning; the adjusted contrast evaluates
    both classes at the same pooled covariate means. No endpoint is inspected.
    """
    groups = {}
    for row in pairs:
        if any(row[key] is None for key in covariates):
            continue
        key = (row['stratum'], row['identity']) if identity else (row['stratum'],)
        groups.setdefault(key, []).append(row)
    groups = {key: rows for key, rows in groups.items() if {r['contact'] for r in rows} == {True, False}}
    if not groups:
        return None
    contrasts, imbalance, log_imbalance, census = [], [], [], []
    design, target, weights = [], [], []
    columns = ('log_distance', *covariates)
    for key, rows in sorted(groups.items()):
        block = []
        for contact in (False, True):
            subset = [r for r in rows if r['contact'] == contact]
            x = np.asarray([[np.log(r['distance']), *[r[c] for c in covariates]] for r in subset])
            y = np.asarray([abs(r['response']) if outcome == 'absolute' else -r['response'] for r in subset])
            block.append((x.mean(0), y.mean(), np.mean([r['distance'] for r in subset])))
            design.extend(x-x.mean(0)); target.extend(y-y.mean())
            weights.extend([1/(2*len(groups)*len(subset))]*len(subset))
        contrasts.append(block[1][1]-block[0][1])
        imbalance.append(block[1][2]-block[0][2])
        log_imbalance.append(block[1][0]-block[0][0])
        census.append(dict(stratum=list(key), contacts=sum(r['contact'] for r in rows), noncontacts=sum(not r['contact'] for r in rows), distance_imbalance=float(imbalance[-1]), covariate_imbalance=log_imbalance[-1].tolist()))
    w = np.sqrt(weights)
    beta = np.linalg.lstsq(np.asarray(design)*w[:,None], np.asarray(target)*w, rcond=None)[0]
    raw = float(np.mean(contrasts))
    return dict(contrast=raw, adjusted_contrast=float(raw-np.mean(log_imbalance, axis=0)@beta),
                distance_imbalance=float(np.mean(imbalance)), log_distance_imbalance=float(np.mean(log_imbalance, axis=0)[0]),
                adjustment_columns=list(columns), adjustment_slopes=beta.tolist(), strata=census,
                matched_strata=len(census), matched_receivers=sum(c['contacts']+c['noncontacts'] for c in census),
                candidate_receivers=len(pairs), missing_covariates=sum(any(r[k] is None for k in covariates) for r in pairs))


def analyse_contacts(mutations: list[dict], *, draws=2000, seed=20261006) -> dict:
    """Receiver/class -> stratum -> mutation -> assay -> family equal weighting."""
    seen = set()
    records = []
    for row in mutations:
        key = (row['assay'], row['protein'], row['mutation'])
        if key in seen:
            raise ValueError('duplicate mutation structure unit')
        seen.add(key)
        pairs = row['pairs']
        if len({p['j'] for p in pairs}) != len(pairs):
            raise ValueError('duplicate receiver')
        record = {k: row[k] for k in ('assay', 'family', 'protein', 'mutation')}
        record.update(receivers=len(pairs), missing_rsa=sum(p['rsa'] is None for p in pairs))
        record['absolute'] = _matched_contrast(pairs)
        record['disruption'] = _matched_contrast(pairs, outcome='disruption')
        record['identity'] = _matched_contrast(pairs, identity=True)
        record['baseline'] = _matched_contrast(pairs, covariates=('wt_nll',))
        record['rsa'] = _matched_contrast(pairs, covariates=('rsa',))
        records.append(record)
    columns = [(name, metric) for name in ('absolute','disruption','identity','baseline','rsa') for metric in ('contrast','adjusted_contrast')]
    families = sorted({r['family'] for r in records})
    matrix = np.full((len(families),len(columns)), np.nan)
    assays = {}
    for row in records:
        if row['assay'] in assays and assays[row['assay']] != row['family']:
            raise ValueError('assay crosses families')
        assays[row['assay']] = row['family']
    for i, family in enumerate(families):
        for j, (name, metric) in enumerate(columns):
            means = []
            for assay in sorted(a for a, f in assays.items() if f == family):
                values = [r[name][metric] for r in records if r['assay']==assay and r[name] is not None]
                if values:
                    means.append(np.mean(values))
            if means:
                matrix[i,j] = np.mean(means)
    available = np.isfinite(matrix).sum(0)
    inferential = np.flatnonzero(available >= 8)
    inference = simultaneous_bands(matrix[:,inferential], draws=draws, seed=seed) if len(inferential) else None
    return dict(analysis=4, mutations=records, families=families, columns=columns,
                family_values=[[None if not np.isfinite(v) else float(v) for v in row] for row in matrix],
                inference=inference, inferential_columns=[columns[j] for j in inferential],
                unavailable_columns=[columns[j] for j in range(len(columns)) if j not in inferential],
                inference_status='available for declared supported contrasts' if inference else 'fewer than 8 families for all contrasts',
                estimand='contact minus noncontact downstream native NLL response; absolute primary, negative signed response disruption secondary',
                weights='uniform receiver weights before response/endpoint inspection; equal contact classes and overlap strata within mutation, equal mutations within assay, assays within family, families; family-only resampling',
                missing_rsa_policy='complete-RSA sensitivity only; primary retains missing RSA',
                limitations='conditional on exact mapped structure and strict single-residue token support; adjusted common slope is observational, not causal')


def potential_distance_census(rows: list[dict]) -> dict:
    """Full-panel sequence geometry only; never assert tokenizer/model support."""
    census, excluded = [], []
    seen = set()
    for row in rows:
        if row['assay'] in seen:
            raise ValueError('duplicate sequence-census assay')
        seen.add(row['assay'])
        wild = row['wildtype']
        if not wild or set(wild)-set(AA20):
            raise ValueError('invalid sequence-census WT')
        for mutation in row['mutants']:
            if ':' in mutation:
                excluded.append(dict(assay=row['assay'],mutation=mutation,reason='multiple_substitutions'))
                continue
            match = re.fullmatch(r'([A-Z])(\d+)([A-Z])',mutation)
            if match is None:
                raise ValueError('invalid sequence-census substitution')
            before, position, after = match.groups()
            i = int(position)-1
            if not 0 <= i < len(wild) or wild[i] != before or after not in AA20 or after==before:
                raise ValueError('sequence-census substitution disagrees with WT')
            length = len(wild)-i-1
            edges = [0,min(length,8),min(length,32),min(length,128),length]
            census.append(dict(assay=row['assay'],family=row['cluster'],mutation=mutation,i=i,
                               potential_residues=np.diff(edges).tolist()))
    return dict(schema='distance_bin_potential_v1',analysis=3,stage='sequence_geometry_without_NLL',units=census,
                excluded_multiple_substitutions=excluded,
                support={name:dict(potential_residues=sum(r['potential_residues'][j] for r in census),
                                   potential_nonempty_rows=sum(r['potential_residues'][j]>0 for r in census),
                                   assays=len({r['assay'] for r in census if r['potential_residues'][j]>0}),
                                   families=len({r['family'] for r in census if r['potential_residues'][j]>0})) for j,name in enumerate(BIN_NAMES)},
                structure_required=False,
                qualification='sequence-geometric potential only, not actual token support; original retained arrays must establish scored boundaries, strict alignment and single-residue receivers outside mutation-spanning tokens; no structural restriction for analysis 3')


def evaluate_bins(rows: list[dict], *, seeds=(20260923,20260924,20260925), draws=2000, bootstrap_seed=20261006,
                  outer_splits=5, inner_splits=4) -> dict:
    """Analysis 3: matched full versus drop-bin predictive increments, not mass.

    Each assay provides B (already defined baseline design), measured, mutants,
    cluster and responses in exact mutant order. All designs share rows and
    outer/inner folds. Own and unallocated remainder are always nuisance inputs.
    """
    if not rows or len({r['assay'] for r in rows}) != len(rows) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('empty/duplicate assays or split seeds')
    designs = []
    support = []
    for row in rows:
        n = len(row['mutants'])
        response = row['responses']
        baseline = np.asarray(row['B'], float)
        measured = np.asarray(row['measured'], float)
        if (n < 3 or len(set(row['mutants'])) != n or len(response) != n
                or [r['mutation'] for r in response] != row['mutants']
                or baseline.ndim != 2 or baseline.shape[0] != n
                or measured.shape != (n,) or not np.isfinite(baseline).all() or not np.isfinite(measured).all()):
            raise ValueError('predictive assay rows/identities are not aligned')
        components = np.asarray([[r['own'],r['remainder'],*r['bins']] for r in response], float)
        counts = np.asarray([r['bin_support'] for r in response])
        if components.shape != (n,6) or counts.shape != (n,4) or counts.dtype.kind not in 'iu' or np.any(counts < 0) or not np.isfinite(components).all():
            raise ValueError('invalid response bin support')
        for r, values in zip(response, components):
            if not np.isclose(values.sum(),r['native'],rtol=0,atol=1e-10):
                raise ValueError('response design does not close to native score')
        if np.any((counts==0) & (components[:,2:]!=0)):
            raise ValueError('empty bins must be exactly zero')
        designs.append(np.c_[baseline, np.column_stack([standardized_rank(components[:,k]) for k in range(6)])])
        support.append(counts)
    if len({x.shape[1] for x in designs}) != 1:
        raise ValueError('baseline feature dimensions change across assays')
    full = np.concatenate(designs)
    count = np.concatenate(support)
    supported = [j for j in range(4) if np.any(full[:,full.shape[1]-4+j] != 0)]
    x = {'full': full, **{f'drop_{BIN_NAMES[j]}':np.delete(full,full.shape[1]-4+j,axis=1) for j in supported}}
    aid = np.concatenate([[r['assay']]*len(r['mutants']) for r in rows])
    family = np.concatenate([[r['cluster']]*len(r['mutants']) for r in rows])
    y = np.concatenate([r['measured'] for r in rows])
    by_seed, audit = [], {}
    for seed in seeds:
        predictions = {}
        membership = None
        for name, design in x.items():
            predictions[name], folds = nested_predict(design,y,aid,family,seed=seed,device='cpu',outer_splits=outer_splits,inner_splits=inner_splits)
            current = fold_membership(folds)
            if membership is not None and membership != current:
                raise ValueError('bin designs do not share nested folds')
            membership = current
            audit[f'{seed}:{name}'] = folds
        records = []
        for row in rows:
            index = aid == row['assay']
            target = standardized_rank(y[index])
            metrics = {}
            for name, pred in predictions.items():
                metrics[name] = correlation(rankdata(pred[index]),target)
                metrics[name+'_mse'] = float(np.mean((target-pred[index])**2))
            entry = dict(assay=row['assay'],cluster=row['cluster'])
            for j in supported:
                name = f'drop_{BIN_NAMES[j]}'
                entry[BIN_NAMES[j]+'_spearman'] = None if metrics['full'] is None or metrics[name] is None else metrics['full']-metrics[name]
                entry[BIN_NAMES[j]+'_mse'] = metrics[name+'_mse']-metrics['full_mse']
            records.append(entry)
        by_seed.append(records)
    columns = [BIN_NAMES[j]+suffix for j in supported for suffix in ('_spearman','_mse')]
    families = sorted(set(family.tolist()))
    matrix = np.full((len(families),len(columns)),np.nan)
    averaged = []
    for i, row in enumerate(rows):
        entry = dict(assay=row['assay'],cluster=row['cluster'])
        for key in columns:
            values = [records[i][key] for records in by_seed]
            entry[key] = None if any(v is None for v in values) else float(np.mean(values))
        averaged.append(entry)
    for i, f in enumerate(families):
        for j, key in enumerate(columns):
            values = [r[key] for r in averaged if r['cluster']==f and r[key] is not None]
            if values:
                matrix[i,j] = np.mean(values)
    inferential = np.flatnonzero(np.isfinite(matrix).sum(0)>=8)
    inference = simultaneous_bands(matrix[:,inferential],draws=draws,seed=bootstrap_seed) if len(inferential) else None
    return dict(analysis=3, bins=list(BIN_NAMES), tested_bins=[BIN_NAMES[j] for j in supported],
                support={name:dict(receivers=int(count[:,j].sum()),nonempty_rows=int((count[:,j]>0).sum()),zero_feature=bool(j not in supported)) for j,name in enumerate(BIN_NAMES)},
                seeds=list(seeds), assays=averaged, families=families, contrasts=columns,
                folds=audit, folds_identical=True,
                partition_declaration='new seeded partitions on admitted subset families; matched designs share realized inner/outer memberships; no claim of original full-cohort fold equivalence',
                inference=inference, inferential_columns=[columns[j] for j in inferential],
                unavailable_columns=[columns[j] for j in range(len(columns)) if j not in inferential],
                inference_status='available for declared supported contrasts' if inference else 'no supported contrast or fewer than 8 available families',
                estimand='full B+own+all bins+remainder minus drop-bin held-family predictive rank performance; not response magnitude',
                allocation='only downstream single-residue tokens enter bins; own, upstream, special, multiresidue and native rounding remainder remain nuisance',
                resampling='split-seed contrasts averaged per assay before equal-assay family averages; shared family simultaneous bootstrap conditional on fitted predictions')


RECOVERY_REQUIREMENTS = dict(schema='residue_response_recovery_v1', status='source_required_not_recovered',
    historical_term_file_count=191, historical_fit_file_count=6,
    local_inventory='Reader found no original positional NPZ/tars in repository, archive/logs/R1/position_terms_20260926, results/R1/position_terms_20260926, or bounded sibling /Data/lzp/d2 and /Data/lzp/backup-plm-capability-boundaries-20260927; .pi synthetic NPZ are unrelated',
    remote_launch_package='/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer/packages/20260926223301_0c1728098f40',
    remote_launch_evidence='receipts/dispatch.json (reader inventory; remote not inspected here)',
    remote_fit_directories=['results/position_followup_20260927/paired-progen3','results/position_followup_20260927/numerical-fits/<arm>'],
    remote_inventory_status='exact token archive filenames unresolved; 191 term/six fit historical count does not establish presence of raw arrays; locate original ragged extraction and manifest_<arm>.json, not merely aggregate term/fit archives',
    recovery_before_inference='inspect remote sources and recover/validate retained arrays first; no new inference is justified until remote inspected; missing local arrays do not establish permanently missing extraction',
    required_npz=['position_nats (float32)','position_offsets','position_residue_counts','position_residue_offsets','position_sum_check_nats (=0)','mutants (ordered)','likelihood (native signed NLL delta)','wt_likelihood'],
    required_manifest='manifest_<arm>.json with original extraction identity, retention and packing provenance; compare sidecar/source hashes before admission',
    packing_sidecar='ordered states: exact sequence, complete packed ids, scored span [start,end), counts, residue offset; WT first then original mutant order; executable --mode packing --arm ARM exports and checks with original Packer.state/checked and tokenizer-only load_readout_arm(device=None); no model forward',
    mixed_archive_admission='validate all original identities, retention metadata and native state/delta closure before per-variant exclusions; multiple substitutions, unscored mutations and grid misalignment are excluded with original indices; B, measured, mutant and sequence rows projected together; insufficient assay/family support reported explicitly',
    structural_admission='--coverage must bind site status/source hash/path/chain/model/exact mapping and experimental method; contact_atom must explicitly declare CB or Gly CA; excluded or theoretical coordinates are refused',
    readiness='CPU adapter, mixed-archive selection, tokenizer-only packing export and structural provenance gates executable; native-response computation remains source-recovery dependent',
    identity='original cohort wildtype, ordered mutants and exact mutant sequences; archive and sidecar provenance hashes',
    structure='all exact mapped sites: zero-based j, WT identity, CB (Gly CA), xyz Angstrom, rsa nonnegative, un-clipped (may exceed 1), or null; no selected epistasis pairs',
    prediction='per-assay original B feature rows, measured endpoint, mutant order and frozen family assignments',
    prohibition='raw_M ranks, collapsed own/downstream summaries or inferred token counts are not retained receiver responses; absent raw arrays cannot be recovered from summaries; no fabricated inference')
