"""Structural-contact annotation and matched contact-versus-non-contact contrasts.

This module supports one question on the frozen pairwise cohort: is interaction
signal enriched in residue pairs that touch in an experimental structure, relative
to non-contacting pairs matched on the confounders that would otherwise explain the
difference? It carries the structure side (mmCIF reading, sequence-to-structure
numbering alignment, the contact definition, per-residue relative solvent
accessibility), the matching scheme and its balance report, and the weighted
estimators with their two-stage group/site-pair bootstrap.

The annotation path takes sequences, site-pair indices and coordinates only, so a
contact call or a matching weight cannot be a function of an outcome. The
measurement-side estimators take the annotation and the endpoint separately, as
arrays the caller has already extracted: this module does not open a cohort or a
model score. The binned residualizer is one of those estimators. Its in-sample
curve is fit on the cycles it adjusts; its cross-fit curve for a family group is
fit on the other family groups only.

Sequence separation is a distinct axis from contact and is not a contact
annotation: every matched comparison here balances separation explicitly, because
a contact test that did not would rediscover separation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gzip
import hashlib
import math

import numpy as np

from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS, bootstrap_unit_floor
from ..stability.pairwise_stability import COHORT_SCHEMA

#: Primary contact definition: two residues are in contact when the minimum
#: distance between any pair of their non-hydrogen atoms is at or below this
#: threshold, in angstrom. 5.0 A over all heavy atoms is the standard
#: side-chain-inclusive residue contact and does not depend on a CB placement for
#: glycine.
HEAVY_ATOM_CONTACT_ANGSTROM = 5.0

#: Secondary contact definition, reported beside the primary one: CB-CB distance
#: at or below this threshold in angstrom, with CA standing in for glycine. This
#: is the convention most contact-prediction work uses, and it is retained here as
#: a definition sensitivity rather than as the primary call.
CB_CONTACT_ANGSTROM = 8.0

#: Van der Waals radii in angstrom by element, and the solvent probe radius, for
#: the Shrake-Rupley accessible-surface calculation.
VDW_RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'SE': 1.90, 'P': 1.80}
PROBE_RADIUS = 1.40
SASA_SPHERE_POINTS = 256

#: Maximum accessible surface area per residue in square angstrom, Tien et al.
#: 2013 theoretical values. Relative solvent accessibility is the residue's
#: accessible area divided by its entry here.
MAX_ACCESSIBLE_AREA = {
    'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0, 'C': 167.0, 'E': 223.0,
    'Q': 225.0, 'G': 104.0, 'H': 224.0, 'I': 197.0, 'L': 201.0, 'K': 236.0,
    'M': 224.0, 'F': 240.0, 'P': 159.0, 'S': 155.0, 'T': 172.0, 'W': 285.0,
    'Y': 263.0, 'V': 174.0,
}

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
}

#: Residue classes for the matching covariate. The hydrophobic set is the one the
#: matching coarsens on; the charged set is reported as achieved balance.
HYDROPHOBIC = set('AVILMFWCY')
CHARGED = set('DEKR')

#: Kyte-Doolittle hydropathy, reported as achieved balance on a continuous scale
#: rather than used as a matching cell.
KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}


# --------------------------------------------------------------------------- #
# mmCIF reading. Only the categories this annotation needs are parsed, and an
# absent category is an explicit failure at the call site rather than a default.
# --------------------------------------------------------------------------- #

def _tokenise(line: str) -> list[str]:
    tokens, index, width = [], 0, len(line)
    while index < width:
        if line[index] in ' \t':
            index += 1
            continue
        if line[index] == '#':
            break
        if line[index] in '\'"':
            quote, index = line[index], index + 1
            start = index
            while index < width and not (line[index] == quote
                                         and (index + 1 == width or line[index + 1] in ' \t')):
                index += 1
            tokens.append(line[start:index])
            index += 1
            continue
        start = index
        while index < width and line[index] not in ' \t':
            index += 1
        tokens.append(line[start:index])
    return tokens


def read_cif(text: str) -> dict[str, dict[str, list[str]]]:
    """Parse an mmCIF data block into ``category -> field -> list of values``.

    Single-value categories give one-element lists, so a caller reads both forms
    the same way. Multi-line semicolon-delimited values are joined without their
    newlines, which is what ``pdbx_seq_one_letter_code`` needs.
    """

    lines = text.splitlines()
    result: dict[str, dict[str, list[str]]] = {}
    index, count = 0, len(lines)

    def semicolon_value(position: int) -> tuple[str, int]:
        parts = [lines[position][1:]]
        position += 1
        while position < count and not lines[position].startswith(';'):
            parts.append(lines[position])
            position += 1
        return ''.join(part.strip() for part in parts), position + 1

    while index < count:
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.lower().startswith('data_'):
            index += 1
            continue
        if stripped.lower() == 'loop_':
            index += 1
            names: list[tuple[str, str]] = []
            while index < count and lines[index].strip().startswith('_'):
                category, _, field = lines[index].strip().split()[0].partition('.')
                names.append((category[1:], field))
                index += 1
            columns: list[list[str]] = [[] for _ in names]
            row: list[str] = []
            while index < count:
                current = lines[index]
                if current.startswith(';'):
                    value, index = semicolon_value(index)
                    row.append(value)
                    if len(row) == len(names):
                        for position, value in enumerate(row):
                            columns[position].append(value)
                        row = []
                    continue
                stripped = current.strip()
                if not stripped or stripped.startswith('#'):
                    index += 1
                    continue
                if stripped.startswith('_') or stripped.lower() in ('loop_',) \
                        or stripped.lower().startswith('data_'):
                    break
                row.extend(_tokenise(current))
                while len(row) >= len(names):
                    for position, value in enumerate(row[:len(names)]):
                        columns[position].append(value)
                    row = row[len(names):]
                index += 1
            if row:
                raise ValueError('mmCIF loop ended with an incomplete row')
            for (category, field), values in zip(names, columns):
                result.setdefault(category, {})[field] = values
            continue
        if stripped.startswith('_'):
            category, _, field = stripped.split()[0].partition('.')
            rest = stripped[len(stripped.split()[0]):].strip()
            if rest:
                value = _tokenise(rest)[0]
                index += 1
            elif index + 1 < count and lines[index + 1].startswith(';'):
                value, index = semicolon_value(index + 1)
            else:
                index += 1
                value = _tokenise(lines[index])[0] if index < count else '?'
                index += 1
            result.setdefault(category[1:], {})[field] = [value]
            continue
        index += 1
    return result


@dataclass
class Residue:
    """One polymer residue of one chain in one model."""

    label_seq_id: int
    auth_seq_id: str
    comp_id: str
    residue: str
    atoms: list[str] = field(default_factory=list)
    elements: list[str] = field(default_factory=list)
    coordinates: list[tuple[float, float, float]] = field(default_factory=list)

    def array(self) -> np.ndarray:
        return np.asarray(self.coordinates, dtype=float).reshape(-1, 3)

    def radii(self) -> np.ndarray:
        return np.asarray([VDW_RADII.get(element, 1.80) for element in self.elements], dtype=float)

    def named(self, name: str) -> np.ndarray | None:
        return (np.asarray(self.coordinates[self.atoms.index(name)], dtype=float)
                if name in self.atoms else None)


@dataclass
class Chain:
    """One polymer chain of one model, indexed by its entity sequence position."""

    label_asym_id: str
    auth_asym_id: str
    entity_id: str
    model: int
    residues: dict[int, Residue]


def _entity_sequences(cif: dict) -> dict[str, str]:
    poly = cif.get('entity_poly')
    if not poly:
        raise ValueError('mmCIF carries no _entity_poly category')
    field_name = ('pdbx_seq_one_letter_code_can' if 'pdbx_seq_one_letter_code_can' in poly
                  else 'pdbx_seq_one_letter_code')
    sequences = {}
    for entity, kind, sequence in zip(poly['entity_id'], poly['type'], poly[field_name]):
        if 'polypeptide' not in kind:
            continue
        sequences[entity] = ''.join(sequence.split()).upper()
    return sequences


def load_structure(path) -> dict:
    """Read one mmCIF file into polymer chains per model, plus its provenance.

    Hydrogens, waters, non-polymer heteroatoms and alternate locations other than
    the first are dropped: the contact definition is over the polymer's heavy
    atoms of one conformer.
    """

    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as handle:
        text = handle.read()
    cif = read_cif(text)
    entities = _entity_sequences(cif)
    atoms = cif.get('atom_site')
    if not atoms:
        raise ValueError(f'{path}: mmCIF carries no _atom_site category')
    chains: dict[tuple[int, str], Chain] = {}
    columns = {name: atoms[name] for name in atoms}
    rows = len(columns['id'])
    for index in range(rows):
        if columns['group_PDB'][index] != 'ATOM' and columns['label_comp_id'][index] != 'MSE':
            continue
        element = columns['type_symbol'][index].upper()
        if element in ('H', 'D'):
            continue
        entity = columns['label_entity_id'][index]
        if entity not in entities:
            continue
        alt = columns['label_alt_id'][index]
        if alt not in ('.', '?', 'A'):
            continue
        label_seq = columns['label_seq_id'][index]
        if label_seq in ('.', '?'):
            continue
        model = int(columns['pdbx_PDB_model_num'][index])
        key = (model, columns['label_asym_id'][index])
        chain = chains.get(key)
        if chain is None:
            chain = chains[key] = Chain(
                label_asym_id=columns['label_asym_id'][index],
                auth_asym_id=columns['auth_asym_id'][index],
                entity_id=entity, model=model, residues={})
        position = int(label_seq)
        residue = chain.residues.get(position)
        if residue is None:
            comp = columns['label_comp_id'][index]
            residue = chain.residues[position] = Residue(
                label_seq_id=position, auth_seq_id=columns['auth_seq_id'][index],
                comp_id=comp, residue=THREE_TO_ONE.get(comp, 'X'))
        name = columns['label_atom_id'][index]
        if name in residue.atoms:
            continue
        residue.atoms.append(name)
        residue.elements.append(element)
        residue.coordinates.append((float(columns['Cartn_x'][index]),
                                    float(columns['Cartn_y'][index]),
                                    float(columns['Cartn_z'][index])))
    method = cif.get('exptl', {}).get('method', ['unknown'])[0]
    return {'entities': entities, 'chains': chains, 'method': method,
            'models': sorted({model for model, _ in chains}),
            'entry_id': cif.get('entry', {}).get('id', ['?'])[0]}


def file_digest(path) -> tuple[str, int]:
    sha, size = hashlib.sha256(), 0
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            sha.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), size


# --------------------------------------------------------------------------- #
# Sequence-to-structure numbering. A mismatch is refused loudly at the call
# site: a contact annotation read off the wrong residue is worse than none.
# --------------------------------------------------------------------------- #

def align_positions(query: str, target: str) -> dict:
    """Local alignment of a cohort wild type to an entity sequence, with the map.

    Returns the aligned column count, the identical-column count and a mapping
    from 1-based query position to 1-based target position for matched columns
    only. Scoring is a simple identity match with affine gaps, which is
    sufficient because these two sequences are the same construct up to trimming
    and a small number of engineered substitutions; a remote-homology aligner is
    not what this step needs.
    """

    match, mismatch, gap_open, gap_extend = 2.0, -1.0, 6.0, 1.0
    n, m = len(query), len(target)
    if n == 0 or m == 0:
        raise ValueError('alignment needs two nonempty sequences')
    neg = -1e18
    best_state = np.full((n + 1, m + 1), neg)
    gap_q = np.full((n + 1, m + 1), neg)
    gap_t = np.full((n + 1, m + 1), neg)
    back = np.zeros((n + 1, m + 1, 3), dtype=np.int8)
    # A local alignment may start at any position, so the zero row and column of
    # the match state are zero rather than forbidden. Leaving them at the
    # sentinel made the first aligned column score nothing, which in a repetitive
    # sequence resolved a tie onto an alignment shifted by one residue -- and a
    # shifted numbering map is exactly what this function must not produce.
    best_state[0, :] = 0.0
    best_state[:, 0] = 0.0
    for i in range(1, n + 1):
        row_query = query[i - 1]
        for j in range(1, m + 1):
            options = (best_state[i - 1, j - 1], gap_q[i - 1, j - 1], gap_t[i - 1, j - 1])
            source = int(np.argmax(options))
            score = options[source] + (match if row_query == target[j - 1] else mismatch)
            if score <= 0.0:
                score, source = 0.0, 3
            best_state[i, j] = score
            back[i, j, 0] = source
            opened = best_state[i - 1, j] - gap_open - gap_extend
            extended = gap_q[i - 1, j] - gap_extend
            gap_q[i, j] = max(opened, extended)
            back[i, j, 1] = 1 if extended > opened else 0
            opened = best_state[i, j - 1] - gap_open - gap_extend
            extended = gap_t[i, j - 1] - gap_extend
            gap_t[i, j] = max(opened, extended)
            back[i, j, 2] = 2 if extended > opened else 0
    end = np.unravel_index(int(np.argmax(best_state)), best_state.shape)
    i, j = int(end[0]), int(end[1])
    if best_state[i, j] <= 0.0:
        return {'columns': 0, 'identical': 0, 'mapping': {}, 'query_span': (0, 0),
                'target_span': (0, 0)}
    mapping: dict[int, int] = {}
    columns = identical = 0
    state = 0
    last_i, last_j = i, j
    while i > 0 and j > 0:
        if state == 0:
            source = int(back[i, j, 0])
            columns += 1
            if query[i - 1] == target[j - 1]:
                identical += 1
                mapping[i] = j
            last_i, last_j = i, j
            if source == 3:
                break
            i, j, state = i - 1, j - 1, source
        elif state == 1:
            columns += 1
            source = int(back[i, j, 1])
            last_i, last_j = i, j
            i, state = i - 1, source
        else:
            columns += 1
            source = int(back[i, j, 2])
            last_i, last_j = i, j
            j, state = j - 1, source
    return {'columns': columns, 'identical': identical, 'mapping': mapping,
            'query_span': (last_i, int(end[0])), 'target_span': (last_j, int(end[1]))}


def match_chain(wildtype: str, structure: dict) -> dict:
    """Choose the chain whose entity sequence carries this wild type, or refuse.

    The chosen chain is the one with the highest identity over the wild type, ties
    broken toward the most observed residues. The return carries the identity, the
    numbering map and the per-position mismatches, so the caller applies the
    admission rule rather than this function silently accepting a near match.
    """

    candidates = []
    for entity, sequence in structure['entities'].items():
        alignment = align_positions(wildtype, sequence)
        mapped = len(alignment['mapping'])
        chains = sorted({chain.label_asym_id for (model, _), chain in structure['chains'].items()
                         if chain.entity_id == entity and model == structure['models'][0]})
        for label in chains:
            chain = structure['chains'][(structure['models'][0], label)]
            observed = sum(1 for position in alignment['mapping'].values()
                           if position in chain.residues)
            candidates.append({'entity_id': entity, 'label_asym_id': label,
                               'auth_asym_id': chain.auth_asym_id,
                               'entity_length': len(sequence),
                               'identity_over_wildtype': mapped / len(wildtype),
                               'aligned_columns': alignment['columns'],
                               'identical_columns': alignment['identical'],
                               'observed_mapped_positions': observed,
                               'mapping': alignment['mapping'],
                               'mismatch_positions': sorted(
                                   set(range(1, len(wildtype) + 1)) - set(alignment['mapping']))})
    if not candidates:
        raise ValueError('structure carries no polypeptide entity')
    candidates.sort(key=lambda entry: (entry['identity_over_wildtype'],
                                       entry['observed_mapped_positions']), reverse=True)
    best = candidates[0]
    best['chain_candidates'] = len(candidates)
    best['equal_identity_chains'] = sum(
        1 for entry in candidates
        if entry['identity_over_wildtype'] == best['identity_over_wildtype'])
    return best


# --------------------------------------------------------------------------- #
# The contact definition and relative solvent accessibility.
# --------------------------------------------------------------------------- #

def minimum_heavy_atom_distance(first: Residue, second: Residue) -> float:
    left, right = first.array(), second.array()
    if not len(left) or not len(right):
        return float('inf')
    return float(np.sqrt(((left[:, None, :] - right[None, :, :]) ** 2).sum(-1)).min())


def cb_distance(first: Residue, second: Residue) -> float:
    left = first.named('CB') if first.residue != 'G' else first.named('CA')
    right = second.named('CB') if second.residue != 'G' else second.named('CA')
    if left is None:
        left = first.named('CA')
    if right is None:
        right = second.named('CA')
    if left is None or right is None:
        return float('inf')
    return float(np.linalg.norm(left - right))


def _sphere(points: int) -> np.ndarray:
    """Golden-spiral points on the unit sphere: deterministic and near-uniform."""

    index = np.arange(points, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * index / points)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * index
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi), np.cos(phi)])


def residue_accessibility(chain: Chain, positions: list[int] | None = None,
                          *, points: int = SASA_SPHERE_POINTS) -> dict[int, float]:
    """Shrake-Rupley accessible surface area per residue, in square angstrom.

    The calculation is over the isolated chain: the assay measures one domain's
    folding stability, so burial is defined against that chain and not against
    crystal neighbours. ``positions`` restricts the returned residues; the
    occluding set is always the whole chain.
    """

    order = sorted(chain.residues)
    coordinates = np.concatenate([chain.residues[position].array() for position in order])
    radii = np.concatenate([chain.residues[position].radii() for position in order]) + PROBE_RADIUS
    owner = np.concatenate([np.full(len(chain.residues[position].atoms), position)
                            for position in order])
    sphere = _sphere(points)
    wanted = set(order if positions is None else positions)
    areas: dict[int, float] = {position: 0.0 for position in sorted(wanted)}
    for atom in range(len(coordinates)):
        if owner[atom] not in wanted:
            continue
        radius = radii[atom]
        offsets = coordinates - coordinates[atom]
        distances = np.sqrt((offsets ** 2).sum(-1))
        near = np.flatnonzero((distances < radius + radii) & (distances > 0))
        test = coordinates[atom] + sphere * radius
        if len(near):
            buried = (np.sqrt(((test[:, None, :] - coordinates[near][None, :, :]) ** 2).sum(-1))
                      < radii[near][None, :]).any(axis=1)
            exposed = int((~buried).sum())
        else:
            exposed = points
        areas[owner[atom]] += 4.0 * math.pi * radius ** 2 * exposed / points
    return areas


def relative_accessibility(area: float, residue: str) -> float:
    maximum = MAX_ACCESSIBLE_AREA.get(residue)
    if maximum is None:
        raise ValueError(f'no maximum accessible area declared for residue {residue!r}')
    return area / maximum


# --------------------------------------------------------------------------- #
# The label-free cohort projection and the per-background annotation.
# --------------------------------------------------------------------------- #

def site_pair_plan(cohort: dict) -> dict:
    """Label-free projection of the frozen cohort: names, wild types, site pairs.

    No measurement, epsilon or measurement metadata is copied, so everything
    downstream of this function is independent of the endpoint by construction.
    The wild-type state is re-derived from every cycle rather than trusted once.
    """

    if cohort.get('schema') != COHORT_SCHEMA:
        raise ValueError('unexpected cohort schema')
    backgrounds = []
    for row in cohort['backgrounds']:
        wildtype = row['cycles'][0]['sequences'][0]
        for cycle in row['cycles']:
            if cycle['sequences'][0] != wildtype:
                raise ValueError(f"{row['name']}: cycles disagree on the wild-type state")
        pairs = sorted({tuple(sorted(cycle['positions'])) for cycle in row['cycles']})
        declared = sorted(tuple(sorted(pair)) for pair in row['site_pairs'])
        if pairs != declared:
            raise ValueError(f"{row['name']}: cycle positions disagree with the declared site pairs")
        for low, high in pairs:
            if not 1 <= low < high <= len(wildtype):
                raise ValueError(f"{row['name']}: site pair {low}-{high} is outside the wild type")
        backgrounds.append({'name': row['name'], 'group': row['group'], 'kind': row['kind'],
                            'length': len(wildtype), 'wildtype': wildtype,
                            'site_pairs': [list(pair) for pair in pairs]})
    return {'backgrounds': backgrounds,
            'site_pairs': sum(len(row['site_pairs']) for row in backgrounds)}


def _reported(value: float) -> float | None:
    """A distance for the artefact: ``None`` where no such distance exists.

    An absent neighbouring chain, or a residue carrying neither CB nor CA, has no
    distance rather than an infinite one, and the artefact must stay valid JSON
    that the shared writer will accept.
    """

    return float(value) if np.isfinite(value) else None


def _other_chain_distance(structure: dict, chain: Chain, positions: list[int]) -> float:
    """Minimum heavy-atom distance from these residues to any other polymer chain."""

    others = [other for (model, label), other in structure['chains'].items()
              if model == chain.model and label != chain.label_asym_id]
    if not others:
        return float('inf')
    target = np.concatenate([chain.residues[position].array() for position in positions])
    best = float('inf')
    for other in others:
        coordinates = np.concatenate([residue.array() for residue in other.residues.values()])
        best = min(best, float(np.sqrt(
            ((target[:, None, :] - coordinates[None, :, :]) ** 2).sum(-1)).min()))
    return best


def annotate_background(entry: dict, path) -> dict:
    """Contact calls and matching covariates for one background's site pairs.

    Refuses the background unless its wild type aligns at 100.0% identity over its
    whole length to one polypeptide entity and both residues of every site pair
    are present in the matched chain's coordinates. A structure whose numbering
    does not carry the assayed sequence exactly cannot annotate a contact, and
    accepting a near match would put a contact call on the wrong residue.
    """

    structure = load_structure(path)
    wildtype = entry['wildtype']
    matched = match_chain(wildtype, structure)
    if matched['identity_over_wildtype'] < 1.0:
        raise ValueError(
            f"wild-type identity to the structure is "
            f"{100.0 * matched['identity_over_wildtype']:.1f}% over {len(wildtype)} residues, "
            f"{len(matched['mismatch_positions'])} positions unmatched "
            f"({matched['mismatch_positions'][:8]})")
    first = structure['models'][0]
    chain = structure['chains'][(first, matched['label_asym_id'])]
    mapping = matched['mapping']
    needed = sorted({position for pair in entry['site_pairs'] for position in pair})
    missing = [position for position in needed if mapping[position] not in chain.residues]
    if missing:
        raise ValueError(f'site-pair positions {missing} are unobserved in the matched chain')
    for position in needed:
        observed = chain.residues[mapping[position]].residue
        if observed != wildtype[position - 1]:
            raise ValueError(f'position {position} carries {observed} in the structure and '
                             f'{wildtype[position - 1]} in the wild type')
    areas = residue_accessibility(chain, [mapping[position] for position in needed])
    accessibility = {position: relative_accessibility(areas[mapping[position]],
                                                      wildtype[position - 1])
                     for position in needed}
    records = []
    for low, high in (tuple(pair) for pair in entry['site_pairs']):
        residues = [wildtype[low - 1], wildtype[high - 1]]
        per_model, distances = [], []
        for model in structure['models']:
            other = structure['chains'].get((model, matched['label_asym_id']))
            if other is None or mapping[low] not in other.residues \
                    or mapping[high] not in other.residues:
                continue
            distance = minimum_heavy_atom_distance(other.residues[mapping[low]],
                                                   other.residues[mapping[high]])
            distances.append(distance)
            per_model.append(distance <= HEAVY_ATOM_CONTACT_ANGSTROM)
        primary = distances[0]
        records.append({
            'site_pair': f"{entry['name']}:{low}-{high}",
            'background': entry['name'], 'group': entry['group'],
            'positions': [low, high], 'residues': residues,
            'separation': high - low, 'length': entry['length'],
            'min_heavy_atom_angstrom': primary,
            'contact': bool(primary <= HEAVY_ATOM_CONTACT_ANGSTROM),
            'cb_angstrom': _reported(cb_distance(chain.residues[mapping[low]],
                                                 chain.residues[mapping[high]])),
            'contact_cb': bool(cb_distance(chain.residues[mapping[low]],
                                           chain.residues[mapping[high]])
                               <= CB_CONTACT_ANGSTROM),
            'contact_model_fraction': float(np.mean(per_model)),
            'models_measured': len(per_model),
            'min_heavy_atom_median_angstrom': float(np.median(distances)),
            'rsa': [accessibility[low], accessibility[high]],
            'hydrophobic_count': sum(1 for residue in residues if residue in HYDROPHOBIC),
            'charged_count': sum(1 for residue in residues if residue in CHARGED),
            'hydropathy_mean': float(np.mean([KYTE_DOOLITTLE[residue] for residue in residues])),
            'other_chain_angstrom': _reported(_other_chain_distance(
                structure, chain, [mapping[low], mapping[high]])),
        })
    return {
        'name': entry['name'], 'group': entry['group'], 'kind': entry['kind'],
        'length': entry['length'],
        'structure': {
            'method': structure['method'], 'models': len(structure['models']),
            'entry_id': structure['entry_id'],
            'label_asym_id': matched['label_asym_id'],
            'auth_asym_id': matched['auth_asym_id'],
            'entity_id': matched['entity_id'], 'entity_length': matched['entity_length'],
            'identity_over_wildtype': matched['identity_over_wildtype'],
            'aligned_columns': matched['aligned_columns'],
            'identical_columns': matched['identical_columns'],
            'chain_candidates': matched['chain_candidates'],
            'equal_identity_chains': matched['equal_identity_chains'],
            'observed_residues': len(chain.residues),
            'wildtype_to_entity_offset': (
                mapping[1] - 1 if 1 in mapping and
                all(mapping[position] - position == mapping[1] - 1 for position in mapping)
                else None),
        },
        'site_pairs': records,
    }


# --------------------------------------------------------------------------- #
# The matching scheme. Sequence separation, the wild-type residue class and
# relative solvent accessibility are the confounders a contact contrast must
# balance: separation is a distinct axis of this cohort, burial is what makes a
# pair contact at all, and residue class carries the substitution's own scale.
# Coarsened exact matching is used rather than a fitted propensity score because
# the retained support is 217 site pairs, and a cell with no control is pruned
# and counted rather than extrapolated into.
# --------------------------------------------------------------------------- #

#: Separation cells in residues. The frozen cohort's own strata are 1-2, 3-9 and
#: 10+; the wide distant stratum is split once at 20 so a 10-residue and a
#: 60-residue pair are not matched to each other. Separation 1-2 carries no
#: non-contact control anywhere in this cohort and is therefore ineligible.
SEPARATION_CELLS = ((3, 9), (10, 19), (20, 10 ** 6))

#: Relative-solvent-accessibility cell boundary rule and residue-class cells.
#: The accessibility split is the median over the annotated site pairs, which is
#: a covariate marginal and not a function of any endpoint; the class axis is the
#: number of hydrophobic wild-type residues in the pair, coarsened to none
#: against at least one.
RSA_CELL_RULE = 'median of the pair-mean relative accessibility over annotated site pairs'
HYDROPHOBIC_CELLS = ((0, 0), (1, 2))

#: Covariates whose achieved balance is reported before and after weighting.
BALANCE_COVARIATES = ('separation', 'log10_separation', 'rsa_mean', 'rsa_min', 'rsa_max',
                      'hydrophobic_count', 'charged_count', 'hydropathy_mean', 'length')

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923


def covariates(row: dict) -> dict[str, float]:
    """The matching and balance covariates of one annotated site pair."""

    rsa = row['rsa']
    return {'separation': float(row['separation']),
            'log10_separation': float(np.log10(row['separation'])),
            'rsa_mean': float(np.mean(rsa)), 'rsa_min': float(np.min(rsa)),
            'rsa_max': float(np.max(rsa)),
            'hydrophobic_count': float(row['hydrophobic_count']),
            'charged_count': float(row['charged_count']),
            'hydropathy_mean': float(row['hydropathy_mean']),
            'length': float(row['length'])}


def _cell_index(cells, value) -> int:
    for index, (low, high) in enumerate(cells):
        if low <= value <= high:
            return index
    return -1


def matching_cells(rows: list[dict], *, rsa_boundary: float) -> list[tuple | None]:
    """Coarsened cell of each site pair, or ``None`` when it is ineligible."""

    assigned = []
    for row in rows:
        separation = _cell_index(SEPARATION_CELLS, row['separation'])
        hydrophobic = _cell_index(HYDROPHOBIC_CELLS, row['hydrophobic_count'])
        if separation < 0 or hydrophobic < 0:
            assigned.append(None)
            continue
        accessibility = int(float(np.mean(row['rsa'])) > rsa_boundary)
        assigned.append((separation, accessibility, hydrophobic))
    return assigned


def cem_design(rows: list[dict], treated: np.ndarray, *, rsa_boundary: float) -> dict:
    """Coarsened-exact-matching weights for the contact-versus-control contrast.

    The estimand is the average over retained contact site pairs: each retained
    contact carries weight one, and the controls of its cell carry the cell's
    contact count spread equally among them. A cell holding only one arm is
    pruned, and both pruned counts are returned rather than absorbed.
    """

    cells = matching_cells(rows, rsa_boundary=rsa_boundary)
    treated = np.asarray(treated, dtype=bool)
    occupancy: dict[tuple, list[int]] = {}
    for index, cell in enumerate(cells):
        if cell is None:
            continue
        occupancy.setdefault(cell, [0, 0])[0 if treated[index] else 1] += 1
    retained = {cell for cell, (positive, negative) in occupancy.items()
                if positive > 0 and negative > 0}
    weight = np.zeros(len(rows))
    for index, cell in enumerate(cells):
        if cell is None or cell not in retained:
            continue
        positive, negative = occupancy[cell]
        weight[index] = 1.0 if treated[index] else positive / negative
    kept = weight > 0
    return {
        'cells': cells, 'weight': weight, 'retained': kept,
        'retained_cells': sorted(retained),
        'cell_occupancy': {str(list(cell)): occupancy[cell] for cell in sorted(occupancy)},
        'pruned_ineligible_separation': int(sum(
            1 for index, cell in enumerate(cells) if cell is None)),
        'pruned_contacts_without_control': int(sum(
            1 for index, cell in enumerate(cells)
            if cell is not None and cell not in retained and treated[index])),
        'pruned_controls_without_contact': int(sum(
            1 for index, cell in enumerate(cells)
            if cell is not None and cell not in retained and not treated[index])),
    }


def kish(weight: np.ndarray) -> float:
    """Kish effective count of a weight vector: ``(sum w)^2 / sum w^2``."""

    weight = np.asarray(weight, dtype=float)
    total = weight.sum()
    return float(total ** 2 / (weight ** 2).sum()) if total > 0 else 0.0


def group_equal_weight(weight: np.ndarray, groups: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """Rescale an arm's weights so every contributing group carries equal weight.

    Applied to each arm separately, so the contact arm's group composition and
    the control arm's are each flattened. This is the declared sensitivity to the
    primary site-pair-equal weighting, not a second primary. In a bootstrap,
    groups must identify sampled occurrences, so repeated draws of one source
    group each retain one unit of arm weight.
    """

    out = np.array(weight, dtype=float)
    for side in (True, False):
        rows = np.flatnonzero((np.asarray(treated, dtype=bool) == side) & (out > 0))
        if not len(rows):
            continue
        for group in set(groups[rows]):
            inside = rows[groups[rows] == group]
            out[inside] /= out[inside].sum()
    return out


def weighted_difference(values: np.ndarray, weight: np.ndarray,
                        treated: np.ndarray) -> dict:
    """Weighted contact mean minus weighted control mean, with effective counts."""

    values = np.asarray(values, dtype=float)
    weight = np.asarray(weight, dtype=float)
    treated = np.asarray(treated, dtype=bool)
    out = {}
    for name, mask in (('contact', treated & (weight > 0)),
                       ('control', ~treated & (weight > 0))):
        if not mask.any():
            return {'difference': None, 'contact': None, 'control': None,
                    'effective_contact': 0.0, 'effective_control': 0.0,
                    'contact_site_pairs': int((treated & (weight > 0)).sum()),
                    'control_site_pairs': int((~treated & (weight > 0)).sum())}
        share = weight[mask] / weight[mask].sum()
        out[name] = float((share * values[mask]).sum())
        out[f'effective_{name}'] = kish(weight[mask])
        out[f'{name}_site_pairs'] = int(mask.sum())
    out['difference'] = out['contact'] - out['control']
    return out


def balance_table(rows: list[dict], weight: np.ndarray, treated: np.ndarray,
                  eligible: np.ndarray) -> dict:
    """Standardized mean differences before and after weighting.

    The standardizer is the pooled unweighted standard deviation over the
    eligible site pairs, so the before and after numbers are on one scale.
    """

    treated = np.asarray(treated, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    table = {}
    for name in BALANCE_COVARIATES:
        values = np.asarray([covariates(row)[name] for row in rows], dtype=float)
        inside = np.flatnonzero(eligible)
        left, right = values[inside][treated[inside]], values[inside][~treated[inside]]
        if not len(left) or not len(right):
            continue
        pooled = float(np.sqrt((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2)) \
            if len(left) > 1 and len(right) > 1 else 0.0
        kept = np.flatnonzero(weight > 0)
        matched_left = kept[treated[kept]]
        matched_right = kept[~treated[kept]]
        before = float(left.mean() - right.mean())
        after = float(
            (weight[matched_left] * values[matched_left]).sum() / weight[matched_left].sum()
            - (weight[matched_right] * values[matched_right]).sum() / weight[matched_right].sum())
        table[name] = {
            'contact_mean_matched': float(
                (weight[matched_left] * values[matched_left]).sum() / weight[matched_left].sum()),
            'control_mean_matched': float(
                (weight[matched_right] * values[matched_right]).sum() / weight[matched_right].sum()),
            'difference_before': before, 'difference_after': after,
            'standardized_before': before / pooled if pooled > 0 else None,
            'standardized_after': after / pooled if pooled > 0 else None,
            'pooled_sd': pooled,
        }
    return table


def two_stage_bootstrap(rows: list[dict], values: np.ndarray, treated: np.ndarray,
                        *, rsa_boundary: float, group_equal: bool = False,
                        draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
                        minimum_units: int = MINIMUM_BOOTSTRAP_UNITS) -> dict:
    """Percentile interval on the matched difference, resampled group then site pair.

    The outer draw is over held groups, which is the unit the repository's floor
    is declared on, and the inner draw is over the site pairs inside each drawn
    group. Both arms of a drawn group travel together -- a contact site pair and
    a non-contact site pair of the same background are drawn in the same stage --
    and the matching weights are then re-derived inside the draw, so the control
    arm follows the contacts the draw actually contains instead of carrying
    weights fitted on the observed sample. Weights fixed outside the loop would
    understate the matching's own variability, and drawing the two arms
    independently would break the pairing the matching creates.

    Repeated outer draws remain separate group occurrences when group-equal
    weights are formed; pooling them under the source label would erase their
    bootstrap multiplicity. Matching cells still use the original covariates.
    The cycle count never enters as a sample size. A draw that leaves an arm
    empty is skipped and counted.
    """

    groups = np.asarray([row['group'] for row in rows])
    labels = sorted(set(groups))
    members = {label: np.flatnonzero(groups == label) for label in labels}
    generator = np.random.default_rng(seed)
    point = weighted_difference(values, _weights(rows, treated, rsa_boundary, group_equal),
                                treated)['difference']
    floor = bootstrap_unit_floor(len(labels), minimum_units=minimum_units)
    if floor['degenerate']:
        return {'point': point, 'interval': None, 'draws': 0, 'skipped_draws': 0,
                'excludes_zero': None, 'unit': 'held group, then site pair inside it',
                **floor}
    sample, skipped = [], 0
    for _ in range(draws):
        drawn = generator.choice(len(labels), size=len(labels), replace=True)
        index = np.concatenate([generator.choice(members[labels[position]],
                                                 size=len(members[labels[position]]),
                                                 replace=True) for position in drawn])
        subset = [rows[position] for position in index]
        occurrences = np.repeat(np.arange(len(drawn)),
                                [len(members[labels[position]]) for position in drawn])
        weight = _weights(subset, treated[index], rsa_boundary, group_equal,
                          groups=occurrences)
        record = weighted_difference(values[index], weight, treated[index])
        if record['difference'] is None:
            skipped += 1
            continue
        sample.append(record['difference'])
    if len(sample) < draws // 2:
        raise ValueError(f'only {len(sample)} of {draws} bootstrap draws carried both arms')
    low, high = np.percentile(sample, [2.5, 97.5])
    return {'point': point, 'interval': [float(low), float(high)],
            'draws': len(sample), 'skipped_draws': skipped,
            'excludes_zero': bool(low > 0 or high < 0),
            'unit': 'held group, then site pair inside it', **floor}


def _weights(rows: list[dict], treated: np.ndarray, rsa_boundary: float,
             group_equal: bool, *, groups: np.ndarray | None = None) -> np.ndarray:
    design = cem_design(rows, treated, rsa_boundary=rsa_boundary)
    weight = design['weight']
    if group_equal:
        if groups is None:
            groups = np.asarray([row['group'] for row in rows])
        weight = group_equal_weight(weight, groups, treated)
    return weight


# --------------------------------------------------------------------------- #
# Additive-response residualizer. Epsilon is adjusted for the measured additive
# prediction y_A + y_B - y_WT by subtracting the mean epsilon in a quantile bin
# of that prediction. Twenty bins is the declared width. The in-sample curve
# uses every cycle it then adjusts. The cross-fit curve for one family group
# uses the other family groups only: the held-out group's cycles enter neither
# the quantile edges nor the bin means, and this function never fits those
# edges on the full sample and then relabels the result.
# --------------------------------------------------------------------------- #

ADDITIVE_RESPONSE_BINS = 20


def _as_aligned(*arrays: np.ndarray) -> list[np.ndarray]:
    converted = [np.asarray(array) for array in arrays]
    if any(array.ndim != 1 for array in converted):
        raise ValueError('residualizer inputs must be one-dimensional')
    if len({len(array) for array in converted}) != 1:
        raise ValueError('residualizer inputs must have the same length')
    return converted


def assign_response_bin(additive: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin index from interior quantile edges, clipped into ``[0, n_bins)``.

    ``searchsorted(..., side='right')`` on the interior edges is the assignment
    the in-sample adjustment uses. A value outside the training range falls in
    an end bin; the outer edges are not a second rule.
    """

    if len(edges) != n_bins + 1:
        raise ValueError('a response curve needs one more edge than bins')
    return np.clip(np.searchsorted(edges[1:-1], np.asarray(additive, dtype=float),
                                   side='right'), 0, n_bins - 1)


def fit_binned_response(additive: np.ndarray, epsilon: np.ndarray, n_bins: int) -> dict:
    """Quantile edges and per-bin mean epsilon, fit on these cycles only.

    An empty bin has response 0. The fit does not look at any cycle the caller
    did not pass, which is what makes a complement passed by the cross-fit a
    training set that excludes the held-out family.
    """

    additive, epsilon = _as_aligned(additive, epsilon)
    additive = np.asarray(additive, dtype=float)
    epsilon = np.asarray(epsilon, dtype=float)
    if n_bins < 2:
        raise ValueError('the additive response uses at least two bins')
    if len(additive) < 2:
        raise ValueError('the additive response needs at least two training cycles')
    if not np.isfinite(additive).all() or not np.isfinite(epsilon).all():
        raise ValueError('the additive response is fit on finite cycles only')
    edges = np.quantile(additive, np.linspace(0.0, 1.0, n_bins + 1))
    bins = assign_response_bin(additive, edges, n_bins)
    response = np.zeros(n_bins)
    for index in range(n_bins):
        rows = bins == index
        response[index] = epsilon[rows].mean() if rows.any() else 0.0
    return {'edges': edges, 'response': response, 'bins': bins, 'n_train': int(len(additive))}


def in_sample_binned_adjustment(additive: np.ndarray, epsilon: np.ndarray,
                                n_bins: int = ADDITIVE_RESPONSE_BINS) -> tuple[np.ndarray, dict]:
    """Epsilon minus the bin mean, with the curve fit on the same cycles."""

    curve = fit_binned_response(additive, epsilon, n_bins)
    adjusted = np.asarray(epsilon, dtype=float) - curve['response'][curve['bins']]
    return adjusted, curve


def training_curve(additive: np.ndarray, epsilon: np.ndarray, groups: np.ndarray,
                   holdout, n_bins: int = ADDITIVE_RESPONSE_BINS) -> dict:
    """Response curve fit on every family group except ``holdout``.

    Raises when the holdout is absent or is the whole sample. The returned
    ``n_train`` is the complement's length, so a caller can see that the
    held-out cycles were not in the fit.
    """

    additive, epsilon, groups = _as_aligned(additive, epsilon, groups)
    holdout_rows = groups == holdout
    if not holdout_rows.any():
        raise ValueError(f'holdout group {holdout!r} is not in the sample')
    if holdout_rows.all():
        raise ValueError('cross-fit cannot train when the holdout is the whole sample')
    curve = fit_binned_response(additive[~holdout_rows], epsilon[~holdout_rows], n_bins)
    if curve['n_train'] != int((~holdout_rows).sum()):
        raise RuntimeError('training curve was not fit on the held-in complement')
    if curve['n_train'] == len(groups):
        raise RuntimeError('training curve was fit on the full sample')
    return curve


def cross_fit_binned_residuals(additive: np.ndarray, epsilon: np.ndarray, groups: np.ndarray,
                               n_bins: int = ADDITIVE_RESPONSE_BINS) -> np.ndarray:
    """Leave-one-family-group-out residuals of epsilon on the measured additive.

    For each family group the quantile edges and the bin means are fit on the
    other groups, and only then applied to the held-out group's cycles. A group
    whose own epsilon entered its own curve is a failed cross-fit, and this
    function refuses it rather than returning an in-sample residual under a
    cross-fit name.
    """

    additive, epsilon, groups = _as_aligned(additive, epsilon, groups)
    additive = np.asarray(additive, dtype=float)
    epsilon = np.asarray(epsilon, dtype=float)
    labels = sorted(set(groups.tolist()))
    if len(labels) < 2:
        raise ValueError('cross-fit needs at least two family groups')
    residuals = np.empty(len(epsilon), dtype=float)
    for label in labels:
        held = groups == label
        curve = training_curve(additive, epsilon, groups, label, n_bins)
        if curve['n_train'] != int(len(epsilon) - held.sum()):
            raise RuntimeError(f'group {label!r} contributed cycles to its own curve')
        bins = assign_response_bin(additive[held], curve['edges'], n_bins)
        residuals[held] = epsilon[held] - curve['response'][bins]
    return residuals


def _arm_weights(groups: np.ndarray, arm: np.ndarray, group_equal: bool) -> np.ndarray:
    """Equal row weights, optionally flattened so each group in an arm sums to one."""

    weight = np.ones(len(arm), dtype=float)
    if group_equal:
        weight = group_equal_weight(weight, np.asarray(groups), np.asarray(arm, dtype=bool))
    return weight


def resampled_arm_difference(values: np.ndarray, groups: np.ndarray, arm: np.ndarray,
                             *, group_equal: bool = False,
                             draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
                             minimum_units: int = MINIMUM_BOOTSTRAP_UNITS) -> dict:
    """Mean of ``arm`` minus mean of its complement, without coarsened matching.

    The partition is the one the caller passes and is not re-derived inside a
    draw. The outer resample is the family group and the inner resample is the
    row inside it; group-equal weights, when requested, are re-derived on the
    drawn rows separately for each sampled group occurrence, preserving outer
    multiplicities. A draw that leaves an arm empty is skipped and counted. This is
    the interval for a stratum the matcher deleted, where there is no control
    cell to re-match.
    """

    values, groups, arm = _as_aligned(values, groups, np.asarray(arm, dtype=bool))
    values = np.asarray(values, dtype=float)
    arm = np.asarray(arm, dtype=bool)
    if not arm.any() or arm.all():
        raise ValueError('the arm contrast needs rows on both sides')
    labels = sorted(set(groups.tolist()))
    members = {label: np.flatnonzero(groups == label) for label in labels}
    point = weighted_difference(values, _arm_weights(groups, arm, group_equal), arm)
    floor = bootstrap_unit_floor(len(labels), minimum_units=minimum_units)
    if floor['degenerate']:
        return {'point': point['difference'], 'interval': None, 'draws': 0,
                'skipped_draws': 0, 'excludes_zero': None,
                'unit': 'held group, then site pair inside it', **floor}
    generator = np.random.default_rng(seed)
    sample, skipped = [], 0
    for _ in range(draws):
        drawn = generator.choice(len(labels), size=len(labels), replace=True)
        index = np.concatenate([
            generator.choice(members[labels[position]],
                             size=len(members[labels[position]]), replace=True)
            for position in drawn])
        occurrences = np.repeat(np.arange(len(drawn)),
                                [len(members[labels[position]]) for position in drawn])
        record = weighted_difference(values[index],
                                    _arm_weights(occurrences, arm[index], group_equal),
                                    arm[index])
        if record['difference'] is None:
            skipped += 1
            continue
        sample.append(record['difference'])
    if len(sample) < draws // 2:
        raise ValueError(f'only {len(sample)} of {draws} bootstrap draws carried both arms')
    low, high = np.percentile(sample, [2.5, 97.5])
    return {'point': point['difference'], 'interval': [float(low), float(high)],
            'draws': len(sample), 'skipped_draws': skipped,
            'excludes_zero': bool(low > 0.0 or high < 0.0),
            'unit': 'held group, then site pair inside it', **floor}
