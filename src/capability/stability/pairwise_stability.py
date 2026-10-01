"""Draft measured-cycle cohort and pinned plmc parameter interface.

No experiment is launched by importing this module. Group admission is external;
phenotypes never enter sequence-model features or grouping.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np

QC_WIDTH_KCAL_MOL = 0.5
QC_SOURCE = 'https://www.nature.com/articles/s41586-023-06328-6#Fig2'
#: The combined interval alone leaves per-channel fit-bound values inside the
#: accepted rows: 15,116 accepted rows carry deltaG_t and 2,073 carry deltaG_c
#: exactly at the +-15.000 kcal/mol fit bound, and the trypsin channel width
#: reaches 24.3 kcal/mol at its 99th percentile. The label-instrument
#: qualification therefore applies the same 0.5 kcal/mol boundary per channel,
#: which removes every fit-bound value without a value-dependent filter.
QC_WIDTH_COLUMNS = ('deltaG_95CI', 'deltaG_t_95CI', 'deltaG_c_95CI')
#: Schema string every frozen cohort document carries. Declared once here,
#: beside the builder that writes it, and read by every consumer that admits a
#: cohort, so the value cannot drift away from the pinned artefacts.
COHORT_SCHEMA = 'draft_pairwise_stability_v1'
PLMC_COMMIT = '18c9e55e3bd2f14f4968be19a807b401996c929a'


@dataclass
class PlmcParameters:
    """Positive unnormalized log weight: sum h_i + sum_{i<j} J_ij."""
    alphabet: str
    target: str
    positions: np.ndarray
    fields: np.ndarray
    couplings: np.ndarray  # upper-triangle pairs, row-major amino-acid axes

    def score(self, sequence: str) -> float:
        if len(sequence) != len(self.target):
            raise ValueError('sequence must match the modeled focus columns')
        try:
            codes = [self.alphabet.index(a) for a in sequence]
        except ValueError as exc:
            raise ValueError('sequence contains a symbol outside the model alphabet') from exc
        value = sum(float(self.fields[i, a]) for i, a in enumerate(codes))
        k = 0
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                value += float(self.couplings[k, codes[i], codes[j]])
                k += 1
        return value

    def cycle(self, wildtype: str, single_a: str, single_b: str, double: str) -> float:
        validate_cycle(wildtype, single_a, single_b, double)
        return self.score(double) - self.score(single_a) - self.score(single_b) + self.score(wildtype)


def read_plmc(path: Path) -> PlmcParameters:
    """Read pinned plmc OutputParametersFull v2, little-endian int32/float32.

    Layout follows upstream src/plm.c and scripts/read_params.m. Build arithmetic
    may be double precision; upstream always serializes float32 parameters.
    """
    raw = path.read_bytes()
    if len(raw) < 40:
        raise ValueError('truncated plmc header')
    length, q, n, skipped, _ = np.frombuffer(raw, '<i4', count=5)
    if not (0 < length <= 4096 and 0 < q <= 256 and n > 0 and skipped >= 0):
        raise ValueError('invalid plmc dimensions')
    pairs = int(length * (length - 1) // 2)
    expected = 40 + q + 4 * (n + skipped) + length + 4 * length + 8 * length * q + 8 * pairs * q * q
    if len(raw) != expected:
        raise ValueError('plmc byte count differs from declared dimensions')
    offset = 40
    alphabet = raw[offset:offset + q].decode('ascii'); offset += q
    if len(set(alphabet)) != q:
        raise ValueError('duplicate alphabet symbols')
    offset += 4 * (n + skipped)
    target = raw[offset:offset + length].decode('ascii'); offset += length
    positions = np.frombuffer(raw, '<i4', count=length, offset=offset).copy(); offset += 4 * length
    if len(set(positions.tolist())) != length:
        raise ValueError('duplicate focus positions')
    offset += 4 * length * q  # marginals
    fields = np.frombuffer(raw, '<f4', count=length * q, offset=offset).reshape(length, q).copy()
    offset += 4 * length * q + 4 * pairs * q * q
    couplings = np.frombuffer(raw, '<f4', count=pairs * q * q, offset=offset).reshape(pairs, q, q).copy()
    if not np.isfinite(fields).all() or not np.isfinite(couplings).all():
        raise ValueError('nonfinite plmc parameters')
    return PlmcParameters(alphabet, target, positions, fields, couplings)


#: Prefixes of the source ``mut_type`` values that name an insertion or deletion
#: construct. Case is folded first; no substitution code can collide, because a
#: substitution names a residue, then digits, then a residue.
INDEL_MUT_TYPE_PREFIXES = ('ins', 'del')


def indel_construct(mut_type) -> np.ndarray:
    """Whether each ``mut_type`` names an insertion or deletion construct.

    An indel construct's ``aa_seq`` is truncated to the wild type's length in the
    pinned files, so it is length-matched to the substitution states and would
    otherwise enter a substitution cycle as though it were one. On the pinned
    bytes 53,441 of 575,331 otherwise-admitted rows are such constructs, and they
    carry 735 of the 115,546 length-matched states of the cohort's backgrounds,
    722 of those states exclusively. A cycle built on one of them is not a
    measured double substitution, so the substitution support excludes the rows
    rather than length-matching around them.
    """

    import pandas as pd
    folded = pd.Series(mut_type).astype(str).str.lower()
    return folded.str.startswith(INDEL_MUT_TYPE_PREFIXES).to_numpy()


def validate_cycle(wt: str, a: str, b: str, ab: str) -> tuple[int, int]:
    if not (len(wt) == len(a) == len(b) == len(ab)):
        raise ValueError('cycle requires equal-length substitution sequences')
    da = [i for i in range(len(wt)) if a[i] != wt[i]]
    db = [i for i in range(len(wt)) if b[i] != wt[i]]
    if len(da) != 1 or len(db) != 1 or da[0] == db[0]:
        raise ValueError('cycle requires two distinct single substitutions')
    expected = list(wt); expected[da[0]] = a[da[0]]; expected[db[0]] = b[db[0]]
    if ''.join(expected) != ab:
        raise ValueError('double does not match the constituent singles')
    return da[0], db[0]


def separation_stratum(first: int, second: int) -> str:
    """Prespecified sequence-separation stratum of a site pair, 1-based positions.

    Distance 1-2 can share a tripeptide window, distance 3-9 cannot, and distance
    at least 10 is the distant-sequence sensitivity. This is sequence separation
    in residues, not a structural-contact annotation.
    """
    distance = abs(first - second)
    if distance < 1:
        raise ValueError('a cycle needs two distinct sites')
    return '1-2' if distance <= 2 else '3-9' if distance <= 9 else '10+'


def build_cohort(frame, catalogue: dict, grouping: dict, *, cap: int = 128, seed: int = 20260923) -> dict:
    """Build only after an externally admitted final grouping is supplied.

    Catalogue supplies source-derived kind/sequence. Group assignments must cover
    every natural-labelled catalogue member, preventing silent support selection.
    CI bounds are retained row-wise, never divided by sqrt(replicate count).

    Every natural-labelled background that does not reach the frozen cohort is
    retained in ``excluded`` with its reason and its eligible-cycle count, so the
    difference between the admitted support and the available support is readable
    from the output rather than inferred from a count that went missing.
    """
    import pandas as pd
    if cap < 2 or grouping.get('status') != 'admitted' or not grouping.get('provenance'):
        raise ValueError('positive cap and explicitly admitted grouping provenance required')
    groups = grouping['assignments']
    names = {name for name, row in catalogue.items() if row['kind'] == 'natural'}
    if not names.issubset(groups) or any(not isinstance(groups[n], str) or not groups[n] for n in names):
        raise ValueError('final group map must cover every natural-labelled background')
    required = {'WT_name', 'aa_seq', 'mut_type', 'dG_ML', 'name', *QC_WIDTH_COLUMNS,
                *(f'{column}_{bound}' for column in QC_WIDTH_COLUMNS for bound in ('low', 'high'))}
    if not required.issubset(frame.columns):
        raise ValueError(f'missing measurement columns: {sorted(required - set(frame.columns))}')
    data = frame.copy()
    data['_source_row'] = np.arange(len(data))
    data['_y'] = pd.to_numeric(data.dG_ML, errors='coerce')
    accepted = np.isfinite(data._y) & ~indel_construct(data.mut_type)
    for column in QC_WIDTH_COLUMNS:
        width = data[f'{column}_high'] - data[f'{column}_low']
        finite = np.isfinite(data[column]) & np.isfinite(width)
        if not np.allclose(data.loc[finite, column], width[finite], rtol=1e-6, atol=1e-8):
            raise ValueError(f'{column} differs from its upper minus lower bound')
        accepted &= finite & data[column].between(0, QC_WIDTH_KCAL_MOL)
    rows_available = int(len(data))
    data = data[accepted]
    by_background = {name: part for name, part in data.groupby('WT_name', sort=True)}
    candidates, excluded = [], []
    for name in sorted(names):
        rows = by_background.get(name)
        wt = catalogue[name]['sequence']
        if not wt or any(a not in 'ACDEFGHIKLMNPQRSTVWY' for a in wt):
            raise ValueError('catalogue WT must use the canonical amino-acid alphabet')
        if rows is None or rows[(rows.mut_type == 'wt') & (rows.aa_seq == wt)].empty:
            excluded.append({'name': name, 'group': groups[name], 'reason': 'no_accepted_wt_row',
                             'eligible_cycles': 0})
            continue
        measurements = {}
        for sequence, part in rows.groupby('aa_seq', sort=True):
            if len(sequence) != len(wt) or any(a not in 'ACDEFGHIKLMNPQRSTVWY' for a in sequence):
                continue
            measurements[sequence] = {'value': float(np.median(part._y)), 'rows': [
                {'source_row': int(r['_source_row']), 'name': str(r['name']), 'value': float(r['_y']),
                 'low': float(r['deltaG_95CI_low']), 'high': float(r['deltaG_95CI_high'])}
                for r in part.to_dict('records')]}
        singles, doubles = {}, []
        for sequence in measurements:
            diff = tuple((i, a) for i, a in enumerate(sequence) if a != wt[i])
            if len(diff) == 1: singles[diff[0]] = sequence
            elif len(diff) == 2: doubles.append((sequence, diff))
        cycles = []
        for double, changes in doubles:
            if not all(change in singles for change in changes): continue
            a, b = [singles[c] for c in changes]
            i, j = validate_cycle(wt, a, b, double)
            ys = [measurements[s]['value'] for s in [wt, a, b, double]]
            cycles.append({'sequences': [wt, a, b, double], 'positions': sorted([i + 1, j + 1]),
                           'separation': separation_stratum(i + 1, j + 1),
                           'epsilon': ys[3] - ys[1] - ys[2] + ys[0]})
        if len(cycles) < cap:
            excluded.append({'name': name, 'group': groups[name], 'reason': 'below_cap',
                             'eligible_cycles': len(cycles)})
            continue
        # Stable hash draw does not inspect phenotype magnitudes or predictor outputs.
        cycles.sort(key=lambda r: hashlib.sha256(f'{seed}:{name}:{r["sequences"][3]}'.encode()).digest())
        eligible = len(cycles)
        cycles = cycles[:cap]
        used = {s for r in cycles for s in r['sequences']}
        candidates.append({'name': name, 'group': groups[name], 'kind': catalogue[name]['kind'],
                           'eligible_cycles': eligible, 'length': len(wt),
                           'site_pairs': sorted({tuple(r['positions']) for r in cycles}),
                           'separation': _tally(r['separation'] for r in cycles),
                           'sequences': sorted(used),
                           'cycles': cycles, 'measurements': {s: measurements[s] for s in sorted(used)}})
    chosen: dict = {}
    for row in candidates:
        if row['group'] in chosen:
            excluded.append({'name': row['name'], 'group': row['group'],
                             'reason': 'group_already_represented',
                             'eligible_cycles': row['eligible_cycles']})
        else:
            chosen[row['group']] = row
    backgrounds = list(chosen.values())
    return {'schema': COHORT_SCHEMA, 'status': 'draft', 'seed': seed, 'cap': cap,
            'qc_width_kcal_mol': QC_WIDTH_KCAL_MOL, 'qc_source': QC_SOURCE,
            'qc_width_columns': list(QC_WIDTH_COLUMNS),
            'grouping': grouping, 'backgrounds': backgrounds, 'excluded': excluded,
            'summary': _summarise(backgrounds, groups, catalogue, names, rows_available,
                                  int(accepted.sum()))}


def _tally(values) -> dict:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _summarise(backgrounds, groups, catalogue, natural, rows_available, rows_accepted) -> dict:
    """The frozen workload: inference-deduplicated sequences and stratum coverage."""

    sequences = {s for row in backgrounds for s in row['sequences']}
    strata = _tally(cycle['separation'] for row in backgrounds for cycle in row['cycles'])
    coverage = {
        stratum: sorted({row['group'] for row in backgrounds
                         if any(c['separation'] == stratum for c in row['cycles'])})
        for stratum in ('1-2', '3-9', '10+')
    }
    return {
        'rows_available': rows_available, 'rows_accepted': rows_accepted,
        'final_groups_in_map': len(set(groups.values())),
        'natural_groups_in_map': len({groups[n] for n in natural}),
        'design_backgrounds_not_eligible': sum(
            1 for name, row in catalogue.items() if row['kind'] != 'natural'),
        'backgrounds': len(backgrounds), 'groups_covered': len({row['group'] for row in backgrounds}),
        'cycles': sum(len(row['cycles']) for row in backgrounds),
        'distinct_sequences': len(sequences),
        'residues': sum(len(s) for s in sequences),
        'sequences_per_background_sum': sum(len(row['sequences']) for row in backgrounds),
        'site_pairs': sum(len(row['site_pairs']) for row in backgrounds),
        'distinct_site_pairs_across_backgrounds': len(
            {(row['name'], pair) for row in backgrounds for pair in row['site_pairs']}),
        'separation_strata': strata,
        'groups_per_separation_stratum': {k: len(v) for k, v in coverage.items()},
        'length_range': [min(row['length'] for row in backgrounds),
                         max(row['length'] for row in backgrounds)] if backgrounds else None,
    }
