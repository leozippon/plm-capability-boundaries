"""Orthogonal format, termination and profile outcomes at a residue budget."""
from __future__ import annotations

import hashlib
import re
from collections import Counter

AA = 'ACDEFGHIKLMNPQRSTVWY'


def decompose(raw: str, arm: str, *, residue_budget: int | None = None,
              stop_reason: str | None = None) -> dict:
    """Keep prefix evidence distinct from native format and biological completeness."""
    progen = arm.startswith('progen3')
    terminal = '2<eos>' if progen else '>'
    body, found, _ = raw.partition(terminal)
    compact = re.sub(r'\s+', '', body)
    prefix = re.match(f'[{AA}]*', compact).group()
    eos_seen = '<eos>' in raw if progen else '</s>' in raw
    format_valid = bool(prefix) and prefix == compact
    return {
        'raw_continuation': raw, 'sequence': prefix,
        'sequence_sha256': hashlib.sha256(prefix.encode()).hexdigest(),
        'empty_output': not raw.strip(), 'empty_residue_prefix': not prefix,
        'format_valid': format_valid, 'format_failure': bool(compact) and prefix != compact,
        'native_terminal_observed': bool(found), 'model_eos_observed': eos_seen,
        'native_complete': bool(found) and format_valid,
        'stop_reason': stop_reason or ('native_terminal' if found else 'unknown_without_token_trace'),
        'residue_length': len(prefix), 'residue_budget': residue_budget,
        'residue_budget_overshoot': max(0, len(prefix) - residue_budget) if residue_budget else None,
        'biological_completeness': 'not_established',
    }


def wilson(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/(1+z*z/n)
    return [center-half, center+half]


def summarize(rows: list[dict]) -> dict:
    from .conditioned_generation import near_duplicate_group_ids
    n = len(rows)
    flags = ['empty_output', 'empty_residue_prefix', 'format_failure',
             'native_terminal_observed', 'model_eos_observed', 'native_complete']
    result = {'n_attempts': n, 'outcomes': {}, 'stop_reasons': dict(Counter(r['stop_reason'] for r in rows))}
    for flag in flags:
        k = sum(bool(r[flag]) for r in rows)
        result['outcomes'][flag] = {'count': k, 'denominator': n, 'fraction': k/n if n else None, 'wilson_95': wilson(k,n)}
    annotated = all(r.get('any_profile_hit') is not None for r in rows)
    result['profiles_available'] = annotated
    if annotated:
        groups, policy = near_duplicate_group_ids([r['sequence'] for r in rows], unit='residues')
        for flag, pred in {
            'any_profile_hit': lambda r: r['any_profile_hit'],
            'native_complete_with_profile': lambda r: r['native_complete'] and r['any_profile_hit'],
            'native_complete_without_profile': lambda r: r['native_complete'] and not r['any_profile_hit'],
        }.items():
            k = sum(bool(pred(r)) for r in rows)
            result['outcomes'][flag] = {'count': k, 'denominator': n, 'fraction': k/n if n else None, 'wilson_95': wilson(k,n)}
        result['profile_groups'] = len({int(g) for g,r in zip(groups,rows) if r['any_profile_hit'] and r['sequence']})
        result['complete_profile_groups'] = len({int(g) for g,r in zip(groups,rows) if r['native_complete'] and r['any_profile_hit'] and r['sequence']})
        result['grouping'] = policy
    return result
