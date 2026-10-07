"""Label-blind binding sequence-homology units, not evolutionary independence.

Only a full exhaustive run can export the conservative candidate map. The pilot
freezes inputs/rules and times representative actual pairs; it qualifies no cohort.
No phenotype, feature, model, fit, GPU or network is used by this stage.
"""
from __future__ import annotations

import ctypes
import hashlib
import itertools
import json
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..core.amino_acids import AA20
from ..core.family_grouping import Union, blosum62

MIN_ALIGNED_RESIDUES = 30


def sha(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def seqhash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode('ascii')).hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def canonical(sequence: str) -> None:
    if not sequence or not set(sequence) <= set(AA20):
        raise ValueError('nonempty canonical AA20 sequence required; no truncation/conversion')
    if len(sequence) > 1_000_000:
        raise ValueError('sequence exceeds integer-score safety bound')


@dataclass(frozen=True)
class Alignment:
    score: float
    columns: int
    identical: int
    coverage_a: float
    coverage_b: float
    paired_residues: int
    span_a: int
    span_b: int

    @property
    def identity(self) -> float:
        return 100.0 * self.identical / self.columns if self.columns else 0.0

    def family(self) -> bool:
        return (self.identity >= 30 and self.coverage_a >= 80 and self.coverage_b >= 80
                and self.paired_residues >= MIN_ALIGNED_RESIDUES)

    def anchor(self) -> bool:
        return (self.identity >= 50 and self.coverage_a >= 80
                and self.paired_residues >= MIN_ALIGNED_RESIDUES)


class ExactAligner:
    """One serial C recurrence per call; ctypes releases the GIL for <=4 workers.

    O(len(a)*len(b)) time, O(len(b)) memory. Integer scores give exactly the
    frozen float32 recurrence for these integer scores within the safety bound.
    Builds live only in the supplied ignored runtime directory, not beside code.
    """

    def __init__(self, runtime: Path):
        source = Path(__file__).with_suffix('.c')
        runtime = runtime.resolve()
        # Fail closed on a runtime outside an ignored tree.
        check = subprocess.run(['git', 'check-ignore', '-q', str(runtime / 'backend.so')],
                               cwd=source.parents[3], check=False)
        if check.returncode != 0:
            raise ValueError('backend runtime must be under a Git-ignored directory')
        runtime.mkdir(parents=True, exist_ok=True)
        compiler = '/usr/bin/cc'
        flags = ['-O3', '-std=c99', '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared']
        key = hashlib.sha256((sha(source) + sha(Path(compiler).resolve()) + repr(flags)).encode()).hexdigest()
        binary = runtime / f'phenotype-homology-{key}.so'
        command = [compiler, *flags, str(source), '-o', str(binary)]
        if not binary.exists():
            # Unique temporary output avoids exposing partially compiled shared objects.
            temporary = runtime / f'.build-{os.getpid()}-{time.time_ns()}.so'
            try:
                subprocess.run(command[:-1] + [str(temporary)], check=True, capture_output=True, text=True)
                temporary.replace(binary)
            finally:
                temporary.unlink(missing_ok=True)
        self.library = ctypes.CDLL(str(binary))
        self.function = self.library.phenotype_align
        self.function.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double)]
        self.function.restype = ctypes.c_int
        self.table = np.ascontiguousarray(blosum62()[:20, :20], dtype=np.int32)
        self.index = {r: i for i, r in enumerate(AA20)}
        self.encoded: dict[str, bytes] = {}
        self.provenance = dict(source_sha256=sha(source), binary_sha256=sha(binary),
                               compiler=compiler, compiler_sha256=sha(Path(compiler).resolve()),
                               flags=flags, threads_per_call=1,
                               scoring='canonical BLOSUM62; affine 11 + 1*k; frozen tie rules')

    def prepare(self, sequences) -> None:
        # Called before workers start: no concurrent cache writes.
        for sequence in sequences:
            canonical(sequence)
            self.encoded.setdefault(sequence, bytes(self.index[r] for r in sequence))

    def align(self, a: str, b: str) -> Alignment:
        canonical(a)
        canonical(b)
        ca = self.encoded.get(a)
        cb = self.encoded.get(b)
        if ca is None:
            ca = bytes(self.index[r] for r in a)
        if cb is None:
            cb = bytes(self.index[r] for r in b)
        out = (ctypes.c_double * 8)()
        result = self.function(ca, len(a), cb, len(b),
                               self.table.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), out)
        if result:
            raise MemoryError('exact alignment row allocation failed')
        return Alignment(out[0], int(out[1]), int(out[2]), out[3], out[4],
                         int(out[5]), int(out[6]), int(out[7]))


def relation(a: str, b: str) -> str | None:
    if a == b:
        return 'exact'
    if a in b:
        return 'query_contained_in_subject'
    if b in a:
        return 'subject_contained_in_query'
    return None


def adjudication_gate(receipt: dict) -> dict:
    """Require an explicit completed, row-resolved construct-adjudication gate.

    The authoritative receipt declares the admitted row roster, excluded known
    incompatible IDs, unknown IDs retained as pending, and its admitted status.
    Missing/partial metadata is not interpreted as a clean construct screen.
    """
    gate = receipt.get('adjudication_gate')
    if not isinstance(gate, dict) or gate.get('status') != 'complete':
        raise ValueError('completed adjudication_gate metadata required; legacy mapping is not accepted')
    fields = ('qualified_source_row_ids', 'known_incompatible_source_row_ids', 'unknown_source_row_ids')
    for field in fields:
        ids = gate.get(field)
        if (not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids)
                or len(ids) != len(set(ids))):
            raise ValueError(f'adjudication_gate requires unique explicit {field}')
    admitted = set(gate['qualified_source_row_ids'])
    incompatible = set(gate['known_incompatible_source_row_ids'])
    unknown = set(gate['unknown_source_row_ids'])
    if not admitted or admitted & incompatible or not unknown <= admitted:
        raise ValueError('adjudication roster is empty or includes incompatible/unclassified unknown rows')
    if not isinstance(gate.get('qualified_row_status'), str) or not gate['qualified_row_status']:
        raise ValueError('adjudication_gate must declare qualified_row_status')
    admission = gate.get('admission')
    if (not isinstance(admission, dict) or not isinstance(admission.get('path'), str)
            or not isinstance(admission.get('sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', admission['sha256'])):
        raise ValueError('adjudication_gate must bind the phase-one admission path and sha256')
    return gate


def load_mapping(mapping: Path) -> tuple[list[dict], dict[str, str], dict]:
    """Derive the chain/complex roster ONLY from explicitly qualified labels.

    State inventories can retain old or orphan sequences; they are never used
    as the alignment query roster. Counts are checked against this receipt,
    not against the pre-adjudication census.
    """
    receipt = json.loads((mapping / 'receipt.json').read_text())
    gate = adjudication_gate(receipt)
    inventory = [json.loads(line) for line in (mapping / 'row-mapping.jsonl').read_text().splitlines()]
    by_id = {r['source_row_id']: r for r in inventory}
    if len(by_id) != len(inventory):
        raise ValueError('mapping has duplicate source row IDs')
    admitted = set(gate['qualified_source_row_ids'])
    if not admitted <= set(by_id):
        raise ValueError('qualified adjudication roster has missing mapping rows')
    # Guard against the roster silently omitting rows still declared qualified.
    if {r['source_row_id'] for r in inventory if r['status'] == gate['qualified_row_status']} != admitted:
        raise ValueError('qualified mapping status and adjudication roster disagree')
    explicit_conflicts = {r['source_row_id'] for r in inventory
                          if r.get('construct_status') == 'KNOWN_INCOMPATIBLE'}
    if explicit_conflicts & admitted or not explicit_conflicts <= set(gate['known_incompatible_source_row_ids']):
        raise ValueError('known construct conflicts contradict adjudication roster')
    rows = [by_id[key] for key in sorted(admitted)]
    if {r['source_row_id'] for r in rows if r.get('construct_status') == 'UNKNOWN_CONSTRUCT'} != set(gate['unknown_source_row_ids']):
        raise ValueError('unknown construct rows must remain explicit in adjudication metadata')
    chains: dict[str, str] = {}
    for row in rows:
        if not row['chain_states']:
            raise ValueError('validated row has no partner chains')
        mutated_sequence_hash(row)
        for chain in row['chain_states']:
            sequence, h = chain['sequence'], chain['wt_sha256']
            canonical(sequence)
            if seqhash(sequence) != h or chains.setdefault(h, sequence) != sequence:
                raise ValueError('chain hash/sequence mismatch')
    exact = json.loads((mapping / 'exact-components.json').read_text())
    counts = dict(unique_complete_chain_sequences=len(chains),
                  operational_complexes=len({r['complex_id'] for r in rows}),
                  exact_identity_components=len(set(exact['membership'].values())),
                  different_single_variants=len({r['variant_key'] for r in rows}),
                  validated_operational_labels=len(rows))
    if any(receipt.get('counts', {}).get(k) != v for k, v in counts.items()):
        raise ValueError(f'qualified-row census disagrees with adjudicated receipt: {counts}')
    membership = complex_components(rows, {h: h for h in chains})
    if membership != exact['membership']:
        raise ValueError('input exact-sharing map not reproduced')
    manifest = dict(counts=counts, adjudication_gate=gate,
                    roster_scope='only adjudication-qualified label rows and their complete partner chains; no orphan inventory',
                    files={p.name: sha(p) for p in
                    (mapping / 'receipt.json', mapping / 'row-mapping.jsonl', mapping / 'exact-components.json')})
    return rows, dict(sorted(chains.items())), manifest


def load_sources(root: Path) -> tuple[list[dict], dict]:
    """Reuse the Domainome declaration's complete development-source interfaces.

    ProteinGym WT is reconstructed from every retained assay's first mutant row,
    not substituted by the original-reference CSV's possibly broader catalogue.
    Tsuboyama uses every background in the validated 478-record query index.
    """
    from scripts.capability.mutation.declare_external_confirmation import (
        proteingym_targets, tsuboyama_backgrounds)
    pg = root / 'data/proteingym/DMS_ProteinGym_substitutions'
    ts = root / 'results/shared/megascale_disjointness_20260813/query_index.json'
    targets = proteingym_targets(pg)
    backgrounds = tsuboyama_backgrounds(ts)
    if len(targets) != 217 or len(backgrounds) != 478:
        raise ValueError('complete declared development reference must contain 217 assays and 478 backgrounds')
    sources = [dict(source=source, protein=name, sequence=s, sequence_sha256=seqhash(s))
               for source, sequences in [('ProteinGym', targets), ('Tsuboyama', backgrounds)]
               for name, s in sorted(sequences.items())]
    for source in sources:
        canonical(source['sequence'])
    files = {str(p.relative_to(root)): sha(p) for p in [*sorted(pg.glob('*.csv')), ts]}
    return sources, dict(counts=dict(ProteinGym=len(targets), Tsuboyama=len(backgrounds)),
                         files=files, interface='Domainome declaration proteingym_targets/tsuboyama_backgrounds',
                         interface_sha256=sha(root / 'scripts/capability/mutation/declare_external_confirmation.py'),
                         sequence_manifest_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest())


def study_links(mapping: Path, rows: list[dict]) -> tuple[dict[str, list[str]], dict]:
    """Only numeric PubMed reference fields are admitted; ambiguity stays pending."""
    receipt = json.loads((mapping / 'receipt.json').read_text())
    gate = adjudication_gate(receipt)
    admission = gate['admission']
    path = (mapping / admission['path']).resolve()
    expected = admission['sha256']
    if sha(path) != expected:
        raise ValueError('phase-one admission changed since mapping')
    needed = {r['source_row_id']: r['complex_id'] for r in rows}
    by_complex = defaultdict(set)
    pending = []
    seen = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = row['source_row_id']
            if key not in needed:
                continue
            if key in seen or row['complex_id'] != needed[key]:
                raise ValueError('study/mapping join mismatch')
            seen.add(key)
            reference = row['original']['Reference'].strip()
            if re.fullmatch(r'[1-9][0-9]{5,8}', reference):
                by_complex[row['complex_id']].add(reference)
            else:
                pending.append(dict(source_row_id=key, reference=reference))
    if seen != set(needed):
        raise ValueError('study join incomplete')
    return {k: sorted(v) for k, v in sorted(by_complex.items())}, dict(
        input_sha256=expected, rule='single 6–9 digit PubMed ID only; no guessing composite/text references',
        parsed_labels=len(seen)-len(pending), pending_labels=len(pending), pending=pending,
        parsed_studies=len(set(itertools.chain.from_iterable(by_complex.values()))))


def contract(inputs: dict, sources: dict, studies: dict) -> dict:
    return dict(schema='binding_sequence_homology_contract_v2', label_blind=True,
        inputs=inputs, development_sources=sources,
        algorithm=dict(method='exhaustive exact Smith-Waterman', matrix='canonical BLOSUM62',
            gap_cost='11 + 1*k', optimum='first row-major match-state optimum',
            ties='M before X before Y; open before extend; restart at predecessor <= 0',
            identity='identical / all alignment columns', coverage='aligned coordinate span / complete sequence length',
            truncation=False),
        family_rule=dict(identity_percent=30, coverage_a_percent=80, coverage_b_percent=80,
            min_paired_residues=MIN_ALIGNED_RESIDUES, exact_or_either_containment='admitted at any length',
            reason='Domainome has no minimum; 30 paired residues is a new extension safeguard against chance short-peptide matches'),
        anchor_rule=dict(identity_percent=50, query_coverage_percent=80, subject_coverage='unconstrained',
            min_paired_residues=MIN_ALIGNED_RESIDUES, exact_or_either_containment='admitted at any length',
            source_family_overlap='30/80 query, same 30-residue safeguard; report only, not exclusion'),
        short_chains='length <30: exact/containment admitted; other homology unresolved, never claim independence',
        complexes='union when ANY mutated or partner chain shares a chain family; retain preexclusion map',
        role_diagnostic='report frozen chain families represented among mutated chains and complex components linked through mutated roles only; descriptive, never used for exclusion/candidates',
        exclusion='any admitted anchor link excludes WHOLE connected complex component; cascade to every label/variant in it',
        candidate='only completed exhaustive screen, no admitted anchor link AND no unresolved short chain in component; operational sequence-homology unit only',
        study_sensitivity=dict(rule='union preexclusion complexes sharing parsed single PubMed ID',
            parsed_studies=studies['parsed_studies'], pending_labels=studies['pending_labels'],
            limitation='sensitivity only; composite IDs pending; neither closure nor primary map proves all study dependence gone'),
        pending=['source quality and measured whole-construct WT identity/engineering',
                 'reliability and baseline qualification', 'exact native checkpoint/interface/state manifest join'],
        promotion='No fits, inference, score coverage claim or main-text cohort qualification')


def complex_components(rows: list[dict], families: dict[str, str], studies=None) -> dict[str, str]:
    union = Union({r['complex_id'] for r in rows})
    owner = {}
    for row in rows:
        for chain in row['chain_states']:
            family = families[chain['wt_sha256']]
            if family in owner:
                union.join(row['complex_id'], owner[family], 'chain_family_any_partner')
            owner[family] = row['complex_id']
    if studies:
        owner = {}
        for c, ids in sorted(studies.items()):
            for study in ids:
                if study in owner:
                    union.join(c, owner[study], 'study_sensitivity')
                owner[study] = c
    return {c: union.find(c) for c in sorted({r['complex_id'] for r in rows})}


def pair_tasks(chains: dict[str, str], sources: list[dict]):
    names = list(chains)
    for a, b in itertools.combinations(names, 2):
        yield ('family', a, b, chains[a], chains[b])
    for a, sequence in chains.items():
        for i, source in enumerate(sources):
            yield ('anchor', a, str(i), sequence, source['sequence'])


def align_task(aligner: ExactAligner, task: tuple) -> dict:
    kind, a, b, sa, sb = task
    exact = relation(sa, sb)
    alignment = aligner.align(sa, sb)
    raw_family = alignment.identity >= 30 and alignment.coverage_a >= 80 and alignment.coverage_b >= 80
    raw_anchor = alignment.identity >= 50 and alignment.coverage_a >= 80
    family_overlap = alignment.identity >= 30 and alignment.coverage_a >= 80
    return dict(kind=kind, query=a, subject=b, relation=exact,
                statistics=dict(**asdict(alignment), identity=alignment.identity),
                admitted=bool(exact or (alignment.family() if kind == 'family' else alignment.anchor())),
                family_overlap=bool(exact or (family_overlap and alignment.paired_residues >= MIN_ALIGNED_RESIDUES)),
                minimum_suppressed=bool(not exact and (raw_family if kind == 'family' else raw_anchor)
                                        and alignment.paired_residues < MIN_ALIGNED_RESIDUES),
                family_overlap_minimum_suppressed=bool(not exact and family_overlap
                    and alignment.paired_residues < MIN_ALIGNED_RESIDUES),
                short_chain_pending=bool(not exact and min(len(sa), len(sb)) < MIN_ALIGNED_RESIDUES))


def task_plan(chains, sources) -> dict:
    lengths = [len(s) for s in chains.values()]
    background = [len(s['sequence']) for s in sources]
    family_cells = (sum(lengths)**2 - sum(n*n for n in lengths)) // 2
    anchor_cells = sum(lengths) * sum(background)
    return dict(family_pairs=len(lengths)*(len(lengths)-1)//2,
                anchor_pairs=len(lengths)*len(background),
                family_cells=family_cells, anchor_cells=anchor_cells,
                total_cells=family_cells+anchor_cells,
                chain_lengths=dict(min=min(lengths), median=float(np.median(lengths)), max=max(lengths)),
                short_chain_sequences=sum(n < MIN_ALIGNED_RESIDUES for n in lengths),
                source_lengths=dict(min=min(background), median=float(np.median(background)), max=max(background)))


def pilot(aligner, chains, sources, workers: int, sample_per_kind: int = 32) -> dict:
    if not 1 <= workers <= 4 or sample_per_kind < 1:
        raise ValueError('pilot requires 1–4 workers and a positive sample size')
    # Quantiles of actual DP cell counts, including the longest pair in each class.
    tasks = list(pair_tasks(chains, sources))
    samples = []
    for kind in ('family', 'anchor'):
        group = sorted((t for t in tasks if t[0] == kind), key=lambda t: (len(t[3])*len(t[4]), t[:3]))
        indices = np.linspace(0, len(group)-1, min(sample_per_kind, len(group)), dtype=int)
        samples.extend(group[int(i)] for i in indices)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda t: align_task(aligner, t), samples))
    elapsed = time.monotonic()-started
    cells = sum(len(t[3])*len(t[4]) for t in samples)
    rate = cells/elapsed
    plan = task_plan(chains, sources)
    return dict(status='pilot_only_full_alignment_pending_authorization', workers=workers,
                sampling='32 DP-cell quantiles per class, deterministic, including maximum',
                pairs=len(samples), cells=cells, wall_seconds=elapsed, cells_per_second=rate,
                projected_full_seconds=plan['total_cells']/rate,
                projection_limitation='rough throughput projection, not a completion guarantee; mixed length/load/cache overhead may differ',
                plan=plan, records=records)


def mutated_sequence_hash(row: dict) -> str:
    """Require the mapped mutated WT to identify exactly one declared chain."""
    name, h = row.get('mutated_auth_chain'), row.get('wt_sha256')
    matches = [c for c in row['chain_states'] if c.get('auth_chain') == name]
    if not name or not h or len(matches) != 1 or matches[0]['wt_sha256'] != h:
        raise ValueError('qualified row lacks an unambiguous mapped mutated-chain identity')
    return h


def role_diagnostic(rows: list[dict], families: dict[str, str], primary: dict[str, str]) -> dict:
    """Descriptive role-aware closure using existing frozen chain families only.

    This does not realign, weaken edges, infer performance or change the primary
    ANY-partner exclusion map. A family may span chains mutated in some records
    and partner-only in others; the diagnostic uses only actual mutated roles.
    """
    union = Union(primary)
    owner = {}
    mutated = {}
    primary_families = defaultdict(set)
    for row in rows:
        h = mutated_sequence_hash(row)
        family, c = families[h], row['complex_id']
        mutated[h] = family
        primary_families[primary[c]].add(family)
        if family in owner:
            union.join(c, owner[family], 'mutated_chain_family_diagnostic_only')
        owner[family] = c
    membership = {c: union.find(c) for c in sorted(primary)}
    return dict(scope='interpretability diagnostic only; frozen global chain-family labels, not a redefined family criterion',
        mutated_sequence_family_membership=dict(sorted(mutated.items())),
        mutated_role_complex_membership=membership,
        counts=dict(mutated_chain_sequences=len(mutated), mutated_chain_families=len(set(mutated.values())),
                    mutated_role_complex_components=len(set(membership.values())),
                    conservative_any_partner_complex_components=len(set(primary.values()))),
        primary_component_mutated_family_counts={g: len(v) for g, v in sorted(primary_families.items())},
        limitation='shared partner or antibody families can connect very large support networks; closure collapse is not proof that binding contains no information',
        used_for_candidate_or_anchor_exclusion=False, extra_alignments=0)


def export_components(rows, chains, families, hits, studies) -> dict:
    """Conservative cascade and separate short-chain unresolved quarantine."""
    membership = complex_components(rows, families)
    closure = complex_components(rows, families, studies)
    anchor_chains = {r['query'] for r in hits if r['kind'] == 'anchor' and r['admitted']}
    direct = {r['complex_id'] for r in rows if any(c['wt_sha256'] in anchor_chains for c in r['chain_states'])}
    excluded = {membership[c] for c in direct}
    short = {r['complex_id'] for r in rows if any(len(chains[c['wt_sha256']]) < MIN_ALIGNED_RESIDUES for c in r['chain_states'])}
    unresolved = {membership[c] for c in short}
    candidate = {c: g for c, g in membership.items() if g not in excluded and g not in unresolved}
    by_group = defaultdict(list)
    for row in rows:
        by_group[membership[row['complex_id']]].append(row)
    coverage = []
    for g, group in sorted(by_group.items()):
        cs = {r['complex_id'] for r in group}
        coverage.append(dict(component=g, complexes=len(cs), labels=len(group),
            variants=len({r['variant_key'] for r in group}),
            direct_anchor_complexes=sorted(cs & direct),
            cascade_only_complexes=sorted(cs-direct) if g in excluded else [],
            anchor_excluded=g in excluded, short_chain_pending=g in unresolved,
            candidate=g not in excluded and g not in unresolved))
    return dict(scope='operational sequence-homology units, not proved evolutionary independence',
                preexclusion_membership=membership, candidate_holdout_membership=candidate,
                anchor_excluded_components=sorted(excluded), short_chain_pending_components=sorted(unresolved),
                components=coverage, role_diagnostic=role_diagnostic(rows, families, membership),
                study_closure=dict(preexclusion_membership=closure,
                    primary_components=len(set(membership.values())), closure_components=len(set(closure.values())),
                    pending='unparsed study references and dependence beyond shared study IDs'),
                counts=dict(chain_families=len(set(families.values())), complex_components=len(by_group),
                    direct_anchor_complexes=len(direct), anchor_excluded_complexes=sum(g in excluded for g in membership.values()),
                    candidate_complexes=len(candidate), candidate_components=len(set(candidate.values())),
                    candidate_labels=sum(r['complex_id'] in candidate for r in rows),
                    candidate_variants=len({r['variant_key'] for r in rows if r['complex_id'] in candidate})))


def full_run(aligner, rows, chains, sources, studies, out: Path, workers: int) -> dict:
    """Exhaustive pairs, streaming all admitted/suppressed links; never heuristic."""
    if not 1 <= workers <= 4:
        raise ValueError('full alignment requires 1–4 workers')
    union = Union(chains)
    hits = []
    counts = Counter()
    started = time.monotonic()
    with (out / 'pair-evidence.jsonl').open('x') as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        # Bounded chunks avoid Executor.map eagerly submitting the complete Cartesian product.
        tasks = iter(pair_tasks(chains, sources))
        while chunk := list(itertools.islice(tasks, 256)):
            for record in pool.map(lambda t: align_task(aligner, t), chunk):
                kind = record['kind']
                counts[kind + '_pairs'] += 1
                if record['minimum_suppressed']:
                    counts[kind + '_minimum_suppressed'] += 1
                if kind == 'anchor' and record['family_overlap_minimum_suppressed']:
                    counts['anchor_family_overlap_minimum_suppressed'] += 1
                if record['admitted']:
                    counts[kind + ('_exact_containment_links' if record['relation'] else '_homolog_links')] += 1
                    if kind == 'family':
                        union.join(record['query'], record['subject'], record['relation'] or '30/80/80')
                if (record['admitted'] or record['minimum_suppressed'] or
                    kind == 'anchor' and (record['family_overlap'] or record['family_overlap_minimum_suppressed'])):
                    if kind == 'anchor':
                        source = sources[int(record['subject'])]
                        record['source'] = {k: v for k, v in source.items() if k != 'sequence'}
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
                    hits.append(record)
    plan = task_plan(chains, sources)
    if any(counts[k] != plan[k] for k in ('family_pairs', 'anchor_pairs')):
        raise ValueError('exhaustive pair census incomplete; no candidate map exported')
    families = {h: union.find(h) for h in chains}
    result = export_components(rows, chains, families, hits, studies)
    result['chain_family_membership'] = families
    dump(out / 'component-map.json', result)
    # Protein (assay/background) and sequence-family overlap are deliberately separate.
    anchors = [r for r in hits if r['kind'] == 'anchor']
    overlap = {}
    for source in ('ProteinGym', 'Tsuboyama'):
        records = [r for r in anchors if r['source']['source'] == source]
        overlap[source] = dict(
            admitted_source_proteins=sorted({r['source']['protein'] for r in records if r['admitted']}),
            exact_containment_query_chains=sorted({r['query'] for r in records if r['relation']}),
            homolog_query_chains=sorted({r['query'] for r in records if r['admitted'] and not r['relation']}),
            family_overlap_source_proteins=sorted({r['source']['protein'] for r in records if r['family_overlap']}),
            family_overlap_binding_chain_families=sorted({families[r['query']] for r in records if r['family_overlap']}))
    dump(out / 'anchor-overlap.json', overlap)
    return dict(status='exhaustive_alignment_complete_not_main_text_qualified', wall_seconds=time.monotonic()-started,
                workers=workers, counts=dict(counts), component_counts=result['counts'],
                role_diagnostic_counts=result['role_diagnostic']['counts'],
                evidence_sha256=sha(out / 'pair-evidence.jsonl'))
