#!/usr/bin/env python3
"""Label-free structural annotation of the frozen pairwise cohort's site pairs.

Reads one experimental mmCIF per cohort background, binds each file to its own
SHA-256, aligns the pinned wild-type sequence to the structure's entity sequence,
and emits per site pair the declared contact calls and the confounder covariates
the matched comparison balances: sequence separation, the two wild-type residue
identities and their class, and relative solvent accessibility.

No measured stability, cycle epsilon or model score is read. The cohort file
carries measurements, so this entry point extracts only the label-free fields --
background name, group, wild-type sequence and site-pair indices -- through
``site_pair_plan`` and refuses a cohort whose wild-type state is not consistent
across its own cycles. A contact call or a covariate therefore cannot be a
function of an outcome.

Admission is strict and loud. A background is admitted only when its wild type
aligns at 100.0% identity over its whole length to one polypeptide entity of the
structure and both residues of every site pair are present in the coordinates of
the matched chain; anything else is reported as an excluded background with its
reason rather than annotated from a near match or from a predicted structure.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.interactions.contact_enrichment import (  # noqa: E402
    CB_CONTACT_ANGSTROM, CHARGED, HEAVY_ATOM_CONTACT_ANGSTROM, HYDROPHOBIC,
    KYTE_DOOLITTLE, PROBE_RADIUS, SASA_SPHERE_POINTS, annotate_background,
    file_digest, site_pair_plan)

SCHEMA = 'contact_annotation_v1'


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True,
                        help='frozen pairwise cohort; only its label-free fields are read')
    parser.add_argument('--expect-cohort-sha256', required=True)
    parser.add_argument('--structures', type=Path, required=True,
                        help='directory of <PDB id>.cif.gz files, one per background')
    parser.add_argument('--structure-source', required=True,
                        help='the URL template the structure files were fetched from')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    cohort_sha = digest(args.cohort)
    if cohort_sha != args.expect_cohort_sha256:
        raise SystemExit(f'cohort digest {cohort_sha} does not match the declared value')
    plan = site_pair_plan(json.loads(args.cohort.read_text()))

    backgrounds, excluded = [], []
    for entry in plan['backgrounds']:
        identifier = entry['name'].removesuffix('.pdb')
        path = args.structures / f'{identifier}.cif.gz'
        if not path.is_file():
            excluded.append({'name': entry['name'], 'reason': 'no local experimental structure',
                             'expected_file': path.name})
            continue
        sha256, size = file_digest(path)
        try:
            record = annotate_background(entry, path)
        except ValueError as error:
            excluded.append({'name': entry['name'], 'reason': str(error),
                             'file': path.name, 'file_sha256': sha256})
            continue
        record['structure'].update(file=path.name, file_sha256=sha256, file_bytes=size,
                                   pdb_id=identifier)
        backgrounds.append(record)

    site_pairs = [pair for record in backgrounds for pair in record['site_pairs']]
    report = {
        'schema': SCHEMA,
        'inputs': {'cohort': {'path': str(args.cohort), 'sha256': cohort_sha},
                   'structure_directory': str(args.structures),
                   'structure_source': args.structure_source},
        'definitions': {
            'contact_primary': (
                f'minimum non-hydrogen inter-residue atom distance <= '
                f'{HEAVY_ATOM_CONTACT_ANGSTROM} angstrom in the first deposited model of the '
                'matched chain'),
            'contact_secondary': (
                f'CB-CB distance <= {CB_CONTACT_ANGSTROM} angstrom, CA standing in for glycine, '
                'same model and chain'),
            'contact_ensemble': ('fraction of deposited models in which the primary definition '
                                 'holds; a single-model entry reports 1.0 or 0.0'),
            'accessibility': (
                f'Shrake-Rupley accessible surface area over the isolated matched chain of the '
                f'first model, {SASA_SPHERE_POINTS} golden-spiral points per atom, probe '
                f'{PROBE_RADIUS} angstrom, divided by the Tien et al. 2013 residue maximum'),
            'separation': 'j - i in wild-type residues, the distinct axis contacts are matched on',
            'hydrophobic_set': ''.join(sorted(HYDROPHOBIC)),
            'charged_set': ''.join(sorted(CHARGED)),
            'hydropathy': 'Kyte-Doolittle, reported as achieved balance only',
        },
        'label_independence': (
            'no measured stability, cycle epsilon or model score is read on this path; the '
            'cohort is consumed through site_pair_plan, which copies label-free fields only'),
        'summary': {
            'backgrounds_requested': len(plan['backgrounds']),
            'backgrounds_annotated': len(backgrounds),
            'backgrounds_excluded': len(excluded),
            'site_pairs': len(site_pairs),
            'contacts_primary': sum(1 for pair in site_pairs if pair['contact']),
            'contacts_secondary': sum(1 for pair in site_pairs if pair['contact_cb']),
            'experimental_methods': {
                method: sum(1 for record in backgrounds
                            if record['structure']['method'] == method)
                for method in sorted({record['structure']['method'] for record in backgrounds})},
            'kyte_doolittle_scale_entries': len(KYTE_DOOLITTLE),
        },
        'excluded': excluded,
        'backgrounds': backgrounds,
        'code_sha256': {
            'scripts/capability/interactions/build_contact_annotation.py': digest(Path(__file__)),
            'src/capability/interactions/contact_enrichment.py': digest(ROOT / 'src/capability/interactions/contact_enrichment.py'),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(json.dumps(report['summary'], indent=1))
    if excluded:
        print(json.dumps({'excluded': excluded}, indent=1))


if __name__ == '__main__':
    main()
