#!/usr/bin/env python3
"""Does InstructProtein's native family instruction move its generations toward the requested family?

A feasibility probe, not a campaign, and not a widening of EXP-R2-227. The
checkpoint's native conditioning interface was established from primary sources
before this script existed, and the first run was then re-run against the label
form and the scaffold those sources actually specify.

What the sources carry. The paper's Table 6, on the knowledge-instruction
dataset, gives its *Family Generation* row as ``Instruction: Can you provide me
with a protein belonging to the secretoglobin family? Output: Sure, here's a
protein from the secretoglobin family: {protein}.``, and its Figure 10 renders
four family-conditional de novo design prompts of the form ``Instruction: I would
like a protein that is in metallothionein family. Output:`` followed by the
design. The checkpoint's own repository is more precise than the paper about this
direction: its released benchmark driver ``benchmarks/scripts/run_scope_fold_rank.py``
instructs the released checkpoint with exactly

    Instruction: I would like a protein that is in {}.
\n\nOutput: One of the protein that meets the demand is

and its released toy instruction data (``dump/pretrain/raw/instructions.txt``)
carries 1,008 records in that form, every one with a lower-case prose family name
ending in the entry's own type noun (``family``, ``superfamily``, ``domain`` or
``site``), every one followed by the span ``<protein>ƤM...``. The label is
therefore a **prose family name** rather than an accession, and the template
carries an **assistant prefix** that the first run supplied no part of. That is
the request measured here.

The checkpoint emits text positions badly (EXP-R2-151: 92.8% of its probability
mass on the twenty residue tokens at every text position, a clean text NLL of
19.08 against ln(50304) = 10.83). That is a readout failure at text positions.
Here the instruction is **supplied by us and never sampled**: the prompt is the
checkpoint's own released instruction scaffold, and only the protein span is
generated.

One measurement is hard and one is soft. A hit at the release's gathering
threshold is the hard one and is a strict bar, so the score stage reports a
single secondary reading beside it: for every generation, its 5-mer Jaccard
similarity to the staged members of the requested family against its similarity
to the staged members of the mismatched family. It is labelled a soft read
everywhere, it gates nothing, and what it can and cannot establish is written
where it is reported.

One class, one contrast, n = 200 per cell:

* **requested** -- the prompt names class A and the generations are scored for A;
* **mismatched** -- the prompt names class B and the same generations are scored
  for A.

Both cells are one arm, one sampling configuration and one oracle. The oracle is
EXP-R2-227's own: HMMER 3.4 against the pressed Pfam-A release at the release's
gathering thresholds, with the class's Pfam referent derived exactly as that
campaign derives it -- from a disjoint referent draw of the class's own real
Swiss-Prot exemplars, never from the anchor draw and never from a generation.
The per-class instrument anchor (real exemplars against length-matched random
UniRef50) runs **before any generation is scored**, as that campaign requires.

A low hit rate is only readable if the oracle recognises the class at all, which
is what the anchor prices. The class's exemplars are identified by the staged
`protein2ipr`-derived Pfam span table, so the anchor's real side partly
re-establishes the labelling channel rather than pricing class specificity from
an independent channel; that limitation is recorded per class and carried in the
report rather than papered over.

Stages, in order:

``--stage class``      the eligible family census and the seeded draw of the
                       requested and mismatched classes. No GPU, no oracle.
``--stage anchor``     the per-class instrument price, through HMMER and Pfam-A.
``--stage generate``   the only GPU stage: 200 + 200 samples at one frozen
                       configuration, with the generated ids kept.
``--stage score``      the oracle on the generations, the two rates, their
                       difference with an interval, and the residue accounting.

The oracle is staged host-locally and is not in the repository, so it is built
under ``--work`` unless it is already there: if ``--work/hmmer`` and
``--work/pfam`` already hold a built HMMER and a pressed Pfam-A -- a symlink to
an existing build is enough -- nothing is rebuilt. The repository's own
``conditioned_generation`` module supplies the classifier: the class draw, the
referent derivation, the anchor record, the near-duplicate grouping, the rate
contrast and the frozen sampling constants are all imported from it rather than
restated here.

Re-run:

    python scripts/capability/generation/instructprotein_family_generation_probe.py --stage class
    python scripts/capability/generation/instructprotein_family_generation_probe.py --stage anchor
    python scripts/capability/generation/instructprotein_family_generation_probe.py --stage generate --device cuda:4 --dtype bfloat16
    python scripts/capability/generation/instructprotein_family_generation_probe.py --stage score
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Container, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.capability.core import protein_annotations as ci  # noqa: E402
from src.capability.generation import conditioned_generation as cg  # noqa: E402
from src.capability.models import joint_modes as jm  # noqa: E402
from src.capability.core.arms import REPO, SWISSPROT_FASTA, UNIREF50_FASTA  # noqa: E402
from src.capability.core.sequence_metadata import PFAM_RESIDUE_TSV, load_pfam_spans
from src.capability.models.instructprotein_fitness import (  # noqa: E402
    INSTRUCTPROTEIN_ARM,
    load_instructprotein,
)
from src.capability.core.io import write_json  # noqa: E402
from src.capability.generation.near_duplicates import (  # noqa: E402
    RESIDUE_SHINGLE,
    shingles,
)

#: The staged InterPro release's own shapes, so one pass over it can read the
#: entry, its type and name, and its member signatures.
_INTERPRO_ENTRY = re.compile(r'<interpro id="(IPR\d+)"[^>]*\btype="([A-Za-z_]+)"')
_INTERPRO_NAME = re.compile(r"<name>(.*?)</name>")
_INTERPRO_MEMBER = re.compile(r'<db_xref[^>]*\bdb="([^"]+)"[^>]*\bdbkey="([^"]+)"')

#: One released family instruction, up to the assistant prefix. The released file
#: carries a literal backslash-n, which is why the newlines are escaped here.
_RELEASED_FAMILY_INSTRUCTION = re.compile(
    r"Instruction: I would like a protein that is in (.{0,200}?)\.\\n\\nOutput: "
)

PROBE = "instructprotein_family_generation_probe"

DEFAULT_OUT = REPO / "results/R6/instructprotein_family_probe_20260917"
DEFAULT_WORK = REPO.parent / "work/instructprotein_family_generation_probe"

#: The probe's one seed: the class draw, the referent and anchor draws, the
#: length-matched random side, the per-cell sampling seeds and the bootstrap.
#: Declared once and never moved after a rate exists.
PROBE_SEED = 20260916

#: One requested class and one mismatched class, which is the whole contrast.
CLASSES = 2

#: Generations per cell. EXP-R2-227's own sample size, and the only prior
#: generation measurement in the repository is at the same n.
GENERATIONS_PER_CELL = 200

#: UniRef50 records held for the length-matched random side of the anchor. The
#: campaign's own default; 100 targets at a 10% length tolerance have no use for
#: more, and the scan is over the whole corpus either way.
RESERVOIR = 200_000

#: The family-conditioned instruction sentence and the assistant prefix that
#: follows it, verbatim from the released checkpoint repository: the instruction
#: string of ``benchmarks/scripts/run_scope_fold_rank.py``, which is also the form
#: of all 1,008 family records of ``dump/pretrain/raw/instructions.txt``. The
#: blank line between the two is the released one and not the single newline of
#: this repository's generic ``protein_context_template``, which this instruction
#: therefore does not go through; the BOS prefix that declaration protects is
#: still checked at the generate stage.
FAMILY_INSTRUCTION_SENTENCE = "I would like a protein that is in {label}."
ASSISTANT_PREFIX = "One of the protein that meets the demand is"
FAMILY_INSTRUCTION_TEMPLATE = (
    f"Instruction: {FAMILY_INSTRUCTION_SENTENCE}\n\nOutput: {ASSISTANT_PREFIX}"
)

#: The InterPro entry type to the noun the released instruction data puts after
#: the entry's name. Measured against that data rather than chosen here: of its
#: 479 family instructions carrying one label component, 468 (97.70%) are exactly
#: ``tuned_label(entry name, entry type)``, so the label form is the release's own.
LABEL_TYPE_NOUNS = {
    "Family": "family",
    "Homologous_superfamily": "superfamily",
    "Domain": "domain",
    "Repeat": "repeat",
    "Active_site": "active site",
    "Binding_site": "binding site",
    "Conserved_site": "conserved site",
}

#: The staged InterPro release, the same file ``conditioned_generation`` already
#: declares, so the label's naming channel is a staged input like the rest.
INTERPRO_XML = cg.INTERPRO_XML

#: The released toy instruction data, outside the repository and ignored by git,
#: read where it is staged for the label-form measurement alone. Its absence is
#: recorded rather than fatal: no class draw or rate depends on it.
RELEASED_INSTRUCTIONS = (
    REPO
    / "external/literature/repos/instructprotein/dump/pretrain/raw/instructions.txt"
)

#: The secondary read's shingle length, from ``near_duplicates`` rather than chosen
#: here, because that module already declares the residue shingle at which this
#: repository groups protein records.
SOFT_READ_SHINGLE = RESIDUE_SHINGLE

#: Staged members probed per side when the secondary read's instrument is priced.
#: Both sides are probed at the same size, as the two scored pools are drawn to the
#: same size, because a maximum over a larger set is larger.
SOFT_READ_MEMBERS = 100

#: The oracle's own thresholds, imported from the campaign rather than chosen
#: here: Pfam-A at its curated per-family gathering thresholds.
PFAM_THRESHOLD = cg.PFAM_THRESHOLD

#: Pfam's own release, so a family name is the release's and not a paraphrase.
PFAM_ARCHIVE = REPO / "external/references/pfam/Pfam-A.hmm.gz"
PFAM_CHECKSUM = REPO / "external/references/pfam/Pfam-A.hmm.gz.sha256"
HMMER_TARBALL = REPO / "external/tools/hmmer-3.4.tar.gz"

HMMER_BINARY = "bin/hmmscan"


# ----------------------------------------------------------------- primitives


def residue_run(generated_ids: Sequence[int], residue_letters: Mapping[int, str]) -> str:
    """The leading run of declared residue ids, as residues.

    InstructProtein's rendering declares one dedicated token per residue and
    identifies its residues by *identity* rather than by position, so the scored
    span is decidable from the ids themselves. A generation that leaves the
    residue subspace at its first token yields the empty string, which is a
    genuine failure of the interface and stays in the denominator.
    """

    residues: list[str] = []
    for value in generated_ids:
        letter = residue_letters.get(int(value))
        if letter is None:
            break
        residues.append(letter)
    return "".join(residues)


def prompt_for(declaration: jm.JointRendering, label: str) -> str:
    """The string this checkpoint is fed to produce one sample.

    The released family rendering -- instruction sentence, blank line, ``Output:``,
    assistant prefix -- then the declared protein start delimiter, so the sampled
    span begins in the residue subspace rather than at a delimiter the model would
    have to emit. The space before the delimiter is the released driver's
    ``" ".join([instruction, sequence])``.
    """

    if declaration.protein_start is None:
        raise ValueError(f"{declaration.name} declares no protein start delimiter")
    return (
        FAMILY_INSTRUCTION_TEMPLATE.format(label=label)
        + " "
        + declaration.protein_start
    )


def tuned_label(name: str, entry_type: str) -> str:
    """The label string the released instruction data renders for one entry.

    ``{entry name} {type noun}``, lower-cased, with the noun omitted when the name
    already ends in it -- the rule measured to reproduce 468 of the 479
    single-component family labels in that data. An entry type the released form
    does not cover is refused rather than rendered with a noun nothing measured.
    """

    noun = LABEL_TYPE_NOUNS.get(entry_type)
    if noun is None:
        raise ValueError(
            f"InterPro entry type {entry_type!r} is not one the released label form "
            f"covers ({sorted(LABEL_TYPE_NOUNS)}), so the label for {name!r} is not "
            "renderable in the form the checkpoint was tuned on"
        )
    text = name.strip()
    if not text.lower().endswith(noun):
        text = f"{text} {noun}"
    return text.lower()


def interpro_label_channel(
    accessions: Sequence[str], *, path: Path = INTERPRO_XML
) -> dict[str, Any]:
    """The staged InterPro release, read once for the two label facts.

    Returns the entry carrying each asked-for Pfam accession as a member
    signature, and the set of label strings the whole release renders under
    :func:`tuned_label`. The member list is what keeps this one source: the class
    is a Pfam family, and the label naming it is the entry that declares that very
    signature as a member, so the label and the oracle's family are not two
    different objects with a resemblance between them.
    """

    wanted = {str(accession) for accession in accessions}
    entries: dict[str, tuple[str, str, str]] = {}
    labels: set[str] = set()
    entry: str | None = None
    entry_type = ""
    name: str | None = None
    count = 0
    with gzip.open(Path(path), "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            opening = _INTERPRO_ENTRY.search(line)
            if opening:
                entry, entry_type, name = opening.group(1), opening.group(2), None
                count += 1
            if entry is not None and name is None:
                element = _INTERPRO_NAME.search(line)
                if element:
                    found_name = element.group(1)
                    name = found_name
                    if entry_type in LABEL_TYPE_NOUNS:
                        labels.add(tuned_label(found_name, entry_type))
            for database, key in _INTERPRO_MEMBER.findall(line):
                if database != "PFAM":
                    continue
                accession = key.split(".", 1)[0]
                if accession in wanted:
                    entries.setdefault(accession, (entry or "", entry_type, name or ""))
    if not count:
        raise RuntimeError(f"{path} carries no InterPro entry")
    return {"entries": entries, "labels": labels, "n_entries": count}


def tuned_label_reproduction(labels: Container[str], *, path: Path) -> dict[str, Any]:
    """How much of the released family-instruction data this label rule reproduces.

    The evidence for the label form, as a count rather than an assertion. Only
    instructions carrying a single label component are testable -- a compound
    request names several entries and is not one label -- and both numbers are
    reported so the share is readable against the population it was taken over.
    """

    if not Path(path).is_file():
        return {"available": False, "path": str(path), "reason": "not staged on this host"}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    found = _RELEASED_FAMILY_INSTRUCTION.findall(text)
    single = [label for label in found if " and " not in label and ", has" not in label]
    reproduced = sum(1 for label in single if label in labels)
    return {
        "available": True,
        "path": str(path),
        "n_family_instructions": len(found),
        "n_single_component": len(single),
        "n_reproduced": reproduced,
        "share_reproduced": (reproduced / len(single)) if single else None,
    }


def zero_hit_upper_bound_95(n: int) -> float:
    """The exact one-sided 95% upper bound for zero successes in ``n`` trials.

    From ``(1-p)^n = 0.05``. It is reported only where no success was observed,
    because that identity is the case it holds in; a nonzero count needs the beta
    quantile, which nothing here claims.
    """

    return 1.0 - 0.05 ** (1.0 / n)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Set Jaccard, with two empty sets scored as no similarity rather than one."""

    union = left | right
    return len(left & right) / len(union) if union else 0.0


def soft_read(
    sequences_by_condition: Mapping[str, Sequence[str]],
    groups_by_condition: Mapping[str, np.ndarray],
    pools: Mapping[str, Sequence[str]],
    *,
    requested: str,
    mismatched: str,
    seed: int,
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """The secondary reading: is a generation nearer the requested family's members?

    Every generation is scored twice with the same instrument -- the residue
    5-mer Jaccard from ``near_duplicates``, the shingle this repository already
    groups protein records at -- against the staged Swiss-Prot members of the
    requested family and against those of the mismatched family, taking the
    maximum on each side. A generation is *nearer the requested family* when its
    requested-side maximum is strictly larger.

    The pools are drawn to the same size from each side under the probe's seed,
    because the two classes do not carry the same number of staged records and a
    maximum over a larger set is larger. The instrument is priced on the same read
    before any generation is interpreted: real members of each family, compared
    leave-one-out against their own family and against the other, so a null read
    is distinguishable from an instrument that cannot separate the two families at
    all.

    What this establishes: whether the sampled 5-mer content sits nearer the
    requested family's staged members than the mismatched family's, on the same
    contrast the hard read uses. What it does not: the level of the rate is not
    calibrated -- the two pools differ in diversity, so only the requested-minus-
    mismatched difference is readable -- and 5-mer overlap is a composition and
    order statistic, not profile membership. A positive soft read with a zero hard
    read is a composition-level statement and nothing stronger.
    """

    if requested not in pools or mismatched not in pools:
        raise ValueError("both classes need a staged member pool")
    size = min(len(pools[requested]), len(pools[mismatched]))
    if size < 1:
        raise RuntimeError("a class with no staged Swiss-Prot member cannot be read this way")
    drawn = {
        requested: cg.seeded_draw(pools[requested], n=size, seed=seed),
        mismatched: cg.seeded_draw(pools[mismatched], n=size, seed=seed),
    }
    sets = {
        role: [shingles(sequence, unit="residues", length=SOFT_READ_SHINGLE) for sequence in members]
        for role, members in drawn.items()
    }

    flags: dict[str, list[bool]] = {}
    per_condition: dict[str, Any] = {}
    for role, samples in sequences_by_condition.items():
        requested_side: list[float] = []
        mismatched_side: list[float] = []
        for sample in samples:
            if not sample:
                # An empty generation is a failure of the interface, is kept in the
                # denominator and is nearer neither family.
                requested_side.append(0.0)
                mismatched_side.append(0.0)
                continue
            scored = shingles(sample, unit="residues", length=SOFT_READ_SHINGLE)
            requested_side.append(max((jaccard(scored, other) for other in sets[requested]), default=0.0))
            mismatched_side.append(max((jaccard(scored, other) for other in sets[mismatched]), default=0.0))
        flags[role] = [
            left > right for left, right in zip(requested_side, mismatched_side, strict=True)
        ]
        n_nearer = int(sum(flags[role]))
        per_condition[role] = {
            "n": len(flags[role]),
            "n_nearer_requested": n_nearer,
            "nearer_requested_rate_grouped": cg.grouped_rate(flags[role], groups_by_condition[role]),
            "mean_similarity_requested": float(np.mean(requested_side)),
            "mean_similarity_mismatched": float(np.mean(mismatched_side)),
            "mean_difference_requested_minus_mismatched": float(
                np.mean(np.asarray(requested_side) - np.asarray(mismatched_side))
            ),
            "zero_hit_upper_bound_95": (
                zero_hit_upper_bound_95(len(flags[role])) if n_nearer == 0 else None
            ),
        }

    contrast = cg.two_sample_rate_contrast(
        flags[requested],
        groups_by_condition[requested],
        flags[mismatched],
        groups_by_condition[mismatched],
        resamples=resamples,
        seed=bootstrap_seed,
    )

    instrument: dict[str, Any] = {}
    for role, other in ((requested, mismatched), (mismatched, requested)):
        probes = cg.seeded_draw(pools[role], n=min(SOFT_READ_MEMBERS, len(pools[role])), seed=seed)
        own_side: list[float] = []
        other_side: list[float] = []
        for sequence in probes:
            scored = shingles(sequence, unit="residues", length=SOFT_READ_SHINGLE)
            own_side.append(
                max(
                    (jaccard(scored, member) for member in sets[role] if member != scored),
                    default=0.0,
                )
            )
            other_side.append(
                max((jaccard(scored, member) for member in sets[other]), default=0.0)
            )
        instrument[role] = {
            "n": len(probes),
            "mean_similarity_own_family": float(np.mean(own_side)),
            "mean_similarity_other_family": float(np.mean(other_side)),
            "mean_difference_own_minus_other": float(
                np.mean(np.asarray(own_side) - np.asarray(other_side))
            ),
            "own_family_nearer_rate": float(
                np.mean([left > right for left, right in zip(own_side, other_side, strict=True)])
            ),
        }

    return {
        "shingle_length": SOFT_READ_SHINGLE,
        "member_pool_size_per_side": size,
        "staged_records_per_side": {
            requested: len(pools[requested]),
            mismatched: len(pools[mismatched]),
        },
        "per_condition": per_condition,
        "contrast_nearer_requested": contrast,
        "difference_in_differences": (
            per_condition[requested]["mean_difference_requested_minus_mismatched"]
            - per_condition[mismatched]["mean_difference_requested_minus_mismatched"]
        ),
        "difference_in_differences_note": (
            "a point estimate with no interval: the interval-bearing soft statistic "
            "is contrast_nearer_requested, which is the same estimator the hard read "
            "uses"
        ),
        "instrument_price": instrument,
        "instrument_price_note": (
            "real staged members of each family read by the same 5-mer rule, "
            "leave-one-out against their own family and against the other, before any "
            "generation is interpreted. A soft read of zero is only readable if this "
            "separates the two families on real sequences"
        ),
    }


def eligible_families(
    census: Mapping[str, set[str]], labels: Mapping[str, tuple[str, str]], *, minimum: int
) -> tuple[str, ...]:
    """The canonically ordered admissible class list.

    A class is admissible when the staged span table carries at least
    ``minimum`` Swiss-Prot records for it -- a 100-record referent draw and a
    disjoint 100-record anchor draw is what a class is priced on -- and the Pfam
    release carries a name for it, because the instruction names a family and a
    name nothing publishes is not one.
    """

    admissible = [
        key for key, accessions in census.items() if len(accessions) >= minimum and key in labels
    ]
    return tuple(sorted(admissible))


def class_draw(candidates: Sequence[str], *, seed: int, n: int) -> tuple[str, str]:
    """The requested class and the mismatched class, from one seeded permutation.

    ``seeded_draw`` is the campaign's own rule: a permutation of the canonically
    ordered admissible list, never a prefix of a frequency table and never the
    head of the file. The mismatched class is the next entry of the same
    permutation, which is the two-class case of that campaign's fixed-point-free
    pairing.
    """

    drawn = cg.seeded_draw(candidates, n=n, seed=seed)
    return drawn[0], drawn[1]


# --------------------------------------------------------------------- stages


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _preamble(artifact: str) -> dict[str, Any]:
    return {
        "probe": PROBE,
        "artifact": artifact,
        "created_utc": _timestamp(),
        "is_a_feasibility_probe": (
            "one class, one contrast, n = 200 per cell. This is not EXP-R2-227, it "
            "does not touch that campaign's class queue, compound or results, and no "
            "campaign verdict is issued from it"
        ),
        "readout_failure_is_not_this_estimand": (
            "InstructProtein's text positions are unmeasurable (EXP-R2-151: 92.8% of "
            "the probability mass on the twenty residue tokens at every text position; "
            "clean text NLL 19.08 against ln(50304) = 10.83). This probe supplies the "
            "instruction instead of sampling it and samples only the protein span, so "
            "that readout failure is not the quantity under test"
        ),
    }


def pfam_labels(path: Path) -> dict[str, tuple[str, str]]:
    """Unversioned Pfam accession to ``(NAME, DESC)``, from the release's own file."""

    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} does not exist; the family names come from the release")
    labels: dict[str, tuple[str, str]] = {}
    accession = name = None
    desc = ""
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("NAME  "):
                name = line[6:].strip()
            elif line.startswith("ACC   "):
                accession = line[6:].strip().split(".", 1)[0]
            elif line.startswith("DESC  "):
                desc = line[6:].strip()
            elif line.startswith("HMMER3"):
                if accession is not None:
                    labels[accession] = (name or "", desc)
                accession = name = None
                desc = ""
    if not labels:
        raise RuntimeError(f"{path} carries no profile")
    return labels


def family_census(path: Path) -> dict[str, set[str]]:
    """Accessions per Pfam family, from the staged span table."""

    spans = load_pfam_spans(Path(path))
    census: dict[str, set[str]] = {}
    for accession, entries in spans.items():
        for _, _, family in entries:
            census.setdefault(family, set()).add(accession)
    return census


def class_pool(accessions: Sequence[str], *, path: Path) -> list[str]:
    """The class's real Swiss-Prot sequences, in canonical accession order."""

    sequences = cg.swissprot_sequences(accessions, path=path)
    return [sequences[accession] for accession in sorted(accessions) if accession in sequences]


def run_class(args: argparse.Namespace) -> dict[str, Any]:
    labels = pfam_labels(args.pfam_hmm)
    census = family_census(args.pfam_residue)
    candidates = eligible_families(census, labels, minimum=cg.MIN_CLASS_RECORDS)
    requested, mismatched = class_draw(candidates, seed=args.draw_seed, n=CLASSES)
    declaration = jm.rendering("instructprotein")
    channel = interpro_label_channel((requested, mismatched), path=args.interpro_xml)
    classes = {}
    for role, key in (("requested", requested), ("mismatched", mismatched)):
        entry = channel["entries"].get(key)
        if entry is None:
            raise RuntimeError(
                f"{key} carries a name in the Pfam release but no entry of the staged "
                f"InterPro release declares it as a member signature, so the label "
                "cannot be rendered in the form this checkpoint was tuned on; the draw "
                "stops here rather than substituting a label form of this probe's own"
            )
        entry_id, entry_type, entry_name = entry
        label = tuned_label(entry_name, entry_type)
        classes[role] = {
            "key": key,
            "pfam_name": labels[key][0],
            "pfam_description": labels[key][1],
            "interpro_entry": entry_id,
            "interpro_type": entry_type,
            "interpro_name": entry_name,
            "label": label,
            "n_swissprot_records": len(census[key]),
            "prompt": prompt_for(declaration, label),
        }
    payload = _preamble("instructprotein_family_generation_class")
    payload.update(
        {
            "draw_seed": int(args.draw_seed),
            "draw": {
                "rule": (
                    "seeded permutation of the canonically ordered admissible family list "
                    "(conditioned_generation.seeded_draw); the mismatched class is the "
                    "next entry of the same permutation"
                ),
                "classes": CLASSES,
            },
            "eligibility": {
                "staged_span_table": str(args.pfam_residue),
                "staged_pfam_release": str(args.pfam_hmm),
                "minimum_records": int(cg.MIN_CLASS_RECORDS),
                "minimum_records_reason": (
                    "a class needs a 100-record referent draw and a disjoint 100-record "
                    "anchor draw; a referent fitted on the sequences the anchor is "
                    "priced on would return a real-side rate of one whatever the oracle "
                    "does"
                ),
                "families_in_the_span_table": len(census),
                "families_with_a_name_in_the_release": len(
                    [key for key in census if key in labels]
                ),
                "admissible_families": len(candidates),
            },
            "instruction_template": FAMILY_INSTRUCTION_TEMPLATE,
            "instruction_source": (
                "the released checkpoint repository's own family rendering: "
                "benchmarks/scripts/run_scope_fold_rank.py instructs the released "
                "checkpoint with 'Instruction: I would like a protein that is in "
                "{}.\\n\\nOutput: One of the protein that meets the demand is', and "
                "dump/pretrain/raw/instructions.txt carries 1,008 family records in "
                "that form. The paper carries the same direction at Table 6 (Family "
                "Generation) and Figure 10 (family-instruction-based de novo design) "
                "in the shorter 'I would like a protein that is in {label} family. "
                "Output:' form"
            ),
            "label_source": {
                "rule": (
                    "lower-cased '{InterPro entry name} {type noun}', with the noun "
                    "omitted when the name already ends in it; the entry is the one "
                    "that declares the class's own Pfam accession as a member "
                    "signature, so the label names the family the oracle scores"
                ),
                "type_nouns": dict(LABEL_TYPE_NOUNS),
                "staged_interpro_release": str(args.interpro_xml),
                "entries_in_the_release": int(channel["n_entries"]),
                "label_population": len(channel["labels"]),
                "released_instruction_data": tuned_label_reproduction(
                    channel["labels"], path=args.released_instructions
                ),
            },
            "prompt_rendering": {
                "template": FAMILY_INSTRUCTION_TEMPLATE,
                "protein_start": declaration.protein_start,
                "declared_generic_template": declaration.protein_context_template,
                "deviation": (
                    "the released family rendering separates the instruction from "
                    "'Output:' with a blank line, while joint_modes declares a single "
                    "newline for this checkpoint's generic context. The released "
                    "rendering is the one the checkpoint was tuned on and is what this "
                    "probe supplies; the declaration's BOS prefix is still checked at "
                    "the generate stage"
                ),
            },
            "classes": classes,
        }
    )
    return payload


def _reservoir(args: argparse.Namespace, band: tuple[int, int]) -> list[str]:
    """A cached length band of UniRef50, drawn once under this probe's seed."""

    path = args.work / f"uniref50_reservoir_{band[0]}_{band[1]}_{args.reservoir}_{args.draw_seed}.json"
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if tuple(cached["band"]) != band or int(cached["seed"]) != int(args.draw_seed):
            raise RuntimeError(
                f"{path} was drawn at band {cached['band']} under seed {cached['seed']}, "
                f"not at {list(band)} under {args.draw_seed}"
            )
        return cached["sequences"]
    sequences = cg.uniref50_pool(
        path=args.uniref50,
        size=args.reservoir,
        seed=args.draw_seed,
        min_len=band[0],
        max_len=band[1],
    )
    write_json(path, {"sequences": sequences, "seed": int(args.draw_seed), "band": list(band)})
    return sequences


def _oracle(args: argparse.Namespace) -> tuple[Any, Any]:
    tool = ci.prepare_hmmer(HMMER_TARBALL, args.work / "hmmer")
    database = ci.prepare_pfam(PFAM_ARCHIVE, PFAM_CHECKSUM, args.work / "pfam", tool=tool)
    return tool, database


def run_anchor(args: argparse.Namespace) -> dict[str, Any]:
    payload = _read(args.class_artifact)
    request = payload["classes"]["requested"]
    key = request["key"]
    pool = class_pool(sorted(family_census(args.pfam_residue)[key]), path=args.swissprot)
    if len(pool) < cg.MIN_CLASS_RECORDS:
        raise RuntimeError(
            f"{key} yielded {len(pool)} sequences from the staged Swiss-Prot file, below "
            f"the {cg.MIN_CLASS_RECORDS} a disjoint referent and anchor draw need"
        )
    referent_draw, real_draw = cg.split_draw(pool, seed=args.draw_seed)
    lengths = [len(sequence) for sequence in real_draw]
    band = (
        max(1, int(min(lengths) * (1.0 - cg.LENGTH_MATCH_TOLERANCE))),
        int(max(lengths) * (1.0 + cg.LENGTH_MATCH_TOLERANCE)) + 1,
    )
    reservoir = _reservoir(args, band)
    random_draw = cg.length_matched(real_draw, reservoir, seed=args.draw_seed)

    named: dict[str, str] = {}
    for kind, sequences in (("referent", referent_draw), ("real", real_draw), ("random", random_draw)):
        for index, sequence in enumerate(sequences):
            named[f"{key}|{kind}|{index}"] = sequence

    tool, database = _oracle(args)
    hits, scan = cg.annotate(
        named,
        tool=tool,
        database=database,
        workspace=args.work / "anchor",
        threads=args.hmmscan_threads,
        shards=args.hmmscan_shards,
        label="instructprotein_probe_anchor",
    )
    referent = cg.referent_from_draw(hits, [f"{key}|referent|{i}" for i in range(len(referent_draw))])
    record = {"key": key, "label": request["label"]}
    record.update(
        cg.anchor_record(
            real=cg.assigned(hits, [f"{key}|real|{i}" for i in range(len(real_draw))], referent),
            random=cg.assigned(hits, [f"{key}|random|{i}" for i in range(len(random_draw))], referent),
            referent=referent,
        )
    )
    record["referent_derivation"] = (
        "Pfam families carried by at least "
        f"{cg.REFERENT_FAMILY_SHARE:.2f} of a disjoint {len(referent_draw)}-record draw of the "
        "class's own real exemplars, by this same HMMER/Pfam-A oracle. The exemplars are "
        "identified by the staged protein2ipr-derived Pfam span table, so this real side "
        "re-establishes the labelling channel rather than pricing class specificity from "
        "an independent one; the random side is the part that prices the oracle's "
        "willingness to fire on proteins of the class's length that are not of its class"
    )
    result = _preamble("instructprotein_family_generation_anchor")
    result.update(
        {
            "draw_seed": int(args.draw_seed),
            "class": record,
            "class_admitted": bool(record["admitted"]),
            "draws": {
                "referent": len(referent_draw),
                "real": len(real_draw),
                "random": len(random_draw),
                "real_length_band": [min(lengths), max(lengths)],
            },
            "random_side": {
                "corpus": str(args.uniref50),
                "reservoir_size": int(args.reservoir),
                "reservoir_band": list(band),
                "tolerance": cg.LENGTH_MATCH_TOLERANCE,
                "note": (
                    "real length-matched UniRef50 proteins, which is what EXP-R2-015's "
                    "control was; a synthetic residue string would fail every profile for "
                    "reasons that have nothing to do with class specificity"
                ),
            },
            "hmmscan": scan,
            "hmmer": tool.record(),
            "pfam": database.record(),
            "pfam_threshold": PFAM_THRESHOLD,
        }
    )
    return result


def _sample(
    model: Any, tokenizer: Any, prompt: str, *, n: int, seed: int, batch_size: int
) -> list[tuple[str, list[int], int]]:
    """``n`` sampled continuations with the generated ids kept.

    ``conditioned_generation.sample_continuations`` is the declared door and this
    is its sampling configuration, imported from that module's own constants so
    the two cannot drift. It is re-spelled here for one reason: that function
    returns decoded text only, and the residue accounting this probe owes -- how
    many generated tokens are residue tokens -- needs the ids, which the decode
    does not carry back.

    Returns ``(decoded text with specials kept, generated ids, prompt length)``.
    """

    import torch

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = encoded["input_ids"].to(model.device)
    prompt_length = int(ids.shape[1])
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    rows: list[tuple[str, list[int], int]] = []
    index = 0
    with torch.no_grad():
        while len(rows) < n:
            size = min(batch_size, n - len(rows))
            torch.manual_seed(seed + index)
            generated = model.generate(
                input_ids=ids.repeat(size, 1),
                attention_mask=torch.ones((size, prompt_length), dtype=torch.long, device=ids.device),
                do_sample=True,
                temperature=cg.TEMPERATURE,
                top_p=cg.TOP_P,
                top_k=cg.TOP_K,
                repetition_penalty=cg.REPETITION_PENALTY,
                max_new_tokens=cg.MAX_NEW_TOKENS,
                pad_token_id=pad,
                use_cache=True,
            )
            for row in generated:
                tail = [int(value) for value in row[prompt_length:]]
                rows.append((tokenizer.decode(tail, skip_special_tokens=False), tail, prompt_length))
            index += 1
    return rows[:n]


def run_generate(args: argparse.Namespace) -> dict[str, Any]:
    class_payload = _read(args.class_artifact)
    loaded = load_instructprotein(INSTRUCTPROTEIN_ARM, device=args.device, dtype=args.dtype)
    declaration = loaded.tokenisation.declaration
    residue_letters = {int(value): residue for residue, value in loaded.tokenisation.residue_ids.items()}
    end_id = int(loaded.tokenizer.convert_tokens_to_ids(declaration.protein_end))
    if loaded.tokenizer.convert_ids_to_tokens(end_id) != declaration.protein_end:
        raise RuntimeError(
            f"{declaration.protein_end!r} does not round-trip through this tokenizer; the "
            "terminator this probe counts cannot be identified"
        )
    prefix_ids = jm.prefix_marker_ids(loaded.tokenizer, declaration)

    cells: dict[str, Any] = {}
    for role, block in class_payload["classes"].items():
        prompt = block["prompt"]
        observed_prefix = tuple(jm.encode(loaded.tokenizer, prompt)[: len(prefix_ids)])
        if observed_prefix != prefix_ids:
            raise RuntimeError(
                f"this prompt encodes with prefix {observed_prefix} where the rendering "
                f"declares {prefix_ids}; the declared rendering is not the one under test"
            )
        if not prompt.endswith(declaration.protein_start):
            raise RuntimeError(
                f"this prompt does not end at the declared protein start "
                f"{declaration.protein_start!r}, so the sampled span would not open in "
                "the residue subspace"
            )
        seed = cg.cell_seed(
            seed=args.sampling_seed, arm_name=INSTRUCTPROTEIN_ARM, class_key=block["key"], condition=role
        )
        rows = _sample(
            loaded.model,
            loaded.tokenizer,
            prompt,
            n=args.generations,
            seed=seed,
            batch_size=args.batch_size,
        )
        samples = [residue_run(ids, residue_letters) for _, ids, _ in rows]
        generated_tokens = sum(len(ids) for _, ids, _ in rows)
        residue_tokens = sum(
            sum(1 for value in ids if value in residue_letters) for _, ids, _ in rows
        )
        terminated = sum(1 for _, ids, _ in rows if end_id in ids)
        cells[role] = {
            "key": block["key"],
            "label": block["label"],
            "prompt": prompt,
            "seed": int(seed),
            "samples": samples,
            "raw": [text for text, _, _ in rows],
            "statistics": {
                "n": len(samples),
                "n_empty": int(sum(1 for sample in samples if not sample)),
                "mean_length": float(np.mean([len(sample) for sample in samples])),
                "n_terminated": int(terminated),
                "termination_rate": terminated / len(samples),
                "n_generated_tokens": int(generated_tokens),
                "n_residue_token_ids": int(residue_tokens),
                "residue_id_share": residue_tokens / generated_tokens,
                "residues_per_generated_token": sum(len(s) for s in samples) / generated_tokens,
            },
        }
        print(
            f"{role} {block['key']}: {len(samples)} samples, {cells[role]['statistics']['n_empty']} "
            f"empty, {terminated} reached {declaration.protein_end}",
            flush=True,
        )

    payload = _preamble("instructprotein_family_generation_samples")
    payload.update(
        {
            "arm": INSTRUCTPROTEIN_ARM,
            "checkpoint_facts": loaded.facts,
            "sampling": {
                "top_p": cg.TOP_P,
                "temperature": cg.TEMPERATURE,
                "top_k": cg.TOP_K,
                "repetition_penalty": cg.REPETITION_PENALTY,
                "max_new_tokens": cg.MAX_NEW_TOKENS,
                "sampling_seed": int(args.sampling_seed),
                "per_cell_seed_rule": (
                    "conditioned_generation.cell_seed: sha256 of probe, arm, class and "
                    "condition added to the seed, so the two cells never share a sample"
                ),
                "batch_size": int(args.batch_size),
                "batch_size_note": (
                    "a feasibility parameter, not a scientific one; the per-batch seed "
                    "makes the sample depend on it, so it is recorded with the run"
                ),
                "post_selection_filter": cg.POST_SELECTION_FILTER,
                "post_selection_note": cg.POST_SELECTION_NOTE,
            },
            "terminator": {
                "token": declaration.protein_end,
                "id": end_id,
                "note": (
                    "the declared protein end delimiter. It is not this checkpoint's "
                    "end-of-sequence token, so a sample that reaches it and one that "
                    "runs to the token budget are different objects and both are counted"
                ),
            },
            "residue_space": {
                "residue_tokens": len(residue_letters),
                "scored_target_rule": declaration.scored_target_rule,
                "prefix_marker_ids": list(prefix_ids),
                "note": (
                    "the prompt ends at the declared protein start delimiter, so sampling "
                    "begins in the residue subspace; residues_per_generated_token and "
                    "residue_id_share are how much of the sampled span stayed there"
                ),
            },
            "cells": cells,
        }
    )
    return payload


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    anchor = _read(args.anchor_artifact)
    class_payload = _read(args.class_artifact)
    generations = _read(args.generation_artifact)
    referent = tuple(anchor["class"]["referent"])

    named: dict[str, str] = {}
    names: dict[str, list[str]] = {}
    for role, cell in generations["cells"].items():
        names[role] = []
        for index, sequence in enumerate(cell["samples"]):
            name = f"{role}|{index}"
            names[role].append(name)
            if sequence:
                named[name] = sequence

    tool, database = _oracle(args)
    hits, scan = cg.annotate(
        named,
        tool=tool,
        database=database,
        workspace=args.work / "generations",
        threads=args.hmmscan_threads,
        shards=args.hmmscan_shards,
        label="instructprotein_probe_generations",
    )

    flags: dict[str, list[bool]] = {}
    groups: dict[str, np.ndarray] = {}
    grouping: dict[str, Any] = {}
    sequences: dict[str, list[str]] = {}
    for role, cell in generations["cells"].items():
        flags[role] = cg.assigned(hits, names[role], referent)
        groups[role], grouping[role] = cg.near_duplicate_group_ids(cell["samples"], unit="residues")
        sequences[role] = list(cell["samples"])

    rates = {
        role: {
            "p_grouped": cg.grouped_rate(flags[role], groups[role]),
            "p_ungrouped": float(np.mean([1.0 if value else 0.0 for value in flags[role]])),
            "n": len(flags[role]),
            "n_hits": int(sum(flags[role])),
            "n_with_any_family": int(sum(1 for name in names[role] if cg.families_of(hits, name))),
        }
        for role in flags
    }
    for block in rates.values():
        # The exact one-sided 95% bound for zero successes in n trials. It is
        # reported only where no hit was observed, because that identity is the
        # case it holds in; a nonzero count needs the beta quantile, which this
        # probe does not add.
        block["zero_hit_upper_bound_95"] = (
            zero_hit_upper_bound_95(block["n"]) if block["n_hits"] == 0 else None
        )
    contrast = cg.two_sample_rate_contrast(
        flags["requested"],
        groups["requested"],
        flags["mismatched"],
        groups["mismatched"],
        resamples=args.bootstrap,
        seed=args.bootstrap_seed,
    )

    census = family_census(args.pfam_residue)
    pools = {
        role: class_pool(sorted(census[block["key"]]), path=args.swissprot)
        for role, block in class_payload["classes"].items()
    }
    soft = soft_read(
        sequences,
        groups,
        pools,
        requested="requested",
        mismatched="mismatched",
        seed=args.draw_seed,
        resamples=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    soft_contrast = soft["contrast_nearer_requested"]
    soft_lower_bound_positive = (
        None if soft_contrast["ci95"] is None else bool(soft_contrast["ci95"][0] > 0.0)
    )
    soft_instrument_separates = bool(
        soft["instrument_price"]["requested"]["mean_difference_own_minus_other"] > 0.0
        and soft["instrument_price"]["mismatched"]["mean_difference_own_minus_other"] > 0.0
    )
    if not soft_instrument_separates:
        soft_reading = "not_interpretable_soft_instrument_does_not_separate_the_two_families"
    elif soft_lower_bound_positive is None:
        soft_reading = "not_scored_soft_interval_degenerate"
    elif soft_lower_bound_positive:
        soft_reading = "soft_read_separates_the_two_conditions"
    else:
        soft_reading = "soft_read_does_not_separate_the_two_conditions"

    statistics = {role: cell["statistics"] for role, cell in generations["cells"].items()}
    pooled_tokens = sum(block["n_generated_tokens"] for block in statistics.values())
    pooled_residue_ids = sum(block["n_residue_token_ids"] for block in statistics.values())
    pooled_extracted = sum(sum(len(sample) for sample in cell["samples"]) for cell in generations["cells"].values())

    admitted = bool(anchor["class_admitted"])
    lower_bound_positive = None if contrast["ci95"] is None else bool(contrast["ci95"][0] > 0.0)
    if not admitted:
        reading = "not_interpretable_anchor_failed"
    elif lower_bound_positive is None:
        reading = "not_scored_interval_degenerate"
    elif lower_bound_positive:
        reading = "requested_label_moves_generation_toward_the_requested_family"
    else:
        reading = "no_measured_movement_toward_the_requested_family"

    payload = _preamble("instructprotein_family_generation_report")
    payload.update(
        {
            "draw_seed": int(anchor["draw_seed"]),
            "class": anchor["class"],
            "class_admitted": admitted,
            "anchor": {
                "referent": list(referent),
                "real_rate": anchor["class"]["real_rate"],
                "random_rate": anchor["class"]["random_rate"],
                "n_real": anchor["class"]["n_real"],
                "n_random": anchor["class"]["n_random"],
                "real_floor": anchor["class"]["real_floor"],
                "random_ceiling": anchor["class"]["random_ceiling"],
                "referent_derivation": anchor["class"]["referent_derivation"],
                "limitation": (
                    "the real side of this anchor and the class's referent share a "
                    "labelling channel (the staged Pfam span table), so the real rate "
                    "re-establishes that the pipeline extracts, writes and searches real "
                    "sequences and that the release recognises the class. It does not "
                    "price class specificity from an independent channel; the random "
                    "side is what prices the oracle's willingness to fire on proteins of "
                    "the class's length that are not of its class"
                ),
            },
            "oracle": {
                "engine": "HMMER 3.4 hmmscan",
                "database": "staged Pfam-A release",
                "hmmer": tool.record(),
                "pfam": database.record(),
                "threshold": PFAM_THRESHOLD,
                "threshold_note": (
                    "Pfam-A's own curated per-family gathering thresholds (--cut_ga), the "
                    "same invocation EXP-R2-227 uses, so a family call is a statement of "
                    "the release and not a cut chosen inside this probe"
                ),
                "hmmscan": scan,
            },
            "sampling": generations["sampling"],
            "values": {
                "p_requested": rates["requested"]["p_grouped"],
                "p_mismatched": rates["mismatched"]["p_grouped"],
                "difference": contrast["difference"],
                "difference_ci95": contrast["ci95"],
                "n_groups": contrast["n_groups"],
                "degenerate": contrast["degenerate"],
                "degenerate_reason": contrast["degenerate_reason"],
                "resampling_unit": "near-duplicate group",
                "interval_note": (
                    "an unpaired percentile bootstrap of the difference between two "
                    "separate samples of 200, clustered on near-duplicate groups, so a "
                    "burst of near-identical generations contributes once. One class is "
                    "one class: no class-clustered interval is available or claimed, and "
                    "at zero hits in both cells the bootstrap interval is the point "
                    "[0, 0], whose content is the same zero the counts already state"
                ),
            },
            "rates_by_condition": rates,
            "rates_by_condition_note": (
                "p_requested and p_mismatched are the rates at which the oracle assigns a "
                "generation to the requested class's referent; n_with_any_family is how "
                "many generations of that cell carry any Pfam family at all, which is "
                "what separates an oracle that recognises nothing from a cell whose "
                "generations are simply not of the requested class; "
                "zero_hit_upper_bound_95 is reported where no hit was observed"
            ),
            "grouping": grouping,
            "termination": {
                role: {
                    "n_terminated": block["n_terminated"],
                    "rate": block["termination_rate"],
                }
                for role, block in statistics.items()
            },
            "residue_space": {
                "per_cell": {
                    role: {
                        "residue_id_share": block["residue_id_share"],
                        "residues_per_generated_token": block["residues_per_generated_token"],
                        "n_generated_tokens": block["n_generated_tokens"],
                    }
                    for role, block in statistics.items()
                },
                "pooled_residue_id_share": pooled_residue_ids / pooled_tokens,
                "pooled_residues_per_generated_token": pooled_extracted / pooled_tokens,
                "pooled_extracted_residues": int(pooled_extracted),
                "note": (
                    "two different quantities, reported together because they answer "
                    "different questions. The residue-id share is of the whole sampled "
                    "span, including any continuation past the terminator, and is what "
                    "shows the span was decoded in the residue space rather than through "
                    "text. Residues per generated token counts only the extracted run, "
                    "which stops at the first terminator, so it is lower wherever a "
                    "sample kept generating after closing its protein block"
                ),
            },
            "reading": reading,
            "reading_note": (
                "one class under one oracle at n = 200 per cell. A positive interval says "
                "the request moved this checkpoint's generations toward the requested "
                "family on this class and this instrument, which is what makes a third "
                "conditioned arm worth considering; it is not a rate this probe has "
                "measured on any other class"
                if admitted
                else "the instrument anchor did not admit this class, so the two rates "
                "above are not readable as a class-selective statement"
            ),
            "soft_read": soft,
            "soft_read_reading": soft_reading,
            "soft_read_note": (
                "SECONDARY, and not a gate. The hard read is a profile hit at the "
                "release's gathering threshold, which a generation carrying family-like "
                "content can miss, so this read asks the weaker question directly: does "
                "a generation's 5-mer content sit nearer the staged members of the "
                "requested family than those of the mismatched family. It establishes "
                "at most that: 5-mer overlap is a composition and order statistic, not "
                "profile membership and not structure, and the level of the rate is not "
                "calibrated because the two pools differ in diversity, so only the "
                "requested-minus-mismatched contrast is readable. A soft separation "
                "with a zero hard read is a weaker and different statement from a family "
                "call, and neither is reported as the other"
            ),
        }
    )
    return payload


# ----------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", required=True, choices=("class", "anchor", "generate", "score"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--pfam-residue", type=Path, default=PFAM_RESIDUE_TSV)
    parser.add_argument("--pfam-hmm", type=Path, default=None, help="the pressed Pfam-A.hmm; defaults to --work/pfam/Pfam-A.hmm")
    parser.add_argument("--interpro-xml", type=Path, default=INTERPRO_XML)
    parser.add_argument("--released-instructions", type=Path, default=RELEASED_INSTRUCTIONS)
    parser.add_argument("--swissprot", type=Path, default=SWISSPROT_FASTA)
    parser.add_argument("--uniref50", type=Path, default=UNIREF50_FASTA)
    parser.add_argument("--class-artifact", type=Path, default=None)
    parser.add_argument("--anchor-artifact", type=Path, default=None)
    parser.add_argument("--generation-artifact", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--generations", type=int, default=GENERATIONS_PER_CELL)
    parser.add_argument("--draw-seed", type=int, default=PROBE_SEED)
    parser.add_argument("--sampling-seed", type=int, default=PROBE_SEED)
    parser.add_argument("--bootstrap", type=int, default=cg.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=PROBE_SEED)
    parser.add_argument("--reservoir", type=int, default=RESERVOIR)
    parser.add_argument("--hmmscan-threads", type=int, default=8)
    parser.add_argument("--hmmscan-shards", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.pfam_hmm is None:
        args.pfam_hmm = args.work / "pfam/Pfam-A.hmm"
    if args.class_artifact is None:
        args.class_artifact = args.out / "class.json"
    if args.anchor_artifact is None:
        args.anchor_artifact = args.out / "anchor.json"
    if args.generation_artifact is None:
        args.generation_artifact = args.out / "generations.json"

    stages = {
        "class": (run_class, args.out / "class.json"),
        "anchor": (run_anchor, args.out / "anchor.json"),
        "generate": (run_generate, args.out / "generations.json"),
        "score": (run_score, args.out / "report.json"),
    }
    function, destination = stages[args.stage]
    payload = function(args)
    write_json(destination, payload)
    print(json.dumps({k: v for k, v in payload.items() if k in ("reading", "class", "classes")}, indent=2, default=str))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
