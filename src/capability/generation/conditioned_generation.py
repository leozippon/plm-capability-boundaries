"""The native conditioning interface, exercised as a capability rather than scored.

EXP-R2-227. Every measurement before this one is a
*scoring* measurement -- a likelihood, a context information, a rank correlation.
Nothing in this programme had ever asked a generative model to generate. Two
panel-adjacent checkpoints carry a conditioning interface that was trained into
them and has never been exercised:

* **ZymCTRL's EC tag.** The tag is priced at 1.73 nats on the cohort it was measured
  on, so the tag demonstrably moves the distribution. **Nobody has asked whether
  it moves it in the requested direction.** A leak of that size is equally
  consistent with a tag that selects the right enzyme class and with a tag that
  shifts composition and length while selecting nothing, and the within-arm
  mismatched-label reference below is the one thing that separates them.
* **ProLLaMA Stage 2's superfamily instruction**, declared in
  :data:`src.capability.models.joint_modes.JOINT_RENDERINGS` and never run.

**This is not the retired steering line, and this module says so before any
sequence exists.** Audit section 9.1 retires EC steering, the feature atlas and
the drug-design line, all of which concerned *internal-feature intervention* and
returned a measured 0/8 on significant positive EC classes across three separate
attempts. Nothing here injects a feature, a direction or a coefficient. The
intervention is **the prompt the model was trained to receive**, and the estimand
is behavioural. The retirement is untouched and is not reopened.

The estimand, per class *c* and per conditioned arm, is the rate ``p(c)`` at
which generations produced under the native request for *c* are assigned to *c*
by an external, non-neural oracle, judged against three references measured in
the same run on the same oracle: the within-arm **mismatched-label** negative,
the **unconditioned** floor at comparable scale, and the per-class **real-protein
anchor** that prices the instrument. Every one of those is what
:data:`CONDITIONS` enumerates, and the compound in :func:`compound_verdict` is
the only gate.

The prior generation numbers in this repository -- EXP-R2-013/014, unsteered
ZymCTRL lysozyme at Pfam 0.820 and CLEAN exact EC 0.775, n=200 -- are single
class, pre-discipline, with no near-duplicate grouping, no homology reporting and
no bootstrap. They are **attainability evidence only**: not a baseline, not a
comparison, and not a result this campaign reproduces or supersedes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.arms import (
    AA20,
    CONDITIONING_END,
    CONDITIONING_START,
    N_TO_C_MARKER,
    REPO,
    SWISSPROT_FASTA,
    UNIREF50_FASTA,
    ZYMCTRL_FASTA,
    env_path,
    iter_fasta,
)
from ..models.joint_modes import rendering
from .near_duplicates import SHINGLE_UNITS, near_duplicate_groups
from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS, paired_group_bootstrap

SCHEMA_VERSION = "r2_conditioned_generation_v1"
PRE_REGISTRATION = "EXP-R2-227"

# ----------------------------------------------------------------- the ceiling
#
# Written into the code's own output, not only into the log, because a later
# reader reaches the artefact and not the registration entry.

CEILING: dict[str, str] = {
    "predicted_structure_and_inferred_labels": (
        "predicted structures, pLDDT and sequence-inferred labels cannot demonstrate "
        "functional competence or acquired protein knowledge. A high assignment rate "
        "says the generation matches a profile, an EC predictor or a fold search -- "
        "not that it works"
    ),
    "max_identity_is_a_covariate": (
        "maximum identity to the arm's own retrievable corpus is REPORTED AS A "
        "COVARIATE and is never claimed as novelty. F15 is the reason: an "
        "alignment-level certificate does not exclude profile-level homology, since a "
        "profile search seeded only by an anchor's prefix recruited the same-fold "
        "partner on 56 of 199 triples where the alignment screen found nothing. A "
        "generation no alignment matches may still be inside the family the model was "
        "trained on"
    ),
    "behavioural_not_mechanistic": (
        "this is a behavioural measurement of a prompt interface, not of a mechanism. "
        "It says nothing about where or how the conditioning is implemented, and it "
        "does not reopen internal-feature steering (audit 9.1), the drug-design line "
        "or the wet-lab line. The retired steering attempts intervened on internal "
        "features and returned 0/8; this exercises the native prompt"
    ),
    "not_a_knowledge_claim": (
        "this is not a knowledge claim and is not admissible under audit 7.0; a "
        "measurement of what a model does is itself the result"
    ),
    "the_oracle_bounds_the_reading": (
        "L9: a Pfam or EC channel cannot support a statement carrying more information "
        "than that channel contains, and a superfamily assignment is a homology "
        "statement by construction"
    ),
    "the_cohort_is_not_independent_of_the_arm": (
        "a positive is about these classes on this cohort. The classes are drawn from "
        "the arm's own labelled corpus, so the cohort is not independent of the arm's "
        "training distribution, and this campaign does not correct for that"
    ),
    "cross_arm_rates_are_descriptive": (
        "ZymCTRL and ProLLaMA are asked for different kinds of class through different "
        "oracles; their rates are reported side by side and never differenced"
    ),
    "prior_lysozyme_numbers_are_attainability": (
        "EXP-R2-013/014 (Pfam 0.820, CLEAN exact EC 0.775, n=200, one class) are cited "
        "as attainability only. They are not reproduced, not superseded and not "
        "compared against"
    ),
}

# ------------------------------------------------------------ frozen constants

#: The one sampling configuration for the entire campaign, identical across arms,
#: classes and conditions. No released ``generation_config.json`` on any of these
#: checkpoints declares a sampling hyper-parameter, so nothing here overrides a
#: publisher's choice; ProLLaMA's *inference script* declares its own
#: (temperature 0.2, top-k 40, top-p 0.9, repetition penalty 1.2), which is
#: recorded in the artefact as an observed fact and deliberately not adopted --
#: one configuration across arms is what makes the conditions comparable.
#: It is not tuned after a rate is seen.
TOP_P = 0.95
TEMPERATURE = 1.0
TOP_K = 0
REPETITION_PENALTY = 1.0
MAX_NEW_TOKENS = 400
SAMPLING_SEED = 20260826

#: Generations per (arm x class x condition). Reproduces the sample size of the
#: only prior generation measurement in the repository.
GENERATIONS_PER_CELL = 200

#: **No perplexity-based post-selection filter is applied.** The published
#: ZymCTRL protocol's selection step scores candidates by the model's own
#: likelihood, which would make the endpoint partly a self-selection. If it is
#: ever run it is a separately labelled stratum and never the main estimate.
POST_SELECTION_FILTER = None
POST_SELECTION_NOTE = (
    "no perplexity-based post-selection filter is applied; the published ZymCTRL "
    "protocol's selection step scores candidates by the model's own likelihood, "
    "which would make the endpoint partly a self-selection"
)

#: The class-clustered bootstrap. The resampling unit is the CLASS, so the class
#: count is the sample size. These two numbers are not revised after a result
#: exists.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260826

#: The seeded permutation every draw in this campaign is taken under, matching
#: ``arms.selected_positions``' convention: ``numpy.random.default_rng(seed)``
#: permutes the canonically ordered candidate list and the first ``n`` entries
#: are the draw. Never a prefix of a frequency table, never the head of a file.
DRAW_SEED = 20260826

#: Sixteen classes rather than the eight-unit floor is a deliberate choice about
#: the bootstrap: eight units is the smallest cohort on which a percentile
#: interval is defined at all, and it is not a cohort on which one is informative.
CLASSES_PER_ARM = 16

#: An arm with fewer surviving classes than this is not scored, and the shortfall
#: is reported. The cohort is never topped up with classes drawn after an anchor
#: has been seen.
MINIMUM_CLASSES = MINIMUM_BOOTSTRAP_UNITS

#: A class needs a disjoint referent draw and a disjoint anchor draw, which is
#: why the eligible-class cut is at 200 records and not at 100: an oracle whose
#: class-to-profile map was fitted on the same sequences the anchor prices it on
#: would return a real-side rate of 1 by construction.
REFERENT_DRAW = 100
ANCHOR_DRAW = 100
MIN_CLASS_RECORDS = REFERENT_DRAW + ANCHOR_DRAW

#: Appendix B rule 2 applied to an oracle instead of to a gate: an oracle channel
#: is admitted on a class only if its real-versus-random anchor separates at this
#: pre-declared margin on THAT class. A class that fails is removed from the
#: cohort **before any generation is scored** and is reported as unmeasurable,
#: not as a failing class.
ANCHOR_REAL_FLOOR = 0.70
ANCHOR_RANDOM_CEILING = 0.10

#: A Pfam family enters a class's referent when it is carried by at least this
#: share of the class's referent draw. One declared threshold rather than a
#: hand-picked family list; a class whose referent comes out empty is
#: unmeasurable and is reported as such.
REFERENT_FAMILY_SHARE = 0.25

#: The random side of the anchor. EXP-R2-015 used real length-matched UniRef50
#: proteins, not synthetic residue strings, and this reproduces that: a random
#: *protein* is the control that prices a class-specific profile call, because a
#: synthetic string would fail every profile for reasons that have nothing to do
#: with class specificity.
LENGTH_MATCH_TOLERANCE = 0.10

#: Pfam-A is searched at its own **gathering thresholds**, the curated per-family
#: cut the release ships, rather than at an E-value chosen here. That is what
#: makes "this sequence carries this family" a curated statement instead of a
#: threshold decision taken inside the campaign being measured.
PFAM_THRESHOLD = "gathering"

#: What a generation must be to enter a rate. A generation that decodes to no
#: canonical residue is a genuine failure of the interface and is counted as a
#: non-hit; it is never dropped, because a denominator selected on the outcome is
#: the shape rule 27 names.
MINIMUM_SCORED_RESIDUES = 1

CONDITIONS = ("requested", "mismatched", "unconditioned_floor")

#: The oracle channels this campaign declares. ``clean_ec`` is declared and
#: **never stubbed**: see :func:`instrument_availability`.
ORACLE_CHANNELS = ("pfam_hmmer", "clean_ec")
PRIMARY_ORACLE = "pfam_hmmer"


def require_frozen_parameters(
    *,
    resamples: int,
    bootstrap_seed: int,
    sampling_seed: int,
    generations: int,
    top_p: float,
    temperature: float,
    max_new_tokens: int,
) -> None:
    """Refuse a run whose frozen parameters were moved after registration.

    Raised rather than recorded, because every one of these was fixed before any
    generation existed and a run at another setting is a different measurement
    that must not be written into this campaign's artefact.
    """

    frozen = {
        "resamples": (resamples, BOOTSTRAP_RESAMPLES),
        "bootstrap_seed": (bootstrap_seed, BOOTSTRAP_SEED),
        "sampling_seed": (sampling_seed, SAMPLING_SEED),
        "generations_per_cell": (generations, GENERATIONS_PER_CELL),
        "top_p": (top_p, TOP_P),
        "temperature": (temperature, TEMPERATURE),
        "max_new_tokens": (max_new_tokens, MAX_NEW_TOKENS),
    }
    moved = {name: pair for name, pair in frozen.items() if pair[0] != pair[1]}
    if moved:
        raise ValueError(
            f"{PRE_REGISTRATION} froze these before any generation existed and they "
            f"are not revisable after a rate is seen: "
            + ", ".join(f"{name}={got!r} (frozen {want!r})" for name, (got, want) in moved.items())
        )


# --------------------------------------------------------------- the class draw


def canonical_ec_order(label: str) -> tuple[int, ...]:
    """Sort key for a full EC number, so the candidate list is file-order free."""

    parts = label.split(".")
    if len(parts) != 4:
        raise ValueError(f"{label!r} is not a full four-field EC number")
    key: list[int] = []
    for part in parts:
        digits = re.sub(r"[^0-9]", "", part)
        key.append(int(digits) if digits else -1)
    return tuple(key)


def ec_class_census(path: Path = ZYMCTRL_FASTA) -> dict[str, int]:
    """Records per full EC number in the arm's own declared evaluation cohort.

    The census is over the whole file, not over a length band: the queue is a
    queue of *classes*, and a band would silently reselect which classes exist.
    """

    counts: Counter[str] = Counter()
    for header, _ in iter_fasta(Path(path)):
        fields = header.split("|")
        if len(fields) != 2:
            raise ValueError(f"unexpected EC-labelled header {header!r}")
        counts[fields[1]] += 1
    if not counts:
        raise RuntimeError(f"{path} carries no EC-labelled record")
    return dict(counts)


def seeded_draw(candidates: Sequence[str], *, n: int, seed: int) -> tuple[str, ...]:
    """``n`` items from a canonically ordered candidate list under one seed.

    ``numpy.random.default_rng(seed).permutation`` is the convention
    ``arms.selected_positions`` already uses, so the two draws in this repository
    are the same object. The first ``n`` of the permutation are taken, which is
    what makes the draw independent of the candidate list's own order beyond the
    canonical sort applied by the caller.
    """

    if n < 1:
        raise ValueError("a draw needs at least one class")
    if len(candidates) < n:
        raise RuntimeError(
            f"{len(candidates)} admissible classes cannot supply a draw of {n}; the "
            "realised count is reported and the cohort is never topped up with "
            "classes drawn after an anchor has been seen"
        )
    order = np.random.default_rng(seed).permutation(len(candidates))
    return tuple(candidates[int(index)] for index in order[:n])


def derangement(n: int, *, seed: int) -> tuple[int, ...]:
    """A fixed-point-free permutation of ``0..n-1`` under one seed.

    The mismatched-label reference is defined by a permutation that maps **no
    class to itself**, so a permutation with a fixed point would silently make one
    class its own negative. Rejection sampling from the same generator the draws
    use keeps this reproducible from the seed alone.
    """

    if n < 2:
        raise ValueError("a fixed-point-free permutation needs at least two classes")
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        candidate = rng.permutation(n)
        if not (candidate == np.arange(n)).any():
            return tuple(int(value) for value in candidate)
    raise RuntimeError("no fixed-point-free permutation was drawn; this cannot happen for n >= 2")


@dataclass(frozen=True)
class ClassEntry:
    """One class of a frozen queue, with the label its arm's prompt spells."""

    key: str
    label: str
    n_corpus_records: int
    mismatched_key: str
    mismatched_label: str

    def record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "n_corpus_records": int(self.n_corpus_records),
            "mismatched_key": self.mismatched_key,
            "mismatched_label": self.mismatched_label,
        }


def build_queue(
    entries: Sequence[tuple[str, str, int]], *, seed: int
) -> tuple[ClassEntry, ...]:
    """Attach the frozen mismatched pairing to a drawn class list."""

    mapping = derangement(len(entries), seed=seed)
    return tuple(
        ClassEntry(
            key=key,
            label=label,
            n_corpus_records=count,
            mismatched_key=entries[mapping[index]][0],
            mismatched_label=entries[mapping[index]][1],
        )
        for index, (key, label, count) in enumerate(entries)
    )


def queue_digest(payload: Mapping[str, Any]) -> str:
    """Content digest of a frozen queue: the classes and the draw, nothing else.

    Deliberately excludes the timestamp and the census totals so that rebuilding
    the queue from the same corpus and the same seed reproduces the digest a
    generation run was pinned to.
    """

    material = json.dumps(
        {
            "pre_registration": payload["pre_registration"],
            "draw": payload["draw"],
            "arms": {
                arm: [entry["key"] for entry in block["classes"]]
                for arm, block in sorted(payload["arms"].items())
            },
            "labels": {
                arm: [entry["label"] for entry in block["classes"]]
                for arm, block in sorted(payload["arms"].items())
            },
            "mismatched": {
                arm: [entry["mismatched_key"] for entry in block["classes"]]
                for arm, block in sorted(payload["arms"].items())
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def load_queue(path: Path) -> dict[str, Any]:
    """Read a frozen queue and refuse one that has drifted from its own digest.

    The queue is built before any generation exists and every later stage is read
    against it, so a queue whose contents no longer hash to the digest it carries
    is a different cohort wearing the same name. That is refused here rather than
    discovered in a rate.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = payload.get("digest")
    if not recorded:
        raise ValueError(f"{path} carries no digest; it is not a frozen queue")
    observed = queue_digest(payload)
    if observed != recorded:
        raise ValueError(
            f"{path} hashes to {observed} and declares {recorded}: the frozen class "
            "queue has drifted from the digest a generation run was pinned to. It is "
            "a different cohort under the same name and is refused"
        )
    if payload.get("pre_registration") != PRE_REGISTRATION:
        raise ValueError(
            f"{path} was frozen for {payload.get('pre_registration')!r}, not {PRE_REGISTRATION!r}"
        )
    return payload


def queue_entries(payload: Mapping[str, Any], arm: str) -> tuple[ClassEntry, ...]:
    """The frozen classes of one arm, refusing an arm the queue does not carry."""

    arms = payload["arms"]
    if arm not in arms:
        raise KeyError(f"the frozen queue carries no arm {arm!r}; it carries {sorted(arms)}")
    return tuple(
        ClassEntry(
            key=entry["key"],
            label=entry["label"],
            n_corpus_records=int(entry["n_corpus_records"]),
            mismatched_key=entry["mismatched_key"],
            mismatched_label=entry["mismatched_label"],
        )
        for entry in arms[arm]["classes"]
    )


# ------------------------------------------------- the superfamily label space

#: ProLLaMA's own released label vocabulary. Its digest is checked rather than
#: assumed: the prompt's label space is part of the measurement's identity.
PROLLAMA_SUPERFAMILIES = env_path(
    "TRANSFER_PROLLAMA_SUPERFAMILIES",
    REPO / "external/literature/repos/prollama/superfamilies.txt",
)
PROLLAMA_SUPERFAMILIES_SHA256 = (
    "5fc21a5353654efce0ef65f6d792b26fa30d51edef736c01df51665747c7d87c"
)

#: The staged InterPro release. ``entry.list`` names the homologous superfamilies;
#: ``interpro.xml.gz`` carries each entry's member signatures, which is how a
#: superfamily reaches a set of proteins at all.
INTERPRO_XML = env_path("TRANSFER_INTERPRO_XML", REPO / "data/interpro/interpro.xml.gz")

_SUPERFAMILY_LINE = re.compile(r"^Superfamily=<(.+)>$")
_INTERPRO_ENTRY = re.compile(r'<interpro id="(IPR\d+)"[^>]*type="([A-Za-z_]+)"')
_CATHGENE3D_MEMBER = re.compile(r'<db_xref[^>]*db="CATHGENE3D"[^>]*dbkey="G3DSA:([0-9.]+)"')

#: **Measured on the staged release, not assumed: no InterPro
#: ``Homologous_superfamily`` entry declares a PFAM member signature.** All 3,510
#: of them carry structural signatures only (CATH-Gene3D / SSF). So the literal
#: reading of "carries at least one Pfam-A member" admits nothing, and the
#: class-to-profile map for this arm has to be built the way the EC arm's is:
#: from a **disjoint referent draw of the superfamily's own real exemplars**,
#: measured by the same HMMER/Pfam-A oracle. That is the same oracle, not a
#: substitute one, and the admissibility requirement it replaces is enforced at
#: the anchor instead -- a superfamily whose referent draw yields no Pfam family
#: is unmeasurable and is reported as such. The measured count reaches the
#: artefact so a reader sees why the route was taken.
SUPERFAMILY_PFAM_MEMBER_NOTE = (
    "no InterPro Homologous_superfamily entry in the staged release declares a PFAM "
    "member signature (measured: 0 of 3510), so a superfamily's Pfam referent is "
    "derived from a disjoint referent draw of its own Swiss-Prot exemplars by the "
    "same HMMER/Pfam-A oracle, exactly as the EC arm's referent is. A superfamily "
    "whose referent draw yields no Pfam family is reported unmeasurable rather than "
    "assigned a family set nothing measured"
)


def single_superfamily_labels(path: Path = PROLLAMA_SUPERFAMILIES) -> tuple[str, ...]:
    """The labels naming ONE superfamily, in the file's own order, deduplicated.

    A comma-joined combination is a different request from a single superfamily
    and has no single InterPro entry to price the oracle against, so it is
    excluded here rather than resolved by taking the first component.
    """

    seen: dict[str, None] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = _SUPERFAMILY_LINE.match(line)
        if match is None:
            raise ValueError(f"unexpected superfamily line {line!r}")
        body = match.group(1)
        if "," in body:
            continue
        seen.setdefault(body, None)
    if not seen:
        raise RuntimeError(f"{path} carries no single-superfamily label")
    return tuple(seen)


def homologous_superfamily_entries(path: Path) -> dict[str, str]:
    """Entry name to InterPro accession for every homologous superfamily."""

    entries: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:3] != ["ENTRY_AC", "ENTRY_TYPE", "ENTRY_NAME"]:
            raise ValueError(f"{path} does not carry InterPro's entry.list header")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"malformed entry.list row {line!r}")
            if fields[1] == "Homologous_superfamily":
                entries[fields[2]] = fields[0]
    if not entries:
        raise RuntimeError(f"{path} carries no Homologous_superfamily entry")
    return entries


def interpro_cath_members(path: Path = INTERPRO_XML) -> tuple[dict[str, frozenset[str]], int]:
    """CATH-Gene3D signatures per InterPro entry, and the PFAM-member count.

    Returns the member map and how many homologous-superfamily entries declared a
    PFAM member signature -- the number :data:`SUPERFAMILY_PFAM_MEMBER_NOTE`
    reports, measured on the staged file rather than quoted.
    """

    import gzip

    members: dict[str, set[str]] = {}
    pfam_bearing_superfamilies = 0
    current: str | None = None
    current_type: str | None = None
    inside = False
    saw_pfam = False
    with gzip.open(Path(path), "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            entry = _INTERPRO_ENTRY.search(line)
            if entry is not None:
                if current_type == "Homologous_superfamily" and saw_pfam:
                    pfam_bearing_superfamilies += 1
                current, current_type = entry.group(1), entry.group(2)
                members.setdefault(current, set())
                inside = False
                saw_pfam = False
                continue
            if current is None:
                continue
            if "<member_list>" in line:
                inside = True
                continue
            if "</member_list>" in line:
                inside = False
                continue
            if not inside:
                continue
            cath = _CATHGENE3D_MEMBER.search(line)
            if cath is not None:
                members[current].add(cath.group(1))
            if 'db="PFAM"' in line:
                saw_pfam = True
    if current_type == "Homologous_superfamily" and saw_pfam:
        pfam_bearing_superfamilies += 1
    if not members:
        raise RuntimeError(f"{path} carries no InterPro entry")
    return {key: frozenset(value) for key, value in members.items()}, pfam_bearing_superfamilies


def superfamily_exemplars(
    labels: Sequence[str],
    *,
    entry_list: Path,
    interpro_xml: Path,
    cath_table: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Swiss-Prot exemplar accessions per candidate superfamily label.

    The route is the staged release's own: label name -> InterPro homologous
    superfamily -> its CATH-Gene3D member signatures -> the already-extracted
    accession/CATH table :mod:`src.capability.core.families` derives from
    ``protein2ipr``. The exemplars are therefore identified by **structural**
    signatures, independently of Pfam, which is what keeps the Pfam anchor an
    instrument price rather than a tautology.
    """

    from ..core.families import load_cath_superfamilies

    names = homologous_superfamily_entries(Path(entry_list))
    members, pfam_bearing = interpro_cath_members(Path(interpro_xml))
    by_superfamily: dict[str, set[str]] = {}
    for accession, codes in load_cath_superfamilies(path=Path(cath_table)).items():
        for code in codes:
            by_superfamily.setdefault(code, set()).add(accession)

    resolved: dict[str, dict[str, Any]] = {}
    n_named = 0
    n_with_cath = 0
    for label in labels:
        entry = names.get(label)
        if entry is None:
            continue
        n_named += 1
        codes = members.get(entry, frozenset())
        if not codes:
            continue
        n_with_cath += 1
        accessions = sorted(set().union(*(by_superfamily.get(code, set()) for code in codes)))
        resolved[label] = {
            "interpro": entry,
            "cath_superfamilies": sorted(codes),
            "accessions": accessions,
        }
    census = {
        "single_superfamily_labels": len(labels),
        "resolving_to_a_homologous_superfamily": n_named,
        "with_a_cath_gene3d_member": n_with_cath,
        "homologous_superfamilies_declaring_a_pfam_member": int(pfam_bearing),
        "pfam_member_note": SUPERFAMILY_PFAM_MEMBER_NOTE,
        "exemplar_route": (
            "label name -> InterPro Homologous_superfamily entry -> its CATH-Gene3D "
            "member signatures -> the accession/CATH table extracted from protein2ipr. "
            "Exemplars are identified structurally and independently of Pfam, which is "
            "what keeps the Pfam anchor an instrument price rather than a tautology"
        ),
    }
    return resolved, census


# ------------------------------------------------------- instrument inventory


def clean_availability(root: Path) -> dict[str, Any]:
    """CLEAN's runnability, imported from where it is already declared.

    Re-exported rather than re-derived so the campaign and
    ``concept_injection.py`` cannot disagree about whether the instrument
    exists. **A stub here would be an instrument that always agrees**, so an
    absent CLEAN produces no EC prediction at all and the channel is reported
    unavailable with the missing files named.
    """

    from ..core.protein_annotations import clean_availability as declared

    record = dict(declared(Path(root)))
    record["never_stubbed"] = (
        "if CLEAN's weights and the ESM-1b encoder are not restored, NO EC prediction "
        "is produced and the EC-assignment channel is reported unavailable rather than "
        "substituted. A stub would be an instrument that always agrees"
    )
    record["historical_anchor_is_not_availability"] = (
        "EXP-R2-015's CLEAN exact-EC anchor of 0.960 against 0.000 is a real historical "
        "measurement and is not evidence that the instrument can run today"
    )
    return record


def structural_covariate_availability(*, esmfold: Path, foldseek_tarball: Path) -> dict[str, Any]:
    """Whether pLDDT and Foldseek TM can be reported at all, stated before any run.

    Both are covariates and neither is ever gated on. They are reported **if and
    only if** ESMFold and Foldseek are available, and Foldseek is unusable without
    predicted structures, so an absent ESMFold withholds both.
    """

    esmfold = Path(esmfold)
    weights = sorted(esmfold.glob("*.safetensors")) + sorted(esmfold.glob("*.bin")) if esmfold.is_dir() else []
    missing: list[str] = []
    if not esmfold.is_dir():
        missing.append(f"ESMFold checkpoint directory ({esmfold})")
    elif not weights:
        missing.append(f"ESMFold weight files under {esmfold}")
    foldseek_present = Path(foldseek_tarball).is_file()
    if not foldseek_present:
        missing.append(f"Foldseek archive ({foldseek_tarball})")
    return {
        "runnable": not missing,
        "esmfold_directory_present": esmfold.is_dir(),
        "esmfold_weight_files": [str(path) for path in weights],
        "foldseek_archive_present": foldseek_present,
        "missing": missing,
        "reason": (
            "pLDDT and Foldseek top-TM are covariates and are never gated on. Foldseek "
            "needs a predicted structure, so an absent ESMFold withholds both channels; "
            "the absence is reported with the missing paths named rather than filled in"
        ),
        "ceiling": CEILING["predicted_structure_and_inferred_labels"],
    }


# ------------------------------------------------------------ the arm interfaces


@dataclass(frozen=True)
class GenerationArm:
    """One checkpoint this campaign samples from, and how it is prompted.

    ``role`` separates the two things an arm can be here. A ``conditioned`` arm
    carries the native class request under test; a ``floor`` arm generates under
    **no** class request at all and supplies reference (b). Nothing else is
    sampled, and there is no third role: an arm that could be prompted but was
    not is absent from this table rather than silently unconditioned.

    ``loader`` names the declared door the weights come through --
    :func:`src.capability.core.arms.load_arm` for a panel member and
    :func:`src.capability.models.joint_lineage.load_rung` for the ProLLaMA lineage, whose
    joint checkpoints are deliberately not in ``arms.py``.
    """

    name: str
    role: str
    channel: str
    modality: str
    loader: str
    checkpoint: str
    end_delimiter: str | None
    note: str
    #: Whether this checkpoint's modeling code honours Transformers' current
    #: ``Cache`` interface. **Do not assume it uniformly.** RITA-XL ignores
    #: ``past_key_values`` outright, and ProGen2's released remote code carries the
    #: legacy tuple cache and the pre-4.40 ``prepare_inputs_for_generation(self,
    #: input_ids, past=None)`` signature, so ``generate`` builds a ``DynamicCache``
    #: the model never receives and raises before a token is produced. An arm
    #: declared ``False`` here samples with the cache off, which recomputes the
    #: prefix every step -- slower, and correct.
    kv_cache: bool = True

    @property
    def conditioned(self) -> bool:
        return self.role == "conditioned"


ARMS: dict[str, GenerationArm] = {
    "zymctrl": GenerationArm(
        name="zymctrl",
        role="conditioned",
        channel="ec",
        modality="protein",
        loader="panel",
        checkpoint="zymctrl",
        end_delimiter="<end>",
        note=(
            "the native EC tag, rendered exactly as Cohort.input_strings renders it: "
            "'<ec><sep><start>' is the prompt and '<end>' closes the sequence. The "
            "tag is priced at 1.73 nats of leak, which is why the mismatched-label "
            "reference and not the raw rate is the primary statistic"
        ),
    ),
    "prollama": GenerationArm(
        name="prollama",
        role="conditioned",
        channel="superfamily",
        modality="protein",
        loader="lineage",
        checkpoint="ProLLaMA",
        end_delimiter=">",
        note=(
            "Stage 2's declared instruction form, '[Generate by superfamily] "
            "Superfamily=<...> Seq=<', resolved from joint_modes.JOINT_RENDERINGS "
            "rather than spelled here. EXP-R2-226 deliberately did NOT use this form "
            "-- supplying a true superfamily there would have been a conditioning leak on "
            "the quantity it measured -- and here the label IS the measurement"
        ),
    ),
    "progen2-medium": GenerationArm(
        name="progen2-medium",
        role="floor",
        channel="unconditioned",
        modality="protein",
        loader="panel",
        checkpoint="progen2-medium",
        end_delimiter="2",
        # Measured on this host, not assumed: `_prepare_cache_for_generation`
        # builds a `DynamicCache` from `config.num_hidden_layers`, which
        # `ProGenConfig` does not declare, and the released
        # `prepare_inputs_for_generation` takes `past=` rather than
        # `past_key_values=`, so a cache generate created would never reach the
        # model anyway. Sampling with the cache off is the correct path.
        kv_cache=False,
        note=(
            "764.8M, the declared floor for ZymCTRL: an unconditioned protein arm at "
            "comparable scale generating under no class request. Prompted with the "
            "n-to-c direction marker alone, which is the whole of its rendering when "
            "no content precedes"
        ),
    ),
    "protgpt2": GenerationArm(
        name="protgpt2",
        role="floor",
        channel="unconditioned",
        modality="protein",
        loader="panel",
        checkpoint="protgpt2",
        end_delimiter=None,
        note=(
            "about 774M and the same architecture family as ZymCTRL, reported beside "
            "ProGen2-medium as a second unconditioned floor. Prompted with the "
            "end-of-text token and a newline, which is the FASTA rendering its BPE "
            "merges were learned over"
        ),
    ),
    "qwen2.5-0.5b": GenerationArm(
        name="qwen2.5-0.5b",
        role="conditioned",
        channel="script",
        modality="text",
        loader="panel",
        checkpoint="qwen2.5-0.5b",
        end_delimiter=None,
        note="text positive control arm; a base checkpoint, so compliance is not assumed",
    ),
    "llama-3.2-3b": GenerationArm(
        name="llama-3.2-3b",
        role="conditioned",
        channel="script",
        modality="text",
        loader="panel",
        checkpoint="llama-3.2-3b",
        end_delimiter=None,
        note="text positive control arm; a base checkpoint, so compliance is not assumed",
    ),
}

#: Which floor each conditioned protein arm is read against. ProLLaMA's is the
#: same pair: this campaign does not difference the two conditioned arms' rates,
#: and the floors are what each is read against in its own channel.
FLOORS: dict[str, tuple[str, ...]] = {
    "zymctrl": ("progen2-medium", "protgpt2"),
    "prollama": ("progen2-medium", "protgpt2"),
    "qwen2.5-0.5b": ("qwen2.5-0.5b",),
    "llama-3.2-3b": ("llama-3.2-3b",),
}


def arm(name: str) -> GenerationArm:
    if name not in ARMS:
        raise KeyError(f"unknown generation arm {name!r}; declared: {sorted(ARMS)}")
    return ARMS[name]


#: The text positive control's classes: scripts whose Unicode ranges are
#: pairwise disjoint, so the oracle is deterministic and zero-parameter. Latin is
#: deliberately **not** a class -- it is the script the prompt itself is written
#: in and the default output of every one of these checkpoints, so "assigned to
#: Latin" is not a request-driven event -- but it IS counted in the denominator,
#: so an English continuation cannot be assigned to Greek on the strength of two
#: Greek letters. Twelve rather than the eight the registration requires as a
#: minimum, for the reason the protein cohort is sixteen: eight units is where a
#: percentile interval becomes defined, not where it becomes informative.
SCRIPTS: tuple[tuple[str, str, tuple[tuple[int, int], ...]], ...] = (
    ("greek", "Greek", ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    ("cyrillic", "Russian", ((0x0400, 0x04FF), (0x0500, 0x052F))),
    ("hebrew", "Hebrew", ((0x0590, 0x05FF),)),
    ("arabic", "Arabic", ((0x0600, 0x06FF), (0x0750, 0x077F))),
    ("devanagari", "Hindi", ((0x0900, 0x097F),)),
    ("bengali", "Bengali", ((0x0980, 0x09FF),)),
    ("tamil", "Tamil", ((0x0B80, 0x0BFF),)),
    ("thai", "Thai", ((0x0E00, 0x0E7F),)),
    ("georgian", "Georgian", ((0x10A0, 0x10FF),)),
    ("ethiopic", "Amharic", ((0x1200, 0x137F),)),
    ("armenian", "Armenian", ((0x0530, 0x058F),)),
    ("hangul", "Korean", ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F))),
)

_LATIN_RANGES: tuple[tuple[int, int], ...] = ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F))

#: A generation is assigned to a script when at least this share of its
#: script-bearing characters lie in that script's ranges.
SCRIPT_DOMINANCE = 0.5

#: The one conditioning template for the text control, frozen with everything
#: else. These are base checkpoints, not instruction-tuned ones, so compliance is
#: not assumed and the attainability screen below exists to say so.
TEXT_PROMPT_TEMPLATE = "The following passage is written in {label}.\n\n"


def script_classes() -> tuple[ClassEntry, ...]:
    """The frozen script queue, with the same fixed-point-free pairing rule."""

    entries = [(key, label, 0) for key, label, _ in SCRIPTS]
    return build_queue(entries, seed=DRAW_SEED)


def assign_script(text: str) -> str | None:
    """The declared script a passage is written in, or ``None``.

    Deterministic and zero-parameter by construction: a character belongs to at
    most one declared range, the denominator is every character that belongs to a
    declared range **or to Latin**, and a passage with no script-bearing character
    is assigned to nothing rather than to the first class.
    """

    counts: Counter[str] = Counter()
    bearing = 0
    for character in text:
        point = ord(character)
        if any(low <= point <= high for low, high in _LATIN_RANGES):
            bearing += 1
            continue
        for key, _, ranges in SCRIPTS:
            if any(low <= point <= high for low, high in ranges):
                counts[key] += 1
                bearing += 1
                break
    if bearing == 0 or not counts:
        return None
    key, count = counts.most_common(1)[0]
    return key if count / bearing >= SCRIPT_DOMINANCE else None


def script_ranges_are_disjoint() -> None:
    """Refuse a script table whose ranges overlap; the oracle depends on it."""

    seen: dict[int, str] = {}
    for key, _, ranges in SCRIPTS:
        for low, high in ranges:
            if high < low:
                raise ValueError(f"{key}: inverted range {low:#x}-{high:#x}")
            for point in (low, high):
                owner = seen.get(point)
                if owner is not None and owner != key:
                    raise ValueError(f"{key} and {owner} share the code point {point:#x}")
        for other_key, _, other_ranges in SCRIPTS:
            if other_key == key:
                continue
            for low, high in ranges:
                for other_low, other_high in other_ranges:
                    if low <= other_high and other_low <= high:
                        raise ValueError(
                            f"{key} {low:#x}-{high:#x} overlaps {other_key} "
                            f"{other_low:#x}-{other_high:#x}; the script oracle is only "
                            "deterministic while the ranges are disjoint"
                        )
    for key, _, ranges in SCRIPTS:
        for low, high in ranges:
            for latin_low, latin_high in _LATIN_RANGES:
                if low <= latin_high and latin_low <= high:
                    raise ValueError(f"{key} overlaps the Latin denominator range")


script_ranges_are_disjoint()


# ------------------------------------------------------------------- prompting


def prompt_for(handle: Any, spec: GenerationArm, label: str | None) -> str:
    """The string this arm is fed to produce one sample.

    Resolved from the declaration that already renders this arm's inputs --
    ``Cohort.input_strings`` for the panel members and
    ``JointRendering.render_protein`` for the lineage -- so a generation is
    prompted the way the model was trained and not the way a call site guessed.
    A conditioned arm refuses a missing label and a floor arm refuses a supplied
    one: the two conditions are what the whole estimand rests on.
    """

    if spec.modality == "text":
        # A text arm is its OWN unconditioned floor -- the same checkpoint sampled
        # with no request at all -- because there is no second text checkpoint at
        # comparable scale whose floor would be commensurable. The protein floors
        # are separate arms, which is why the refusal below applies only there.
        if label is None:
            return ""
        return TEXT_PROMPT_TEMPLATE.format(label=label)
    if spec.conditioned and label is None:
        raise ValueError(
            f"{spec.name} is a conditioned protein arm; its unconditioned reference is "
            f"a separate floor arm ({', '.join(FLOORS.get(spec.name, ()))}), not this "
            "checkpoint with the tag removed"
        )
    if spec.name == "zymctrl":
        return f"{label}<sep>{CONDITIONING_START}"
    if spec.name == "prollama":
        family = rendering("prollama")
        return family.protein_context_template.format(context=label) + family.protein_start
    if label is not None:
        raise ValueError(f"{spec.name} is an unconditioned floor; it takes no class label")
    if spec.name == "progen2-medium":
        return N_TO_C_MARKER
    if spec.name == "protgpt2":
        end_of_text = handle.tokenizer.eos_token
        if end_of_text is None:
            raise ValueError("protgpt2: tokenizer has no end-of-text token")
        return end_of_text + "\n"
    raise ValueError(f"no prompt is declared for {spec.name!r}")


def end_delimiter_for(handle: Any, spec: GenerationArm) -> str:
    """The delimiter that closes a generated sequence for this arm."""

    if spec.end_delimiter is not None:
        return spec.end_delimiter
    token = handle.tokenizer.eos_token
    if token is None:
        raise ValueError(f"{spec.name}: no end delimiter is declared and the tokenizer has no eos")
    return token


def cell_seed(*, seed: int, arm_name: str, class_key: str, condition: str) -> int:
    """A per-cell sampling seed derived from the campaign seed alone.

    One seed for the whole campaign would give two cells that happen to share a
    prompt the identical sample -- which is exactly what a class and its
    mismatched partner would do -- and would make the two conditions statistically
    dependent in a way the class-clustered bootstrap does not model. Derived
    rather than drawn, so the campaign is still reproducible from
    :data:`SAMPLING_SEED`.
    """

    material = f"{PRE_REGISTRATION}|{arm_name}|{class_key}|{condition}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
    return (int(seed) + offset) % (2**31 - 1)


def ensure_generate(model: Any) -> Any:
    """Attach GenerationMixin and a GenerationConfig when remote-code LMs lack them.

    Transformers 4.50+ stopped giving ``PreTrainedModel`` ``generate``. RITA and
    ProteinGLM still declare ``prepare_inputs_for_generation`` and need the mixin.
    Mixing the method in is not enough: ``generate`` then reads
    ``generation_config``, which those checkpoints leave unset.
    """

    if not callable(getattr(model, "generate", None)):
        from transformers.generation.utils import GenerationMixin

        cls = model.__class__
        model.__class__ = type(cls.__name__, (cls, GenerationMixin), {})
        if not callable(getattr(model, "generate", None)):
            raise TypeError(f"{cls.__name__} still has no generate after mixing GenerationMixin")
    if getattr(model, "generation_config", None) is None:
        from transformers.generation.configuration_utils import GenerationConfig

        config = getattr(model, "config", None)
        if config is None:
            raise TypeError(
                f"{type(model).__name__} has no generation_config and no config to build one"
            )
        model.generation_config = GenerationConfig.from_model_config(config)
    return model


def sample_continuations(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    n: int,
    seed: int,
    batch_size: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
    use_cache: bool = True,
    add_special_tokens: bool = True,
) -> list[str]:
    """``n`` sampled continuations of one prompt, decoded with specials kept.

    Every sample in a cell shares one prompt, so a batch needs no padding and the
    generated span is exactly the tail past the prompt length. Specials are kept
    in the decode because the end delimiter of two of these arms IS a special
    token, and stripping it would leave a run-on of several sequences that the
    residue extractor would read as one.

    ``batch_size`` is a feasibility parameter, not a scientific one, but the
    per-batch seed makes the sample depend on it, so it is recorded with the run.
    """

    import torch

    if n < 1 or batch_size < 1:
        raise ValueError("generation needs a positive count and batch size")
    model = ensure_generate(model)
    device = getattr(model, "device", None)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
    ids = encoded["input_ids"]
    if ids.shape[1] == 0:
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        if bos is None:
            raise ValueError("an empty prompt needs a bos or eos token to start from")
        ids = torch.tensor([[int(bos)]], dtype=torch.long)
    ids = ids.to(device)
    prompt_length = int(ids.shape[1])
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    outputs: list[str] = []
    index = 0
    with torch.no_grad():
        while len(outputs) < n:
            size = min(batch_size, n - len(outputs))
            torch.manual_seed(seed + index)
            generated = model.generate(
                input_ids=ids.repeat(size, 1),
                attention_mask=torch.ones((size, prompt_length), dtype=torch.long, device=ids.device),
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=REPETITION_PENALTY,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad,
                use_cache=use_cache,
            )
            outputs.extend(
                tokenizer.decode(row[prompt_length:], skip_special_tokens=False)
                for row in generated
            )
            index += 1
    return outputs[:n]


def extract_protein(text: str, *, end_delimiter: str) -> str:
    """The residue run a continuation spells, through the declared extractor."""

    from ..core.protein_annotations import extract_generated_sequence

    return extract_generated_sequence(text, end_delimiter=end_delimiter)


def composition(sequences: Sequence[str]) -> dict[str, Any]:
    """Length and amino-acid composition of one condition.

    Reported and never gated: a conditioning tag that changes only these is
    precisely the alternative the mismatched-label reference exists to expose, so
    the two have to be readable side by side.
    """

    lengths = [len(sequence) for sequence in sequences]
    residues = Counter()
    for sequence in sequences:
        residues.update(sequence)
    total = sum(residues.values())
    return {
        "n": len(sequences),
        "n_empty": int(sum(1 for length in lengths if length < MINIMUM_SCORED_RESIDUES)),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_length": float(np.median(lengths)) if lengths else 0.0,
        "residues_total": int(total),
        "frequency": {
            residue: (residues.get(residue, 0) / total if total else 0.0) for residue in AA20
        },
    }


def text_statistics(passages: Sequence[str]) -> dict[str, Any]:
    """The text control's analogue of :func:`composition`."""

    lengths = [len(passage) for passage in passages]
    return {
        "n": len(passages),
        "n_empty": int(sum(1 for length in lengths if length == 0)),
        "mean_characters": float(np.mean(lengths)) if lengths else 0.0,
        "median_characters": float(np.median(lengths)) if lengths else 0.0,
    }


# ---------------------------------------------------------------- the oracle


def annotate(
    sequences: Mapping[str, str],
    *,
    tool: Any,
    database: Any,
    workspace: Path,
    threads: int,
    shards: int,
    label: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Pfam-A families per named sequence, at the release's gathering thresholds.

    Sharded because the campaign puts tens of thousands of sequences through one
    profile database and HMMER parallelises a single scan only so far; each shard
    is the declared :func:`src.capability.concept_injection.run_hmmscan` call
    verbatim, and the tables are merged by the declared parser. A sequence that
    matches nothing is absent from the returned mapping, which is what "no family"
    means and is not the same as a sequence that was never searched -- the
    returned record carries the count that was.
    """

    from concurrent.futures import ThreadPoolExecutor

    from ..core.protein_annotations import parse_hmmscan_table, run_hmmscan, write_fasta

    if shards < 1 or threads < 1:
        raise ValueError("annotation needs a positive shard and thread count")
    named = {name: sequence for name, sequence in sequences.items() if sequence}
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "label": label,
        "n_requested": len(sequences),
        "n_searched": len(named),
        "n_empty": len(sequences) - len(named),
        "threshold": PFAM_THRESHOLD,
        "threshold_note": (
            "Pfam-A's own curated per-family gathering thresholds (--cut_ga), so a "
            "family call is a statement of the release and not a cut chosen inside "
            "this campaign"
        ),
        "shards": int(shards),
        "threads_per_shard": int(threads),
    }
    if not named:
        record["hmmscan_command"] = None
        return {}, record
    names = list(named)
    chunks = [names[index::shards] for index in range(shards)]
    chunks = [chunk for chunk in chunks if chunk]

    def scan(index_and_chunk: tuple[int, list[str]]) -> Path:
        index, chunk = index_and_chunk
        fasta = write_fasta(
            workspace / f"{label}.{index}.fasta", {name: named[name] for name in chunk}
        )
        table = workspace / f"{label}.{index}.tbl"
        command, _ = run_hmmscan(
            tool, database, fasta, table, threads=threads, gathering_threshold=True
        )
        record.setdefault("hmmscan_command", command)
        return table

    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        tables = list(pool.map(scan, enumerate(chunks)))
    hits: dict[str, list[dict[str, Any]]] = {}
    for table in tables:
        for name, entries in parse_hmmscan_table(table).items():
            hits.setdefault(name, []).extend(entries)
    for entries in hits.values():
        entries.sort(key=lambda entry: entry["evalue"])
    record["n_with_any_family"] = len(hits)
    return hits, record


def families_of(hits: Mapping[str, Sequence[Mapping[str, Any]]], name: str) -> frozenset[str]:
    """Unversioned Pfam accessions on one sequence."""

    return frozenset(entry["accession_unversioned"] for entry in hits.get(name, ()))


def referent_from_draw(
    hits: Mapping[str, Sequence[Mapping[str, Any]]],
    names: Sequence[str],
    *,
    share: float = REFERENT_FAMILY_SHARE,
) -> tuple[str, ...]:
    """The Pfam families a class's referent draw carries, as the class's referent.

    Neither an EC number nor an InterPro superfamily is a Pfam accession, so the
    class-to-profile map has to come from somewhere. It comes from the class's
    own **referent draw**, which is disjoint from the anchor draw and from every
    generation, so the anchor that prices the oracle is not scored against a
    family set fitted on the same sequences. A class whose referent comes out
    empty is unmeasurable and is reported as such, never assigned a family set
    nobody measured.
    """

    if not 0.0 < share <= 1.0:
        raise ValueError("the referent share must lie in (0, 1]")
    if not names:
        raise ValueError("a referent needs at least one drawn record")
    counts: Counter[str] = Counter()
    for name in names:
        counts.update(families_of(hits, name))
    floor = share * len(names)
    return tuple(sorted(family for family, count in counts.items() if count >= floor))


def assigned(
    hits: Mapping[str, Sequence[Mapping[str, Any]]],
    names: Sequence[str],
    referent: Sequence[str],
) -> list[bool]:
    """Per sequence, whether the oracle assigns it to the class.

    A sequence carrying **any** family of the class's referent is assigned to the
    class. That is a homology statement by construction (L9) and is what the
    channel contains; it is not a statement that the sequence performs the
    class's chemistry.
    """

    wanted = {str(value).split(".", 1)[0] for value in referent}
    if not wanted:
        raise ValueError("an empty referent assigns nothing; the class is unmeasurable")
    return [bool(families_of(hits, name) & wanted) for name in names]


# ------------------------------------------------------------- the draw pools


def ec_class_records(
    keys: Sequence[str], *, path: Path = ZYMCTRL_FASTA
) -> dict[str, list[str]]:
    """Every canonical-alphabet sequence of each requested EC class, in file order."""

    wanted = set(keys)
    allowed = set(AA20)
    pools: dict[str, list[str]] = {key: [] for key in keys}
    for header, body in iter_fasta(Path(path)):
        label = header.split("|")[-1]
        if label not in wanted:
            continue
        if CONDITIONING_START not in body or CONDITIONING_END not in body:
            continue
        sequence = body.split(CONDITIONING_START)[1].split(CONDITIONING_END)[0]
        if set(sequence) <= allowed and sequence:
            pools[label].append(sequence)
    return pools


def swissprot_sequences(accessions: Iterable[str], *, path: Path = SWISSPROT_FASTA) -> dict[str, str]:
    """Canonical-alphabet Swiss-Prot sequences for the requested accessions."""

    from ..core.sequence_metadata import swissprot_accession

    wanted = set(accessions)
    allowed = set(AA20)
    found: dict[str, str] = {}
    for header, sequence in iter_fasta(Path(path)):
        accession = swissprot_accession(header)
        if accession in wanted and set(sequence) <= allowed and sequence:
            found[accession] = sequence
    return found


def split_draw(pool: Sequence[str], *, seed: int) -> tuple[list[str], list[str]]:
    """A class's referent and anchor draws: two disjoint windows of one permutation.

    The permutation is the seeded one every draw in this campaign uses. Disjoint
    rather than nested, because a referent fitted on the sequences the anchor is
    measured on would return a real-side rate of one whatever the oracle does.
    """

    if len(pool) < MIN_CLASS_RECORDS:
        raise RuntimeError(
            f"{len(pool)} records cannot supply a {REFERENT_DRAW}-record referent draw "
            f"and a disjoint {ANCHOR_DRAW}-record anchor draw"
        )
    order = np.random.default_rng(seed).permutation(len(pool))
    referent = [pool[int(index)] for index in order[:REFERENT_DRAW]]
    anchor = [pool[int(index)] for index in order[REFERENT_DRAW:MIN_CLASS_RECORDS]]
    return referent, anchor


def uniref50_pool(
    *, path: Path = UNIREF50_FASTA, size: int, seed: int, min_len: int, max_len: int
) -> list[str]:
    """A seeded reservoir of canonical-alphabet UniRef50 sequences in a length band.

    One pass, uniform over every eligible record of the whole corpus rather than a
    prefix of it (Appendix B rule 1): a biological corpus is ordered by cluster, so
    the head of the file is a region and not a sample.
    """

    if size < 1:
        raise ValueError("a reservoir needs a positive size")
    allowed = set(AA20)
    rng = np.random.default_rng(seed)
    reservoir: list[str] = []
    seen = 0
    for _, sequence in iter_fasta(Path(path)):
        if not (min_len <= len(sequence) <= max_len) or not set(sequence) <= allowed:
            continue
        seen += 1
        if len(reservoir) < size:
            reservoir.append(sequence)
            continue
        position = int(rng.integers(0, seen))
        if position < size:
            reservoir[position] = sequence
    if len(reservoir) < size:
        raise RuntimeError(
            f"{path} yielded {len(reservoir)} eligible records for a reservoir of {size}"
        )
    return reservoir


def length_matched(
    targets: Sequence[str], pool: Sequence[str], *, seed: int, tolerance: float = LENGTH_MATCH_TOLERANCE
) -> list[str]:
    """One length-matched random UniRef50 protein per real exemplar, without replacement.

    EXP-R2-015's control was real length-matched UniRef50 proteins, not synthetic
    residue strings, and this reproduces it: a synthetic string would fail every
    profile for reasons that have nothing to do with class specificity, so it
    could not price a class-specific call.
    """

    if not 0.0 < tolerance < 1.0:
        raise ValueError("the length tolerance must lie in (0, 1)")
    lengths = np.array([len(sequence) for sequence in pool])
    order = np.random.default_rng(seed).permutation(len(pool))
    available = [int(index) for index in order]
    used: set[int] = set()
    matched: list[str] = []
    for target in targets:
        want = len(target)
        low, high = want * (1.0 - tolerance), want * (1.0 + tolerance)
        chosen = None
        for index in available:
            if index in used:
                continue
            if low <= lengths[index] <= high:
                chosen = index
                break
        if chosen is None:
            candidates = [index for index in available if index not in used]
            if not candidates:
                raise RuntimeError("the UniRef50 reservoir is exhausted")
            chosen = min(candidates, key=lambda index: abs(int(lengths[index]) - want))
        used.add(chosen)
        matched.append(pool[chosen])
    return matched


# ------------------------------------------------------ rates and near duplicates


def near_duplicate_group_ids(
    sequences: Sequence[str], *, unit: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Group ids for a condition's generations, with the empty ones held together.

    A generation that decodes to nothing is a genuine failure of the interface and
    is kept in the denominator, but a hundred of them are one repeated outcome and
    not a hundred independent ones, so they share a single group. Everything else
    goes through the declared shingle-containment grouping unchanged.
    """

    if unit not in SHINGLE_UNITS:
        raise ValueError(f"unknown symbol unit {unit!r}; declared: {sorted(SHINGLE_UNITS)}")
    ids = np.full(len(sequences), -1, dtype=np.int64)
    filled = [index for index, value in enumerate(sequences) if value]
    summary: dict[str, Any] = {
        "unit": unit,
        "n_records": len(sequences),
        "n_empty_records": len(sequences) - len(filled),
        "empty_records_share_one_group": True,
    }
    next_group = 0
    if filled:
        groups, grouping = near_duplicate_groups([sequences[index] for index in filled], unit=unit)
        for position, index in enumerate(filled):
            ids[index] = int(groups[position])
        next_group = int(ids.max()) + 1
        summary.update(grouping)
    if len(filled) < len(sequences):
        ids[ids < 0] = next_group
        next_group += 1
    summary["n_groups_including_empty"] = int(next_group)
    return ids, summary


def grouped_rate(hits: Sequence[bool], groups: np.ndarray) -> float:
    """The assignment rate with near-duplicates collapsed to one unit each.

    The class rate is the mean over near-duplicate GROUPS of the within-group hit
    fraction, so a burst of near-identical samples contributes once however many
    times it was drawn.
    """

    values = np.asarray([1.0 if value else 0.0 for value in hits], dtype=np.float64)
    ids = np.asarray(groups)
    if values.shape != ids.shape:
        raise ValueError("the hit vector and the group ids do not align")
    if values.size == 0:
        raise ValueError("a rate needs at least one generation")
    return float(np.mean([values[ids == group].mean() for group in np.unique(ids)]))


def _mean_metric(truth: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(predicted))


def _rate_difference(truth: np.ndarray, predicted: np.ndarray) -> float:
    left = np.asarray(truth).astype(bool)
    if left.all() or (~left).all():
        return float("nan")
    return float(predicted[left].mean() - predicted[~left].mean())


def class_clustered_mean(
    values: Mapping[str, float], *, resamples: int, seed: int
) -> dict[str, Any]:
    """The class-clustered mean of a per-class contrast, with its 95% interval.

    The resampling unit is the CLASS, so the class count is the sample size, and
    the interval comes from the package's one declared group resampler rather than
    from a second one written here. Below the eight-unit floor the interval is not
    reported wider, it is not reported: a nominal 95% percentile interval over so
    few atoms realises well under 95% coverage and is not even monotone in width.
    """

    keys = sorted(values)
    if len(keys) < MINIMUM_CLASSES:
        return {
            "n_classes": len(keys),
            "minimum_classes": MINIMUM_CLASSES,
            "mean": None,
            "ci95": None,
            "degenerate": True,
            "degenerate_reason": (
                f"{len(keys)} classes is below the {MINIMUM_CLASSES}-unit bootstrap "
                "floor; no percentile interval is reported and the arm is not scored"
            ),
        }
    vector = np.asarray([values[key] for key in keys], dtype=np.float64)
    bootstrap = paired_group_bootstrap(
        vector,
        vector,
        np.zeros_like(vector),
        np.arange(vector.size),
        _mean_metric,
        seed=seed,
        n_bootstrap=resamples,
    )
    return {
        "n_classes": len(keys),
        "minimum_classes": MINIMUM_CLASSES,
        "mean": bootstrap["difference"],
        "ci95": bootstrap["difference_ci95"],
        "degenerate": False,
        "degenerate_reason": None,
        "n_bootstrap": bootstrap["n_bootstrap_requested"],
        "n_finite_draws": bootstrap["n_finite_draws"],
        "resampling_unit": "the class",
    }


def two_sample_rate_contrast(
    left_hits: Sequence[bool],
    left_groups: np.ndarray,
    right_hits: Sequence[bool],
    right_groups: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """A within-class rate difference clustered on near-duplicate groups.

    Used where the unit is the generation rather than the class -- the text
    control's per-class attainability screen. The two conditions are separate
    samples, so the resampler receives one row per generation with a condition
    indicator and one group per near-duplicate group per condition, which is the
    shape ``concept_injection.annotation_rate_contrast`` already uses.
    """

    left = np.asarray([1.0 if value else 0.0 for value in left_hits], dtype=np.float64)
    right = np.asarray([1.0 if value else 0.0 for value in right_hits], dtype=np.float64)
    offset = int(np.asarray(left_groups).max()) + 1 if left.size else 0
    condition = np.concatenate([np.ones(left.size, dtype=int), np.zeros(right.size, dtype=int)])
    predicted = np.concatenate([left, right])
    groups = np.concatenate([np.asarray(left_groups), np.asarray(right_groups) + offset])
    distinct = int(np.unique(groups).size)
    if distinct < MINIMUM_BOOTSTRAP_UNITS:
        return {
            "difference": float(left.mean() - right.mean()) if left.size and right.size else None,
            "ci95": None,
            "n_groups": distinct,
            "degenerate": True,
            "degenerate_reason": (
                f"{distinct} near-duplicate groups across both conditions is below the "
                f"{MINIMUM_BOOTSTRAP_UNITS}-unit floor"
            ),
        }
    bootstrap = paired_group_bootstrap(
        condition,
        predicted,
        np.zeros_like(predicted),
        groups,
        _rate_difference,
        seed=seed,
        n_bootstrap=resamples,
    )
    return {
        "difference": bootstrap["difference"],
        "ci95": bootstrap["difference_ci95"],
        "n_groups": distinct,
        "degenerate": False,
        "degenerate_reason": None,
        "n_bootstrap": bootstrap["n_bootstrap_requested"],
    }


def lower_bound_positive(block: Mapping[str, Any] | None) -> bool | None:
    """Whether a reported interval's 95% lower bound clears zero."""

    if not block or block.get("ci95") is None:
        return None
    return bool(block["ci95"][0] > 0.0)


# ------------------------------------------------------------- the instrument gate


def anchor_record(
    *, real: Sequence[bool], random: Sequence[bool], referent: Sequence[str]
) -> dict[str, Any]:
    """One class's instrument price, and whether the oracle may be used on it.

    Appendix B rule 2 applied to an oracle instead of to a gate: the channel is
    admitted on a class only if real exemplars of that class clear
    :data:`ANCHOR_REAL_FLOOR` and length-matched random UniRef50 proteins stay
    under :data:`ANCHOR_RANDOM_CEILING`. A class that fails is removed **before any
    generation is scored** and is an unmeasurable class, not a failing one. An
    oracle calibrated on lysozyme is not thereby calibrated on a class it has
    never been checked on, which is why this is measured per class.
    """

    real_rate = float(np.mean([1.0 if value else 0.0 for value in real])) if real else 0.0
    random_rate = float(np.mean([1.0 if value else 0.0 for value in random])) if random else 0.0
    reasons: list[str] = []
    if not referent:
        reasons.append(
            "the referent draw carried no Pfam family at or above the declared share, "
            "so this class has no profile set the oracle could assign to"
        )
    if real_rate < ANCHOR_REAL_FLOOR:
        reasons.append(
            f"real-exemplar assignment {real_rate:.3f} is below the {ANCHOR_REAL_FLOOR:.2f} floor"
        )
    if random_rate > ANCHOR_RANDOM_CEILING:
        reasons.append(
            f"length-matched random assignment {random_rate:.3f} exceeds the "
            f"{ANCHOR_RANDOM_CEILING:.2f} ceiling"
        )
    return {
        "referent": list(referent),
        "n_referent_families": len(referent),
        "real_rate": real_rate,
        "random_rate": random_rate,
        "n_real": len(real),
        "n_random": len(random),
        "real_floor": ANCHOR_REAL_FLOOR,
        "random_ceiling": ANCHOR_RANDOM_CEILING,
        "admitted": not reasons,
        "unmeasurable_reasons": reasons,
    }


# ------------------------------------------------------------------ the compound

OUTCOMES = (
    "conditioning_moves_generation_toward_the_requested_class",
    "tag_moves_the_distribution_without_selecting_the_requested_class",
    "selective_against_mismatch_but_not_above_the_unconditioned_floor",
    "class_selective_on_part_of_the_label_space",
    "not_scored",
)


def compound_verdict(
    *,
    against_mismatch: Mapping[str, Any],
    against_floor: Mapping[str, Any],
    per_class_contrast: Mapping[str, float],
) -> dict[str, Any]:
    """The only gate: three clauses, all of which must hold.

    Clause 1 is the one a 1.73-nat leak measurement leaves open and is therefore
    the primary statistic.
    Clause 2 asks whether the selected rate is above what an unconditioned arm of
    similar scale produces unprompted. Clause 3 stops an arm-level verdict from
    being carried by one or two classes. Each failure has its own reading, and
    none of them is narrowed to a class subset or re-run at another sampling
    configuration.
    """

    positive = {key: value > 0.0 for key, value in per_class_contrast.items()}
    n_positive = int(sum(positive.values()))
    n_classes = len(positive)
    clause_one = lower_bound_positive(against_mismatch)
    clause_two = lower_bound_positive(against_floor)
    clause_three = (n_positive * 2 >= n_classes) if n_classes else None

    if n_classes < MINIMUM_CLASSES or clause_one is None or clause_two is None:
        outcome = "not_scored"
    elif not clause_one:
        outcome = "tag_moves_the_distribution_without_selecting_the_requested_class"
    elif not clause_two:
        outcome = "selective_against_mismatch_but_not_above_the_unconditioned_floor"
    elif not clause_three:
        outcome = "class_selective_on_part_of_the_label_space"
    else:
        outcome = "conditioning_moves_generation_toward_the_requested_class"

    return {
        "outcome": outcome,
        "outcomes": list(OUTCOMES),
        "clause_1_requested_minus_mismatched": clause_one,
        "clause_2_requested_minus_floor": clause_two,
        "clause_3_half_the_classes_individually_positive": clause_three,
        "n_classes": n_classes,
        "n_classes_individually_positive": n_positive,
        "per_class_positive": positive,
        "gate": (
            "all three clauses must hold. Clause 1 alone failing means the tag moves "
            "the distribution without selecting the requested class, which is the "
            "reading a 1.73-nat leak cannot distinguish. Clause 3 alone failing "
            "issues no arm-level verdict and is reported per class"
        ),
        "licence": (
            "a pass licenses exactly this: this arm's native conditioning interface "
            "moves generation toward the requested class on this class cohort under "
            "this oracle. It is a behavioural capability statement about that arm and "
            "nothing more"
        ),
    }
