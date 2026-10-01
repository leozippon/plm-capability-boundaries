#!/usr/bin/env python3
"""Check the manuscript against the derived-number table.

Reads ``manuscript/`` read-only and refuses when any of the
following does not hold:

* **traceability** --- every number in the manuscript's prose, tables and
  captions resolves to an entry of the derived-number table, or to a declared
  non-evidence class (LaTeX geometry, template markup and the like);
* **one support per comparison** --- no two entries that differ only in their
  support are quoted in one sentence or one table row;
* **an interval carries its resampling unit** --- every interval in the
  manuscript resolves to an entry whose resampling unit and draw count are
  recorded, and the resampling unit is named in the manuscript;
* **a quantity carries its unit** --- every quantity resolves to an entry with a
  unit, and a unit that is not dimensionless is named where the quantity is
  quoted;
* **no bare point estimate** --- an estimate the table carries an interval for
  is not quoted without one;
* **an effective count names its convention** --- a Kish count is quoted only
  where the weighting it was computed under is named, because a count computed
  under a different weighting than its estimator applies is misleading even when
  it is close;
* **a provisional result is marked** --- a representation-level quantity is
  quoted only in a paragraph that says it is provisional.

Declared exceptions live in the policy file and each must name a live anchor in
the manuscript, so an exception cannot outlive the sentence it was written for.
The checker never edits the manuscript.

    python scripts/capability/reporting/check_manuscript_evidence.py
"""
from __future__ import annotations

from pathlib import Path
import argparse
from decimal import Decimal
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.core.evidence_ledger import REPO_ROOT, load_table  # noqa: E402

MANUSCRIPT = REPO_ROOT / "manuscript"
EVIDENCE = REPO_ROOT / 'results/shared/manuscript_evidence_20260927'
SOURCES = ("main.tex", "supplementary-information.tex")

#: LaTeX constructs whose numbers are markup rather than evidence. Each is
#: removed from the prose before any number is read, so a geometry value cannot
#: be mistaken for a measurement and a measurement cannot hide inside markup.
NON_EVIDENCE_PATTERNS = (
    r"(?<!\\)%[^\n]*",
    r"\\documentclass(?:\[[^]]*\])?\{[^}]*\}",
    r"\\usepackage(?:\[[^]]*\])?\{[^}]*\}",
    r"\\(?:setcounter|setcitestyle|setlength|definecolor|newcommand|renewcommand)"
    r"(?:\{[^{}]*\}|\[[^]]*\]|\{\\?[A-Za-z]+\}|\{[.\d]+\})+",
    r"\\includegraphics(?:\[[^]]*\])?\{[^}]*\}",
    r"\\(?:label|ref|cite|url|bibliography|input|include)\{[^}]*\}",
    r"\\(?:author|affil)\*?(?:\[[\d,]+\])?",
    r"\\rowcolor\{[^}]*\}",
    r"\\arraybackslash",
    r"[.\d]+\\(?:textwidth|linewidth|columnwidth|tabcolsep)",
    r"p\{[.\d]*\\?[A-Za-z]*\}",
)

#: How the manuscript may name each unit. A quantity whose unit is not
#: dimensionless has to be quoted where one of these appears.
UNIT_RENDERINGS = {
    "kcal/mol": (r"kcal/mol", r"kcal\,mol"),
    "kcal^2/mol^2": (r"kcal\$\^2\$/mol\$\^2\$", r"kcal\^2/mol\^2", r"squared kcal/mol"),
    "percent": (r"\\%", r"per cent", r"percent"),
    "attempts": (r"attempt",),
    "sequences": (r"sequence",),
    "groups": (r"group",),
    "checkpoints": (r"checkpoint", r"arms?\b", r"cells?\b"),
    "checkpoint--seed cells": (r"cell",),
    "family groups": (r"group",),
    "double-mutant cycles": (r"cycle",),
    "site pairs": (r"site pair",),
    "effective site pairs": (r"site pair",),
    "site tuples": (r"site tuple", r"site tripl"),
    "effective site tuples": (r"site tuple", r"site tripl"),
    "distinct measured states": (r"state",),
    "backgrounds": (r"background",),
    "assays": (r"assay",),
    "variants": (r"variant",),
    "source clusters": (r"cluster",),
    "effective source clusters": (r"cluster",),
    "effective family groups": (r"group",),
    "emitted sequence": (r"sequence",),
    "bootstrap draws": (r"bootstrap", r"draw", r"resample"),
    "split seed": (r"seed",),
    "profiles": (r"profile",),
    "cells": (r"cell",),
    "procedures": (r"procedure", r"checkpoint"),
    "generator cohorts": (r"cohort", r"generator", r"control"),
    "variants per assay": (r"variant",),
    "rows": (r"row", r"variant", r"state"),
    "units": (r"unit", r"group", r"cluster"),
    "effective units": (r"effective", r"Kish"),
    "wild-type clusters": (r"cluster",),
    "residues": (r"residue",),
}
#: Units whose absence changes what a number means. A counting unit is carried
#: by the sentence's own noun ("14 of 33 checkpoints"), so requiring it again
#: would report grammar rather than evidence; a physical or scaled unit is not,
#: and a quantity quoted without one is not readable.
PHYSICAL_UNITS = frozenset({"kcal/mol", "kcal^2/mol^2", "percent", "nats per token",
                            "nats per residue", "scaled-fitness units", "log2 enrichment"})

#: The measured scales a difference can only be taken within. A percentage or a
#: ratio is a share *of* a quantity and pairs with it legitimately, so it is not
#: one of these.
MEASURED_SCALES = frozenset({"kcal/mol", "kcal^2/mol^2", "log2 enrichment", "nats per token",
                             "nats per residue", "scaled-fitness units"})

DIMENSIONLESS = tuple(
    unit for unit in ("dimensionless", "dimensionless Spearman", "dimensionless ratio",
                      "dimensionless probability", "dimensionless fraction", "dimensionless R^2",
                      "dimensionless rate", "dimensionless rate difference", "support_count")
)

#: A decimal with at least this many places identifies an entry on its own. A
#: shorter one has to be attributed by what the sentence names, or be a count,
#: a support size or a declared constant: otherwise a table of several thousand
#: rows would match almost any number and the check would mean nothing.
SPECIFIC_DECIMALS = 3
#: The most decimal places the manuscript quotes. A value is indexed at every
#: rounding up to this, so a number written to six places still finds the entry
#: it came from.
MAX_DECIMALS = 7
COUNTING_KINDS = frozenset({"count", "support_count", "constant", "ratio", "census",
                            "effective_count", "range_bound", "artifact_leaf"})

PROVISIONAL_MARKERS = ("provisional", "held open", "pending", "not reported", "under reassessment")
NUMBER = re.compile(r"(?<![\w.])([-+]?\d[\d,]*(?:\.\d+)?)")
#: Checkpoint and label names carry digits --- ProGen3-3B, Qwen2.5-0.5b,
#: order-3 --- and those digits are not quantities. They are blanked to spaces
#: before numbers are read, which keeps every offset so an interval's extent
#: still lines up, while the unblanked text remains what a quantity is
#: attributed and its unit named against.
NAME = re.compile(r"(?<![\w-])[A-Za-z][A-Za-z0-9.]*(?:-[A-Za-z0-9.]+)*(?![\w-])")
SCIENTIFIC = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*\$?\s*\\times\s*10\^\{(-?\d+)\}")
BARE_POWER = re.compile(r"10\^\{(-?\d+)\}")
#: An interval as the manuscript writes it. The whitespace is not decorative: the
#: scientific-notation rewrite above renders ``$-1.847\times10^{-6}$`` with spaces
#: inside the maths delimiters, and a pattern that did not allow them read a correctly
#: bracketed sentence as a bare point estimate.
INTERVAL = re.compile(r"\[\s*\$?\s*([-+]?[\d.,]+)\s*\$?\s*,\s*\$?\s*([-+]?[\d.,]+)\s*\$?\s*\]")


def policy_path(directory: Path) -> Path:
    return directory / "checker-policy.json"


def default_policy() -> dict[str, object]:
    return {
        "schema": "manuscript_evidence_checker_policy_v1",
        "rule": "every exception names a live anchor in the manuscript; a dead anchor is a refusal, so "
                "an exception cannot outlive the sentence it was written for",
        "interval_convention_statement":
            "Brackets denote pointwise 95\\% percentile intervals over the stated biological resampling unit.",
        "cross_support_exceptions": [],
        "cross_unit_exceptions": [],
        "bare_estimate_exceptions": [],
        "effective_count_exceptions": [],
        "provisional_exceptions": [],
        "untraced_accepted": [],
    }


def load_policy(directory: Path) -> dict[str, object]:
    path = policy_path(directory)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(default_policy(), indent=2) + "\n")
    return json.loads(path.read_text())


def _fixed(value: float | Decimal) -> str:
    """A decimal rendering with no exponent, so the number reads as one token."""
    text = format(Decimal(str(value)), "f")
    if "." in text:
        text = text.rstrip("0")
    return text + "0" if text.endswith(".") else text


def strip_markup(text: str) -> str:
    body = text.split(r"\begin{document}", 1)[-1]
    for pattern in NON_EVIDENCE_PATTERNS:
        body = re.sub(pattern, " ", body)
    # An em or en dash is punctuation, not a sign: "163--174" states two counts
    # and "strata---19 native" one. Blanking them keeps every offset.
    body = body.replace("---", "   ").replace("--", "  ")
    # A scientific form is one number. Rendering it in exponent notation would
    # re-tokenize as its mantissa, which reported 1.11 and 2.8 as numbers the
    # manuscript never wrote; a fixed-point rendering keeps it whole.
    body = SCIENTIFIC.sub(lambda m: " " + _fixed(Decimal(m.group(1)) * Decimal(10) ** int(m.group(2))) + " ", body)
    body = BARE_POWER.sub(lambda m: " " + _fixed(Decimal(10) ** int(m.group(1))) + " ", body)
    return body


def _braced(body: str, start: int) -> tuple[str, int]:
    depth, position = 1, start
    while depth and position < len(body):
        if body[position] == "{" and body[position - 1] != "\\":
            depth += 1
        elif body[position] == "}" and body[position - 1] != "\\":
            depth -= 1
        position += 1
    return body[start:position - 1], position


def scopes(body: str) -> list[tuple[str, str, str]]:
    """Split the prose into the units a comparison can happen inside.

    Each unit carries the block it sits in, because the manuscript marks a
    representation-level result as provisional once per paragraph rather than
    once per sentence, and a check has to read it the way it is written.
    """
    units: list[tuple[str, str, str]] = []
    remaining = body
    for match in re.finditer(r"\\caption\{", body):
        caption, _ = _braced(body, match.end())
        remaining = remaining.replace(caption, " ")
        units += [("caption", sentence, caption)
                  for sentence in re.split(r"(?<=\.)\s+(?=[A-Z\\$])", caption) if sentence.strip()]
    for match in re.finditer(r"\\begin\{tabular\}", body):
        end = body.find(r"\end{tabular}", match.end())
        if end < 0:
            continue
        table = body[match.end():end]
        remaining = remaining.replace(table, " ")
        # The enclosing caption supplies units; pair a following interval row
        # with its estimates rather than treating the whole table as a sentence.
        enclosing_start = body.rfind(r"\begin{table}", 0, match.start())
        enclosing_end = body.find(r"\end{table}", end)
        block = body[enclosing_start:enclosing_end] if enclosing_start >= 0 and enclosing_end >= 0 else table
        rows = [row for row in re.split(r"\\\\", table) if "&" in row]
        index = 0
        while index < len(rows):
            row = rows[index]
            if index + 1 < len(rows) and re.search(r"interval", rows[index + 1].split("&")[0], re.I):
                row += " " + rows[index + 1]
                index += 1
            units.append(("table row", row, block))
            index += 1
    for paragraph in re.split(r"\n\s*\n", remaining):
        for sentence in re.split(r"(?<=\.)\s+(?=[A-Z\\$])", paragraph):
            if sentence.strip():
                units.append(("sentence", sentence, paragraph))
    return units


def parse(token: str) -> tuple[float, int]:
    cleaned = token.replace(",", "")
    decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
    return float(cleaned), decimals


class Index:
    """Looks a manuscript number up among the table's entries.

    A number matches an entry's value, or one end of its interval: the
    manuscript quotes both, and an interval bound is traced to the estimate it
    belongs to rather than treated as an untraced number.
    """

    def __init__(self, quantities: list[dict]) -> None:
        self.quantities = quantities
        self.by_rounding: dict[tuple[int, float], list[dict]] = {}
        self.integers: dict[float, list[dict]] = {}
        self.bounds: dict[tuple[int, float], list[dict]] = {}
        # A value smaller than the smallest rounding the index carries would
        # collapse to zero under absolute rounding, so tiny magnitudes are also
        # indexed by significant figures. Without this a decomposition residual
        # of 1.11e-16 could never match the row it came from.
        self.significant: dict[str, list[dict]] = {}
        for entry in quantities:
            value = entry["value"]
            for decimals in range(0, MAX_DECIMALS + 1):
                self.by_rounding.setdefault((decimals, round(value, decimals)), []).append(entry)
            if float(value).is_integer():
                self.integers.setdefault(float(value), []).append(entry)
            for figures in (2, 3, 4):
                self.significant.setdefault(f"{value:.{figures}e}", []).append(entry)
            for key in ("interval_low", "interval_high"):
                bound = entry.get(key)
                if bound is None:
                    continue
                for decimals in range(0, MAX_DECIMALS + 1):
                    self.bounds.setdefault((decimals, round(bound, decimals)), []).append(entry)

    def candidates(self, token: str) -> list[dict]:
        value, decimals = parse(token)
        if decimals == 0:
            # An integer has to match an integral entry exactly: rounding a
            # measured 6.51 onto a count of 7 would trace a number that no
            # entry states.
            return self.integers.get(value, [])
        return self.by_rounding.get((decimals, value), [])

    def tiny_candidates(self, token: str) -> list[dict]:
        """Entries matching a very small number by significant figures."""
        value, _ = parse(token)
        if value and abs(value) >= 1e-4:
            return []
        for figures in (2, 3, 4):
            found = self.significant.get(f"{value:.{figures}e}")
            if found:
                return found
        return []

    def bound_candidates(self, token: str) -> list[dict]:
        value, decimals = parse(token)
        return self.bounds.get((decimals, value), [])


def _words(text: str) -> set[str]:
    return {re.sub(r"[^a-z0-9]", "", word) for word in text.lower().split()}


def _attribution(entry: dict) -> set[str]:
    """The identifiers a sentence would have to name to be quoting this entry:
    the distinctive segments of its id, such as a checkpoint name."""
    tokens = set()
    for segment in entry["id"].split("/"):
        flat = re.sub(r"[^a-z0-9]", "", segment.lower())
        if flat and any(character.isdigit() for character in flat) and \
                any(character.isalpha() for character in flat):
            tokens.add(flat)
    return tokens


#: Words that carry no attribution, so an overlap on them means nothing.
STOPWORDS = frozenset("""a an and are as at be by each for from in is it its of on one or over own
per same the their this those to under with without which what when where how much many its than
that these both all any""".split())


def _claim_words(entry: dict) -> set[str]:
    return {word for word in _words(entry["claim"]) if word and word not in STOPWORDS}


def attributable(token: str, candidates: list[dict],
                 scope_words: set[str]) -> tuple[list[dict], bool]:
    """Narrow a value match to the entries the sentence is actually quoting.

    First by what the sentence names --- a checkpoint name in the entry's id ---
    and then by how much of the entry's own claim the sentence repeats. The
    second flag says whether the narrowing was strong: the sentence named the
    entry, or only one entry ever matched the value. A weak attribution still
    counts as traced, because the value does come from the evidence, but no
    condition is refused against it: a refusal has to be about a quantity the
    table can pin down, not about whichever of several rows shares a number.
    """
    if not candidates:
        return [], False
    _, decimals = parse(token)
    strong = len(candidates) == 1
    # A declared threshold must be attributed by its declaration, even when
    # the same sentence names a checkpoint whose fitted estimate rounds to it.
    declared = [entry for entry in candidates if entry["kind"] == "constant"
                and len(_claim_words(entry) & scope_words) >= 2]
    if declared and scope_words & {"declared", "threshold", "gate", "minimum", "maximum"}:
        return declared, True
    named = [entry for entry in candidates if _attribution(entry) & scope_words]
    if named:
        candidates, strong = named, True
    elif decimals < SPECIFIC_DECIMALS:
        candidates = [entry for entry in candidates if entry["kind"] in COUNTING_KINDS]
        strong = strong and bool(candidates)
    if len({entry["family"] for entry in candidates}) > 1:
        scored = [(len(_claim_words(entry) & scope_words), entry) for entry in candidates]
        best = max(score for score, _ in scored)
        if best:
            candidates = [entry for score, entry in scored if score == best]
    return candidates, strong


def supports_named(descriptions: list[str], scope_words: set[str]) -> bool:
    """Whether the sentence names every support involved.

    A support's description carries its own counts and nouns; if one of each
    support's distinctive words appears where the quantities are quoted, the
    comparison is declared in the text and needs no exception.
    """
    for description in descriptions:
        distinctive = {word for word in _words(description)
                       if word and word not in STOPWORDS and (len(word) > 4 or word.isdigit())}
        if not distinctive & scope_words:
            return False
    return True


def agrees(candidates: list[dict], field: str) -> bool:
    """Whether every surviving candidate answers one refusal the same way.

    A refusal does not need the row pinned down; it needs the property it tests
    to be the same whichever surviving row the sentence meant.
    """
    return len({entry[field] for entry in candidates}) == 1


def anchors_live(policy: dict[str, object], text: str, findings: list[dict]) -> None:
    for name in ("cross_support_exceptions", "cross_unit_exceptions", "bare_estimate_exceptions",
                 "effective_count_exceptions", "provisional_exceptions"):
        for exception in policy.get(name, []):
            if exception["anchor"] not in text:
                findings.append({
                    "check": "dead_exception_anchor", "policy_list": name,
                    "anchor": exception["anchor"],
                    "detail": "the sentence this exception was written for is no longer in the "
                              "manuscript, so the exception is withdrawn rather than carried",
                })



def interval_convention_present(policy: dict[str, object], texts: list[str]) -> bool:
    statement = policy.get("interval_convention_statement")
    return not statement or any(statement in text for text in texts)


def exempt(policy: dict[str, object], name: str, scope: str) -> bool:
    return any(exception["anchor"] in scope for exception in policy.get(name, []))


def unit_named(unit: str, scope: str) -> bool:
    if unit not in PHYSICAL_UNITS:
        return True
    for pattern in UNIT_RENDERINGS.get(unit, ()):
        if re.search(pattern, scope):
            return True
    return False


def check(directory: Path = EVIDENCE, manuscript: Path = MANUSCRIPT) -> dict[str, object]:
    table = load_table(directory / "derived-numbers.json")
    policy = load_policy(directory)
    index = Index(table["quantities"])
    supports = table["supports"]

    findings: list[dict] = []
    ambiguous: list[dict] = []
    weak: list[dict] = []
    unmatched_bounds: list[dict] = []
    traced = 0
    untraced: list[dict] = []
    accepted = set(policy.get("untraced_accepted", []))

    sections: dict[str, int] = {}
    texts = {name: (manuscript / name).read_text()
             for name in SOURCES if (manuscript / name).is_file()}
    if not texts:
        raise FileNotFoundError(f"no manuscript source found under {manuscript}")
    if not interval_convention_present(policy, list(texts.values())):
        findings.append({"check": "interval_convention_statement_absent",
                         "detail": policy["interval_convention_statement"]})
    for name, text in texts.items():
        # Where the declarative half of each source starts. A number that does
        # not trace matters differently on each side: in the claim-bearing half
        # it is a result a reader cannot check, and in the declarative half it
        # is a design constant no declaration states.
        marker = text.find("\\section{Methods}")
        sections[name] = marker if marker >= 0 else 0
    # An exception's anchor is live when it is in any source, not in each.
    anchors_live(policy, "\n".join(texts.values()), findings)
    for name, text in texts.items():
        body = strip_markup(text)
        for kind, scope, block in scopes(body):
            intervals = [match.span() for match in INTERVAL.finditer(scope)]
            scope_words = _words(scope)
            numeric_scope = NAME.sub(lambda m: " " * len(m.group(0)), scope)
            matched: list[tuple[str, list[dict], bool, bool]] = []
            for match in NUMBER.finditer(numeric_scope):
                # A trailing thousands comma belongs to the sentence, not the
                # number: "1,614," is one count and a comma.
                token = match.group(1).rstrip(",")
                if not token or token in "+-":
                    continue
                matches = index.candidates(token) or index.tiny_candidates(token)
                candidates, strong = attributable(token, matches, scope_words)
                inside = any(start <= match.start() < end for start, end in intervals)
                if not candidates:
                    if attributable(token, index.bound_candidates(token), scope_words)[0]:
                        traced += 1
                        continue
                    if inside:
                        # An endpoint of a bracketed interval is not a point
                        # estimate, and reporting it as an untraced number of
                        # its own invites a correction to a sentence that is
                        # written correctly. It is recorded as what it is.
                        unmatched_bounds.append({"source": name, "scope_kind": kind, "token": token,
                                                 "context": scope.strip()[:180]})
                        continue
                    if token not in accepted:
                        position = texts[name].find(scope.strip()[:60])
                        if name != "main.tex":
                            half = "supplement"
                        elif sections[name] and position >= sections[name]:
                            half = "main_declarative"
                        else:
                            half = "main_claim_bearing"
                        untraced.append({"source": name, "scope_kind": kind, "token": token,
                                         "half": half, "context": scope.strip()[:220]})
                    continue
                traced += 1
                if not strong and len({entry["family"] for entry in candidates}) > 1:
                    weak.append({"source": name, "scope_kind": kind, "token": token,
                                 "families": sorted({entry["family"] for entry in candidates}),
                                 "entries": sorted({entry["id"] for entry in candidates})[:6],
                                 "context": scope.strip()[:180]})
                    continue
                matched.append((token, candidates, inside, strong))

            for token, candidates, inside, strong in matched:
                if inside:
                    continue
                families = {entry["family"] for entry in candidates}
                if len(families) > 1:
                    # The table does not attribute this number to one quantity,
                    # so no condition can be checked against it. That is worth
                    # reporting --- a number the table cannot attribute is a
                    # number a reader cannot check --- but it is not a refusal.
                    ambiguous.append({"source": name, "scope_kind": kind, "token": token,
                                      "families": sorted(families),
                                      "entries": sorted({entry["id"] for entry in candidates})[:6],
                                      "context": scope.strip()[:180]})
                    continue
                estimates = [entry for entry in candidates if entry["kind"] == "estimate"]

                # a quantity carries its unit. Only a number specific enough to
                # identify its own entry is checked: a one-decimal token that
                # happens to equal some recorded field would report a unit the
                # sentence never claimed.
                if parse(token)[1] >= SPECIFIC_DECIMALS and agrees(candidates, "unit") and \
                        all(not unit_named(entry["unit"], block if kind == "table row" else scope) for entry in candidates):
                    findings.append({
                        "check": "unit_not_named", "source": name, "scope_kind": kind, "token": token,
                        "units": sorted({entry["unit"] for entry in candidates}),
                        "entries": sorted({entry["id"] for entry in candidates})[:4],
                        "context": scope.strip()[:220],
                    })

                # no bare point estimate
                with_interval = [entry for entry in estimates if entry["interval_low"] is not None]
                if with_interval and len(with_interval) == len(candidates) and not intervals \
                        and not exempt(policy, "bare_estimate_exceptions", scope):
                    findings.append({
                        "check": "bare_point_estimate", "source": name, "scope_kind": kind,
                        "token": token, "entries": sorted({entry["id"] for entry in with_interval})[:4],
                        "detail": "the table carries an interval for this estimate and the manuscript "
                                  "quotes it without one",
                        "context": scope.strip()[:220],
                    })

                # an interval carries its resampling unit, named here or by the
                # manuscript-wide convention statement
                for entry in with_interval:
                    if not entry["resampling_unit"]:
                        findings.append({
                            "check": "interval_without_resampling_unit", "source": name,
                            "token": token, "entry": entry["id"], "context": scope.strip()[:220],
                        })

                # an effective count names its weighting convention
                effective = [entry for entry in candidates if entry["kind"] == "effective_count"]
                if effective and len(effective) == len(candidates) \
                        and not exempt(policy, "effective_count_exceptions", scope):
                    named = any(word in scope.lower()
                                for word in ("weighting", "convention", "cycle share", "cycle-share",
                                             "groups equal", "group-equal", "equally weighted", "site-pair-equal matching weights"))
                    if not named:
                        findings.append({
                            "check": "effective_count_without_convention", "source": name,
                            "scope_kind": kind, "token": token,
                            "entries": sorted({entry["id"] for entry in effective})[:4],
                            "detail": "an effective (Kish) count computed under a weighting the "
                                      "estimator does not apply is misleading even when it is close, so "
                                      "the convention has to be named where the count is quoted",
                            "context": scope.strip()[:220],
                        })

                # a provisional result is marked
                provisional = [entry for entry in candidates if entry["provisional"]]
                # Only a number specific enough to be this quantity, or one the
                # sentence names, is held to the provisional marking: a bare
                # count that happens to equal a provisional row would report a
                # coincidence.
                if provisional and len(provisional) == len(candidates) \
                        and (strong or parse(token)[1] >= SPECIFIC_DECIMALS) \
                        and not exempt(policy, "provisional_exceptions", scope):
                    if not any(marker in block.lower() for marker in PROVISIONAL_MARKERS):
                        findings.append({
                            "check": "provisional_not_marked", "source": name, "scope_kind": kind,
                            "token": token,
                            "entries": sorted({entry["id"] for entry in provisional})[:4],
                            "context": scope.strip()[:220],
                        })

            # one support per comparison: refuse only when no assignment of the
            # scope's estimates to candidate entries keeps them on one support
            estimate_tokens = [(token, [entry for entry in candidates if entry["kind"] == "estimate"])
                               for token, candidates, inside, _ in matched
                               if not inside and agrees(candidates, "support_id")]
            estimate_tokens = [(token, entries) for token, entries in estimate_tokens if entries]
            # one unit per comparison: two supports now carry different
            # measured units --- kcal/mol and log2 enrichment --- and a
            # difference between them is not a quantity at all.
            unit_tokens = [(token, [entry for entry in candidates
                                    if entry["unit"] in MEASURED_SCALES])
                           for token, candidates, inside, _ in matched if not inside]
            unit_tokens = [(token, entries) for token, entries in unit_tokens if entries]
            if len(unit_tokens) > 1 and not exempt(policy, "cross_unit_exceptions", scope):
                shared_units = None
                for _, entries in unit_tokens:
                    units = {entry["unit"] for entry in entries}
                    shared_units = units if shared_units is None else (shared_units & units)
                if not shared_units:
                    findings.append({
                        "check": "cross_unit_comparison", "source": name, "scope_kind": kind,
                        "tokens": [token for token, _ in unit_tokens],
                        "units": sorted({entry["unit"] for _, entries in unit_tokens
                                         for entry in entries}),
                        "detail": "two quantities in different measured units are quoted in one "
                                  "sentence or row; a difference between them is not a quantity",
                        "context": scope.strip()[:260],
                    })
            if len(estimate_tokens) > 1 and not exempt(policy, "cross_support_exceptions", scope):
                shared = None
                for _, entries in estimate_tokens:
                    ids = {entry["support_id"] for entry in entries}
                    shared = ids if shared is None else (shared & ids)
                involved = sorted({entry["support_id"]
                                   for _, entries in estimate_tokens for entry in entries})
                if not shared and not supports_named(
                        [supports[key] for key in involved if key in supports], scope_words):
                    findings.append({
                        "check": "cross_support_comparison", "source": name, "scope_kind": kind,
                        "tokens": [token for token, _ in estimate_tokens],
                        "supports": sorted({entry["support_id"]
                                            for _, entries in estimate_tokens for entry in entries}),
                        "detail": "two quantities that differ in their support are quoted in one "
                                  "sentence or row; the supports have to be named apart or the "
                                  "comparison declared",
                        "context": scope.strip()[:260],
                    })

    report = {
        "schema": "manuscript_evidence_consistency_report_v1",
        "manuscript": manuscript.as_posix(),
        "derived_number_table": {
            "quantities": table["counts"]["quantities"],
            "gaps": table["counts"]["gaps"],
            "supports": len(supports),
        },
        "traced_numbers": traced,
        "untraced_numbers": len(untraced),
        "untraced_split": {
            half: {
                "occurrences": sum(entry["half"] == half for entry in untraced),
                "distinct_tokens": len({entry["token"] for entry in untraced
                                        if entry["half"] == half}),
            }
            for half in ("main_claim_bearing", "main_declarative", "supplement")
        },
        "untraced": untraced,
        "numbers_the_table_cannot_attribute_to_one_quantity": len(ambiguous) + len(weak),
        "numbers_matched_but_not_pinned_to_one_row": len(weak),
        "interval_bounds_matching_no_recorded_interval": len(unmatched_bounds),
        "unmatched_interval_bounds": unmatched_bounds,
        "ambiguous_matches": ambiguous,
        "weak_attributions": weak,
        "findings_by_check": _tally(findings),
        "findings": findings,
        "status": "refused" if findings or untraced or unmatched_bounds else "pass",
    }
    (directory / "consistency-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _tally(findings: list[dict]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for finding in findings:
        tally[finding["check"]] = tally.get(finding["check"], 0) + 1
    return dict(sorted(tally.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    options = parser.parse_args()
    report = check(options.evidence_dir)
    print(json.dumps({key: report[key] for key in
                      ("status", "traced_numbers", "untraced_numbers", "findings_by_check",
                       "derived_number_table")}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
