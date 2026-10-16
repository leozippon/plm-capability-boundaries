#!/usr/bin/env python3
"""Stage the compact manuscript package and write its manifest.

``manuscript/`` is a working-tree snapshot that Git does not track, so this
script is the single place that decides what the package contains and the only
thing that writes ``manuscript/package-manifest.json``. Every size and digest
it records is read from the bytes on disk; nothing is asserted.

Derived here, so that it cannot drift from the repository:

* the packaged code, as the transitive first-party import closure of the entry
  points declared below, function-level and sibling imports included, so a
  module an entry point starts importing is packaged the same day;
* the renderer's declared inputs and display tables, read out of the figure
  validation receipt, for any the curated inventory does not already cover;
* the ``layout`` block, read out of the compiled PDFs with pdfinfo/pdftotext;
* every ``bytes``, ``sha256`` and ``source_sha256``.

Declared here: those entry points, and the two files under ``code/`` that have
no repository source.

Carried forward from the existing manifest, which is the only record of it: the
curated data and asset inventory with each entry's recorded source path, the
schema and snapshot notes, the symlink records, and the provenance history in
``data_supplements`` and ``manuscript_updates``. A recorded source path is
never rewritten; where the receipt names a table the inventory does not carry,
the gap is filled rather than the record replaced.

A packaged copy must be byte-identical to its source, so a copy that differs is
re-staged from the source. ``data/evidence-ledger.csv.gz`` is the one entry
with an encoding: it is re-gzipped without a timestamp so its bytes are
reproducible, and it is verified by what it decompresses to. Caches are never
packaged. Anything the refresh cannot establish -- a source that has
disappeared, a recorded symlink that no longer points where it did, a figure
whose caption page cannot be read -- ends the run with the paths named instead
of being written as a guess.

Run from the repository root in the validated ct environment:

    python scripts/capability/reporting/build_manuscript_package.py --check
    python scripts/capability/reporting/build_manuscript_package.py --note "why"
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import ast
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.capability.reporting import publication_paths  # noqa: E402

PACKAGE = REPO_ROOT / "manuscript"
MANIFEST_NAME = "package-manifest.json"
SCHEMA = "compact-manuscript-handoff-v1"

#: Entry points whose closure is packaged under ``code/analysis/`` at
#: repository paths, so the packaged copies import each other by the names the
#: repository uses.
ANALYSIS_ENTRY_POINTS = (
    "scripts/capability/generation/analyse_e20_confirmatory.py",
    "scripts/capability/generation/analyse_progen3_robustness.py",
    "scripts/capability/reporting/build_derived_numbers.py",
    "scripts/capability/reporting/build_ncs_figures.py",
    "scripts/capability/reporting/check_manuscript_evidence.py",
    "scripts/capability/reporting/regenerate_manuscript_evidence.py",
)

#: Entry points whose closure is packaged flat under ``code/``, where
#: ``run_figures.py`` stages them into a repository-shaped temporary workspace.
FLAT_ENTRY_POINTS = (
    "scripts/capability/reporting/export_prediction_details.py",
    "scripts/capability/reporting/render_figures.py",
    "scripts/capability/reporting/render_framework.py",
)

#: The only packaged files with no repository source: the package's own
#: instructions, and the flat-layout adapter that exists nowhere else. The
#: adapter's absence of a source is recorded on its manifest entry, because
#: ``manuscript/`` is outside Git and nothing else preserves that file.
PACKAGE_ONLY_CODE = {
    "code/README.md": None,
    "code/run_figures.py": ("package-only: no repository source, so Git carries no history of "
                            "this file and only a copy of the package preserves it"),
}

#: The library module the packaged analysis carries, because it defines the
#: ledger's own schema, refusals and evidence directory rather than measuring
#: anything, and it needs nothing but the standard library.
LIBRARY_CARRIED = ("src/capability/core/evidence_ledger.py",)

#: The library modules the packaged analysis imports and deliberately does not
#: carry; ``code/README.md`` names each one and what the packaged code reads
#: from it. Their own imports are relative, so copying them would pull most of
#: the measurement library, the checkpoint-scoring layer included, into a
#: traceability handoff. A dependency that appears here neither carried nor
#: recorded ends the run: the decision is a human one.
LIBRARY_REFERENCED = (
    "src/capability/context/cross_measure_association.py",
    "src/capability/context/family_potts.py",
    "src/capability/context/potts_gauge.py",
    "src/capability/core/io.py",
    "src/capability/core/statistics.py",
    "src/capability/decoding/guided_decoding.py",
    "src/capability/forcing/forcing_design.py",
    "src/capability/generation/__init__.py",
    "src/capability/generation/fold_recapitulation.py",
    "src/capability/interactions/pairwise_epistasis.py",
)

FIGURE_RECEIPT = "scripts/capability/reporting/figure_data/main/figure_validation.json"
CACHE_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                        ".ipynb_checkpoints"})
CACHE_SUFFIXES = frozenset({".pyc", ".pyo"})

DOCUMENTS = {"main": "submission/main.pdf",
             "supplementary-information": "submission/supplementary-information.pdf"}
CAPTIONS = {"main": re.compile(r"^[ \t]*Fig\.[ \t]+(\d+)[ \t]+[A-Z]", re.M),
            "supplementary-information": re.compile(
                r"^[ \t]*Supplementary Fig\.[ \t]+(\d+)[ \t]+[A-Z]", re.M)}
REFERENCES = re.compile(r"^[ \t]*References[ \t]*$", re.M)


def digest_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def gzip_bytes(data: bytes) -> bytes:
    """Gzip without a timestamp, so the packaged bytes are reproducible."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as handle:
        handle.write(data)
    return buffer.getvalue()


def content_digest(path: Path, encoding: str | None) -> str:
    """Digest of what a packaged file carries, not of how it is stored."""
    if encoding is None:
        return digest_file(path)
    if encoding != "gzip":
        raise SystemExit(f"{path}: unsupported encoding {encoding!r}")
    return hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()


def imported_names(path: Path) -> set[str]:
    """Every module name a file imports, at module level or inside a function."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def resolve(name: str, importer: str) -> str | None:
    """Repository path of a first-party module, or None for anything else.

    ``scripts.*`` and ``src.*`` resolve against the repository root. A bare name
    resolves beside the importing file, which is how these scripts import their
    siblings: a script's own directory leads ``sys.path``.
    """
    parts = name.split(".")
    if parts[0] in ("scripts", "src"):
        base = Path(".")
    elif len(parts) == 1:
        base = Path(importer).parent
    else:
        return None
    target = base.joinpath(*parts)
    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
        if (REPO_ROOT / candidate).is_file():
            return candidate.as_posix()
    return None


def closure(entry_points: tuple[str, ...]) -> tuple[set[str], set[str]]:
    """The scripts the entry points reach, and the library modules they name.

    Traversal stops at the library: a ``src`` module is recorded as part of the
    derivation's dependency surface but not followed, because the library's own
    imports are relative and their closure is most of ``src/capability``,
    including the checkpoint-scoring layer this package does not carry.
    """
    scripts: set[str] = set()
    library: set[str] = set()
    pending = list(entry_points)
    while pending:
        module = pending.pop()
        if module in scripts:
            continue
        scripts.add(module)
        for name in sorted(imported_names(REPO_ROOT / module)):
            target = resolve(name, module)
            if target is None:
                continue
            if target.startswith("src/"):
                library.add(target)
            else:
                pending.append(target)
    return scripts, library


def derived_code() -> tuple[dict[str, str], set[str]]:
    """Package path -> source for the packaged code, and the library surface."""
    staged: dict[str, str] = {}
    analysis, library = closure(ANALYSIS_ENTRY_POINTS)
    for module in sorted(analysis | set(LIBRARY_CARRIED)):
        staged[f"code/analysis/{module}"] = module
    flat, flat_library = closure(FLAT_ENTRY_POINTS)
    for module in sorted(flat):
        name = f"code/{Path(module).name}"
        if staged.setdefault(name, module) != module:
            raise SystemExit(f"{name}: {staged[name]} and {module} cannot share one flat name")
    return staged, library | flat_library


def renderer_inputs() -> dict[str, str]:
    """Package path -> source for what the figure receipt declares.

    The receipt is the renderer's own record of the tables and insets it read
    and the display tables it wrote, so a new figure's inputs are packaged
    without anyone listing them.
    """
    receipt = json.loads((REPO_ROOT / FIGURE_RECEIPT).read_text())
    inputs: dict[str, str] = {}
    for source in receipt["source_sha256"]:
        # The structure insets are the one input the renderer reads from a copy
        # of the package rather than the other way round: the inset bundle is
        # built from the authored images under ``assets/``, so attributing those
        # images to it would invert their provenance.
        if Path(source).parent.name != "structure-insets":
            inputs[f"data/{Path(source).name}"] = source
    for check in receipt["checks"]:
        plotted = check["plotted_source_path"]
        # Main display tables are packaged under their submission names; the
        # supplementary ones keep the figure identifier they are written as.
        stem = (Path(check["pdf_path"]).stem if Path(plotted).parent.name == "main"
                else check["figure"])
        inputs[f"data/{stem}-plotted.csv"] = plotted
    return inputs


def curated_inventory(previous: dict) -> dict[str, tuple[str, str | None]]:
    """The non-code entries' recorded sources, which only the manifest holds."""
    return {entry["path"]: (entry["source"], entry.get("encoding"))
            for entry in previous["files"]
            if "source" in entry and not entry["path"].startswith("code/")}


def package_contents(package: Path) -> tuple[list[str], list[str], list[str]]:
    """Package-relative regular files, symlinks and cache paths."""
    files: list[str] = []
    symlinks: list[str] = []
    caches: list[str] = []
    for directory, names, filenames in os.walk(package):
        here = Path(directory)
        kept = []
        for name in sorted(names):
            if name in CACHE_DIRS:
                caches.append((here / name).relative_to(package).as_posix())
            else:
                kept.append(name)
        names[:] = kept
        for name in sorted(filenames):
            path = here / name
            relative = path.relative_to(package).as_posix()
            if relative == MANIFEST_NAME:
                continue
            if path.suffix in CACHE_SUFFIXES:
                caches.append(relative)
            elif path.is_symlink():
                symlinks.append(relative)
            else:
                files.append(relative)
    return sorted(files), sorted(symlinks), sorted(caches)


def missing_modules(derived: dict[str, str], package: Path = PACKAGE) -> list[str]:
    """Derived code the package does not carry, or carries differently.

    This is the invariant the package exists for: a packaged entry point that
    imports a module the package does not hold fails on import and reproduces
    nothing.
    """
    faults = []
    for path, source in sorted(derived.items()):
        target = package / path
        if not target.is_file():
            faults.append(f"{path}: not packaged, but the entry points import {source}")
        elif digest_file(target) != digest_file(REPO_ROOT / source):
            faults.append(f"{path}: differs from {source}")
    return faults


def adapter_modules(package: Path = PACKAGE) -> tuple[str, ...]:
    """The flat module names ``run_figures.py`` stages, read from the adapter."""
    for node in ast.walk(ast.parse((package / "code/run_figures.py").read_text())):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "SOURCES"
                for target in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise SystemExit("code/run_figures.py no longer declares SOURCES")


def pdf_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True).stdout
    except FileNotFoundError as error:
        raise SystemExit(f"{command[0]} is needed to read the compiled layout: {error}")


def compiled_layout(package: Path = PACKAGE) -> dict:
    """Page counts, figure pages and reference pages, read out of the PDFs."""
    main_figures = sum(1 for key in publication_paths.PDF_NAMES if key.startswith("Fig"))
    expected = {"main": main_figures,
                "supplementary-information": len(publication_paths.PDF_NAMES) - main_figures}
    record = {}
    for name, relative in DOCUMENTS.items():
        pdf = package / relative
        count = re.search(r"^Pages:\s+(\d+)$", pdf_output(["pdfinfo", str(pdf)]), re.M)
        if count is None:
            raise SystemExit(f"{relative}: pdfinfo reports no page count")
        total = int(count[1])
        pages = pdf_output(["pdftotext", "-layout", str(pdf), "-"]).split("\f")[:total]
        figures: dict[str, int] = {}
        for number, text in enumerate(pages, 1):
            for figure in CAPTIONS[name].findall(text):
                figures.setdefault(figure, number)
        if sorted(int(key) for key in figures) != list(range(1, expected[name] + 1)):
            raise SystemExit(
                f"{relative}: captions found for figures {sorted(figures, key=int)}, "
                f"against {expected[name]} declared in publication_paths")
        record[name] = {
            "pages": total,
            "figure_pages": {key: figures[key] for key in sorted(figures, key=int)},
            "references_pages": [number for number, text in enumerate(pages, 1)
                                 if REFERENCES.search(text)]}
    return record


def stage(sources: dict[str, tuple[str, str | None]], write: bool,
          problems: list[str]) -> list[str]:
    """Make every packaged copy carry exactly what its source carries."""
    staged = []
    for path, (source, encoding) in sorted(sources.items()):
        origin = REPO_ROOT / source
        if not origin.is_file():
            problems.append(f"{path}: recorded source is gone: {source}")
            continue
        target = PACKAGE / path
        if target.is_file() and content_digest(target, encoding) == digest_file(origin):
            continue
        if not write:
            problems.append(f"{path}: absent; its source is {source}" if not target.is_file()
                            else f"{path}: differs from its source {source}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if encoding == "gzip":
            target.write_bytes(gzip_bytes(origin.read_bytes()))
        else:
            shutil.copyfile(origin, target)
        staged.append(path)
    return staged


def remove(paths: list[str], write: bool, problems: list[str], reason: str) -> None:
    for relative in paths:
        if not write:
            problems.append(f"{relative}: {reason}")
            continue
        target = PACKAGE / relative
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()


def manifest_entries(files: list[str],
                     sources: dict[str, tuple[str, str | None]]) -> list[dict]:
    entries = []
    for relative in files:
        path = PACKAGE / relative
        entry = {"path": relative, "bytes": path.stat().st_size, "sha256": digest_file(path)}
        if PACKAGE_ONLY_CODE.get(relative) is not None:
            entry["provenance"] = PACKAGE_ONLY_CODE[relative]
        if relative in sources:
            source, encoding = sources[relative]
            if encoding is not None:
                entry["encoding"] = encoding
            entry["source"] = source
            entry["source_sha256"] = digest_file(REPO_ROOT / source)
        entries.append(entry)
    return entries


def git_head() -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                          check=True, text=True, capture_output=True).stdout.strip()


def arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage the manuscript package and its manifest.")
    parser.add_argument("--check", action="store_true",
                        help="verify the package against its sources and manifest; write nothing")
    parser.add_argument("--note", help="what this refresh did; required when it changes an entry")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    write = not args.check
    manifest_path = PACKAGE / MANIFEST_NAME
    if not manifest_path.is_file():
        raise SystemExit(f"{manifest_path}: no manifest to refresh; the curated inventory and "
                         "the provenance history exist nowhere else")
    previous = json.loads(manifest_path.read_text())
    if previous.get("schema") != SCHEMA:
        raise SystemExit(f"{manifest_path}: unexpected schema {previous.get('schema')!r}")
    problems: list[str] = []

    derived, library = derived_code()
    for module in sorted(library - set(LIBRARY_CARRIED) - set(LIBRARY_REFERENCED)):
        problems.append(f"{module}: the packaged code imports it, but it is neither carried "
                        "nor recorded as deliberately referenced")
    for module in sorted(set(LIBRARY_REFERENCED) - library):
        problems.append(f"{module}: recorded as referenced, but the packaged code no longer "
                        "imports it")
    declared_inputs = renderer_inputs()
    sources = curated_inventory(previous)
    for path, source in declared_inputs.items():
        sources.setdefault(path, (source, None))
    sources.update({path: (source, None) for path, source in derived.items()})

    _, _, caches = package_contents(PACKAGE)
    remove(caches, write, problems, "generated cache; never part of the package")
    staged = stage(sources, write, problems)

    files, symlinks, caches = package_contents(PACKAGE)
    strays = [path for path in files if path.startswith("code/")
              and path not in sources and path not in PACKAGE_ONLY_CODE]
    remove(strays, write, problems, "under code/ with no repository source and not declared")
    if write:
        files = [path for path in files if path not in strays]

    problems.extend(missing_modules(derived))
    for path, source in sorted(declared_inputs.items()):
        if path not in files:
            problems.append(f"{path}: the figure receipt declares {source}, but it is not packaged")
    for path in PACKAGE_ONLY_CODE:
        if path not in files:
            problems.append(f"{path}: declared package-only file is missing")
    if "code/run_figures.py" in files:
        for name in adapter_modules():
            if f"code/{name}" not in files:
                problems.append(f"code/{name}: run_figures.py stages it, but it is not packaged")
    recorded_links = {link["path"]: link["target"] for link in previous["links"]}
    for path in symlinks:
        if recorded_links.get(path) != os.readlink(PACKAGE / path):
            problems.append(f"{path}: symlink does not match the recorded link record")
    for path in recorded_links:
        if path not in symlinks:
            problems.append(f"{path}: recorded link is no longer a symlink in the package")
    if write:
        for path in caches:
            problems.append(f"{path}: cache reappeared while staging")

    entries = manifest_entries(files, sources)
    layout = compiled_layout()
    before = {entry["path"]: entry for entry in previous["files"]}
    after = {entry["path"]: entry for entry in entries}
    changed = sorted(path for path in set(before) | set(after)
                     if before.get(path) != after.get(path))
    if layout != previous.get("layout"):
        changed.append("layout")

    print(f"package: {len(entries)} files, {len(derived)} code files derived from the import "
          f"closure of {len(ANALYSIS_ENTRY_POINTS) + len(FLAT_ENTRY_POINTS)} entry points, "
          f"{len(library)} library modules on the dependency surface, {len(symlinks)} links")
    for path in staged:
        print(f"  staged  {path}")
    if write:
        for path in strays + caches:
            print(f"  removed {path}")
    for path in changed:
        print(f"  changed {path}")
    for problem in problems:
        print(f"  FAULT   {problem}")
    if problems:
        print(f"{len(problems)} fault(s); the manifest was not written" if write
              else f"{len(problems)} fault(s)")
        return 1
    if args.check:
        print("manifest, package and sources agree")
        return 0
    if changed and not args.note:
        print(f"{len(changed)} recorded entries move; rerun with --note to record why")
        return 1

    supplements = list(previous["data_supplements"])
    if changed:
        # The history is dated in local time, as the snapshots and the logbook
        # are; ``created_utc`` is the one field that carries the UTC instant.
        supplements.append({"date": datetime.now().strftime("%Y-%m-%d"),
                            "paths": changed, "note": args.note})
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_head": git_head(),
        "snapshot_note": previous["snapshot_note"],
        "data_supplements": supplements,
        "layout": layout,
        "files": entries,
        "links": previous["links"],
        "manuscript_updates": previous["manuscript_updates"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {manifest_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
