"""Conditions that must hold when a reviewed subset's provenance is established.

A provenance document is only worth reading if it refuses to be written from
inputs that do not match its own declarations, and a reference MSA is only worth
qualifying if its held-out split is a function of the aligned rows rather than of
the experimental panel that sits beside them. These tests fix the refusals --
digest mismatch, missing alignment columns, a wild type that does not bind across
the allowlist, cohort, plan and query set -- and fix the fit's own input contract:
a singles panel is refused explicitly and a paired cohort missing its cycles is
refused rather than fitted with an empty contrast list.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.capability.context.homology import ALIGNMENT_FIELDS, DIAMOND_FIELDS, Hit, parse_hits
from src.capability.context.profiles import build_profile


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROVENANCE = load(ROOT / "scripts/capability/interactions/verify_pairwise_provenance.py",
                  "verify_pairwise_provenance")
FIT = load(ROOT / "scripts/capability/interactions/fit_pairwise_sequence_baseline.py",
           "fit_pairwise_sequence_baseline")
FIT_SCRIPT = ROOT / "scripts/capability/interactions/fit_pairwise_sequence_baseline.py"
PROVENANCE_SCRIPT = ROOT / "scripts/capability/interactions/verify_pairwise_provenance.py"
BUILDER_SCRIPT = ROOT / "scripts/capability/interactions/build_pairwise_profile_features.py"

WILDTYPE = "ACDEFGHIKLMNPQRSTVWY"
REVIEWED = ["1E0L.pdb", "1F0M.pdb", "1IFY.pdb", "1K1V.pdb", "1UFM.pdb", "1V1C.pdb",
            "2K1B.pdb", "2KWH.pdb", "2LGW.pdb", "2LVN.pdb", "3DKM.pdb", "3L1X.pdb"]


def hit(subject: str = "s1", *, query: str = "q", bitscore: float = 100.0,
        qseq: str = WILDTYPE, sseq: str | None = None) -> Hit:
    return Hit(query=query, subject=subject, pident=100.0, length=len(WILDTYPE),
               nident=len(WILDTYPE), qstart=1, qend=len(WILDTYPE), qlen=len(WILDTYPE),
               slen=len(WILDTYPE), evalue=1e-9, bitscore=bitscore, qseq_gapped=qseq,
               sseq_gapped=sseq if sseq is not None else qseq)


def hits_tsv(hits: list[Hit]) -> str:
    """The retained DIAMOND table in the field order the search was run under."""

    rows = []
    for entry in hits:
        rows.append("\t".join(str(value) for value in (
            entry.query, entry.subject, entry.pident, entry.length, entry.nident,
            entry.qstart, entry.qend, entry.qlen, entry.slen, entry.evalue, entry.bitscore,
            entry.qseq_gapped, entry.sseq_gapped)))
    return "\n".join(rows) + "\n"


def cohort_row(name: str, wildtype: str = WILDTYPE, variants: int = 256, **extra) -> dict:
    return {"name": name, "group": f"nat-{name}", "cluster": 1, "length": len(wildtype),
            "wildtype": wildtype, "variants": [""] * variants, **extra}


def cohort(rows: list[dict], schema: str = PROVENANCE.SINGLES_SCHEMA) -> dict:
    return {"schema": schema, "backgrounds": rows}


class DigestRefusals(unittest.TestCase):
    def test_a_declared_digest_that_disagrees_with_the_bytes_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "asset.bin"
            path.write_bytes(b"payload")
            observed = PROVENANCE.sha256_file(path)
            with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "sha256"):
                PROVENANCE.asset_record(path, "0" * 64)
            record = PROVENANCE.asset_record(path, observed)
        self.assertTrue(record["recomputed"])
        self.assertEqual(record["sha256"], observed)

    def test_an_undeclared_digest_cannot_be_adopted_without_reading_the_asset(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "asset.bin"
            path.write_bytes(b"payload")
            with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "undeclared"):
                PROVENANCE.asset_record(path, None, recompute=False)

    def test_a_named_asset_that_is_not_on_disk_is_refused(self):
        with self.assertRaises(FileNotFoundError):
            PROVENANCE.asset_record(Path("/nonexistent/named/asset"), None)


class AlignmentRefusals(unittest.TestCase):
    def test_a_hit_table_without_the_aligned_columns_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "hits.tsv"
            path.write_text("q\ts1\t100.0\t20\t20\t1\t20\t20\t20\t1e-9\t100.0\n")
            with self.assertRaisesRegex(ValueError, "expected"):
                parse_hits(path, fields=ALIGNMENT_FIELDS)
            parsed = parse_hits(path, fields=DIAMOND_FIELDS)
        self.assertIsNone(parsed[0].qseq_gapped)

    def test_an_empty_alignment_is_refused(self):
        with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "empty alignment"):
            PROVENANCE.require_aligned_fields("q", [hit(sseq="")])

    def test_column_mismatched_aligned_strings_are_refused(self):
        with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "columns"):
            PROVENANCE.require_aligned_fields("q", [hit(sseq=WILDTYPE + "A")])

    def test_a_hit_without_aligned_strings_is_refused(self):
        bare = Hit(query="q", subject="s1", pident=100.0, length=20, nident=20, qstart=1,
                   qend=20, qlen=20, slen=20, evalue=1e-9, bitscore=100.0)
        with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "ALIGNMENT_FIELDS"):
            PROVENANCE.require_aligned_fields("q", [bare])


class ReferenceBindings(unittest.TestCase):
    def bind(self, rows, *, queries=None, plan_rows=None):
        plan = {"schema": PROVENANCE.PLAN_SCHEMA,
                "backgrounds": plan_rows or [{"name": r["name"], "wildtype": r["wildtype"]}
                                             for r in rows]}
        bound = queries if queries is not None else {r["name"]: r["wildtype"] for r in rows}
        return PROVENANCE.bind_references([r["name"] for r in rows], cohort(rows), plan, bound)

    def test_a_wild_type_that_differs_from_the_query_is_refused(self):
        rows = [cohort_row("A.pdb")]
        with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "query sequence"):
            self.bind(rows, queries={"A.pdb": "ACDEFGHIKLMNPQRSTVWA"})

    def test_a_wild_type_that_differs_from_the_plan_is_refused(self):
        rows = [cohort_row("A.pdb")]
        with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "plan wild type"):
            self.bind(rows, plan_rows=[{"name": "A.pdb", "wildtype": "ACDE"}])

    def test_a_background_absent_from_the_query_set_is_refused(self):
        with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "query record"):
            self.bind([cohort_row("A.pdb")], queries={})

    def test_a_variant_panel_that_is_not_the_reviewed_size_is_refused(self):
        with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "variants"):
            self.bind([cohort_row("A.pdb", variants=255)])

    def test_a_cohort_of_the_wrong_schema_is_refused(self):
        plan = {"schema": PROVENANCE.PLAN_SCHEMA,
                "backgrounds": [{"name": "A.pdb", "wildtype": WILDTYPE}]}
        with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "schema"):
            PROVENANCE.bind_references(["A.pdb"],
                                       {"schema": PROVENANCE.PAIRWISE_SCHEMA,
                                        "backgrounds": [cohort_row("A.pdb")]},
                                       plan, {"A.pdb": WILDTYPE})

    def test_a_complete_binding_records_the_reviewed_panel(self):
        bound = self.bind([cohort_row("A.pdb"), cohort_row("B.pdb")])
        self.assertEqual([b["length"] for b in bound], [len(WILDTYPE)] * 2)
        self.assertTrue(all(b["variants"] == 256 for b in bound))
        self.assertTrue(all(not b["pairwise_cycles_present"] for b in bound))


class HeldOutCriterion(unittest.TestCase):
    def test_the_split_is_a_function_of_the_aligned_rows_and_not_of_their_order(self):
        rows = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 4], [1, 1, 1, 1]],
                        dtype=np.int8)
        mask = FIT.held_out_mask(rows, FIT.SPLIT_SEED)
        # Order independence: a row carries its own side, so a permutation permutes the mask.
        shuffled = rows[[2, 0, 3, 1]]
        self.assertTrue(np.array_equal(FIT.held_out_mask(shuffled, FIT.SPLIT_SEED),
                                       mask[[2, 0, 3, 1]]))
        # Purity: the split is a deterministic function of the row bytes alone.
        self.assertTrue(np.array_equal(FIT.held_out_mask(rows, FIT.SPLIT_SEED), mask))
        # Identity, not position: a repeated row lands on the side its bytes give it.
        repeated = np.vstack([rows, rows[0]])
        self.assertEqual(bool(FIT.held_out_mask(repeated, FIT.SPLIT_SEED)[-1]), bool(mask[0]))
        # No cross-row leakage: appending an unrelated row cannot move an existing one.
        extra = np.array([[9, 9, 9, 9]], dtype=rows.dtype)
        appended = FIT.held_out_mask(np.vstack([rows, extra]), FIT.SPLIT_SEED)
        self.assertTrue(np.array_equal(appended[:len(rows)], mask))

    def test_the_reference_record_takes_no_label_or_measurement_input(self):
        parameters = set(inspect.signature(PROVENANCE.reference_record).parameters)
        for forbidden in ("cycles", "variants", "measurements", "epsilon", "labels"):
            self.assertNotIn(forbidden, parameters)

    def test_the_reference_record_reports_natural_rows_and_binds_the_support_floors(self):
        hits = [hit(f"s{i}") for i in range(FIT.MIN_TRAIN_ROWS + FIT.MIN_HELD_OUT_ROWS)]
        record, rows, profile = PROVENANCE.reference_record(
            "q", WILDTYPE, hits, FIT, max_sequences=20000, ceiling=10000)
        self.assertEqual(record["reference_inputs"], "natural homolog rows only")
        self.assertEqual(record["retained_rows"], len(hits))
        self.assertIn(record["status"], {"supported", "insufficient_homolog_support"})
        self.assertEqual(record["train_rows"] + record["held_out_rows"], len(rows))
        self.assertIsNotNone(profile)
        self.assertNotIn("variants", record)
        self.assertNotIn("epsilon", record)

    def test_a_background_with_no_homolog_rows_is_reported_rather_than_fitted(self):
        record, rows, profile = PROVENANCE.reference_record(
            "q", WILDTYPE, [], FIT, max_sequences=20000, ceiling=10000)
        self.assertEqual(record["status"], "no_homolog_hits")
        self.assertIsNone(rows)
        self.assertIsNone(profile)


class ProfileParity(unittest.TestCase):
    def cached(self, name: str, profile) -> dict:
        return {f"{name}|frequencies": profile.frequencies.astype(np.float32),
                f"{name}|column_weight": profile.column_weight.astype(np.float32),
                f"{name}|scalars": np.array([profile.neff, profile.log10_neff,
                                             profile.max_identity_over_query,
                                             profile.n_sequences, profile.n_hits])}

    def test_a_rebuilt_profile_reproduces_its_own_cache(self):
        hits = [hit(f"s{i}", bitscore=100.0 - i) for i in range(5)]
        profile = build_profile(WILDTYPE, "q", hits, max_sequences=20000)
        record = PROVENANCE.profile_parity("q", profile, self.cached("q", profile))
        self.assertTrue(record["reproduced"])
        self.assertEqual(record["scalars_max_absolute_difference"], 0.0)

    def test_a_changed_cached_frequency_is_refused(self):
        hits = [hit(f"s{i}", bitscore=100.0 - i) for i in range(5)]
        profile = build_profile(WILDTYPE, "q", hits, max_sequences=20000)
        cache = self.cached("q", profile)
        cache["q|frequencies"] = cache["q|frequencies"].copy()
        cache["q|frequencies"][0, 0] += 1.0
        with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "frequencies"):
            PROVENANCE.profile_parity("q", profile, cache)

    def test_a_changed_cached_scalar_is_refused(self):
        hits = [hit(f"s{i}", bitscore=100.0 - i) for i in range(5)]
        profile = build_profile(WILDTYPE, "q", hits, max_sequences=20000)
        cache = self.cached("q", profile)
        cache["q|scalars"] = cache["q|scalars"].copy()
        cache["q|scalars"][0] += 1.0
        with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "scalars"):
            PROVENANCE.profile_parity("q", profile, cache)

    def test_the_unchanged_builder_reproduces_the_cached_arrays(self):
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            names = ["A.pdb", "B.pdb"]
            plan_path = work / "plan.json"
            plan_path.write_text(json.dumps({
                "schema": PROVENANCE.PLAN_SCHEMA,
                "backgrounds": [{"name": name, "wildtype": WILDTYPE} for name in names]}))
            hits_path = work / "hits.tsv"
            hits_path.write_text(hits_tsv(
                [hit(f"{name}_s{i}", query=name, bitscore=100.0 - i)
                 for name in names for i in range(3)]))
            first = work / "first.npz"
            subprocess.run([sys.executable, str(BUILDER_SCRIPT), "--plan", str(plan_path),
                            "--hits", str(hits_path), "--out", str(first)],
                           check=True, capture_output=True, text=True, cwd=str(ROOT))
            with np.load(first, allow_pickle=False) as archive:
                cache = {key: archive[key] for key in archive.files}
            second = work / "second.npz"
            result = PROVENANCE.builder_rebuild(BUILDER_SCRIPT, plan_path, hits_path, second,
                                                names, cache)
            self.assertFalse(second == first)
            self.assertTrue(all(record["reproduced"] for record in result["backgrounds"]))
            cache["A.pdb|frequencies"] = cache["A.pdb|frequencies"] + 1.0
            with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "frequencies"):
                PROVENANCE.builder_rebuild(BUILDER_SCRIPT, plan_path, hits_path,
                                           work / "third.npz", names, cache)


class GroupRelatedness(unittest.TestCase):
    def test_a_duplicated_group_label_is_refused(self):
        bindings = [{"name": "A.pdb", "group": "nat-001"}, {"name": "B.pdb", "group": "nat-001"}]
        with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "one family"):
            PROVENANCE.relatedness(bindings, {"schema": PROVENANCE.GROUPING_SCHEMA,
                                              "assignments": {"A.pdb": "nat-001", "B.pdb": "nat-001"}},
                                   {})

    def test_a_label_that_disagrees_with_the_frozen_assignment_is_refused(self):
        bindings = [{"name": "A.pdb", "group": "nat-001"}]
        with self.assertRaisesRegex(PROVENANCE.ProvenanceMismatch, "frozen group"):
            PROVENANCE.relatedness(bindings, {"schema": PROVENANCE.GROUPING_SCHEMA,
                                              "assignments": {"A.pdb": "nat-002"}}, {})

    def test_within_subset_purge_pairs_are_reported_by_threshold(self):
        bindings = [{"name": "A.pdb", "group": "nat-001"}, {"name": "B.pdb", "group": "nat-002"}]
        result = PROVENANCE.relatedness(
            bindings, {"schema": PROVENANCE.GROUPING_SCHEMA,
                       "assignments": {"A.pdb": "nat-001", "B.pdb": "nat-002"}},
            {"identity20_coverage50": {"nat-001": ["nat-002", "nat-009"]}})
        self.assertEqual(result["remote_purge_within_allowlist"]["identity20_coverage50"],
                         [("nat-001", "nat-002")])
        self.assertEqual(result["distinct_groups"], 2)


class PairwiseFitContract(unittest.TestCase):
    """The fit's input contract: cycle-required, refused explicitly when absent."""

    def run_fit(self, cohort_path: Path, out: Path) -> subprocess.CompletedProcess:
        (out / "plmc").write_bytes(b"not a real binary; the guard runs before it is used")
        return subprocess.run(
            [sys.executable, str(FIT_SCRIPT), "--cohort", str(cohort_path),
             "--hits", str(out / "hits.tsv"), "--plmc", str(out / "plmc"),
             "--out-dir", str(out), "--retrieval-ceiling", "10000", "--workers", "1"],
            capture_output=True, text=True, cwd=str(ROOT))

    def test_a_singles_cohort_is_refused_before_anything_is_read(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            (out / "hits.tsv").write_text("")
            cohort_path = out / "singles.json"
            cohort_path.write_text(json.dumps(
                cohort([cohort_row("A.pdb")], schema=PROVENANCE.SINGLES_SCHEMA)))
            completed = self.run_fit(cohort_path, out)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("stability_singles_cohort_v1", completed.stderr)
        self.assertIn("unsupported", completed.stderr)

    def test_a_paired_cohort_missing_its_cycles_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            (out / "hits.tsv").write_text("")
            cohort_path = out / "paired.json"
            cohort_path.write_text(json.dumps(
                cohort([{"name": "A.pdb", "group": "nat-001", "length": len(WILDTYPE)}],
                       schema=PROVENANCE.PAIRWISE_SCHEMA)))
            completed = self.run_fit(cohort_path, out)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("no pairwise cycles", completed.stderr)

    def test_a_paired_cohort_with_cycles_passes_the_contract_guard(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            (out / "hits.tsv").write_text("")
            cohort_path = out / "paired.json"
            cohort_path.write_text(json.dumps(
                cohort([{"name": "A.pdb", "group": "nat-001", "length": len(WILDTYPE),
                         "cycles": [{"sequences": ["AA", "CA", "AC", "CC"]}]}],
                       schema=PROVENANCE.PAIRWISE_SCHEMA)))
            completed = self.run_fit(cohort_path, out)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((out / "baseline_q.json").read_text())
        self.assertEqual(report["summary"]["backgrounds"], 1)
        self.assertEqual(report["summary"]["fitted"], 0)


REAL_ASSETS = {
    "hits": ROOT / "archive/logs/R3/pairwise_homologs_20260924/search/hits.tsv",
    "cohort": ROOT / "results/R4/gate_stability_20260924/cohort/cohort.json",
}


@unittest.skipIf(not all(path.exists() for path in REAL_ASSETS.values()),
                 "the retained retrieval and cohort are not on this host")
@unittest.skipUnless(os.environ.get("PROVENANCE_ASSETS") == "1",
                     "set PROVENANCE_ASSETS=1: this rebuilds 12 reference alignments, "
                     "including a 9841-row reweighting, against the real retrieval")
class ReviewedSubsetIntegration(unittest.TestCase):
    """The real 12, bound and rebuilt, without a search, a fit or a corpus re-hash."""

    def test_the_reviewed_subset_binds_and_rebuilds_its_reference_alignments(self):
        with tempfile.TemporaryDirectory() as raw:
            completed = subprocess.run(
                [sys.executable, str(PROVENANCE_SCRIPT), "--out-dir", raw,
                 "--no-hash-corpora", "--skip-builder-rebuild", "--skip-profile-parity"],
                capture_output=True, text=True, cwd=str(ROOT))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            document = json.loads((Path(raw) / "pairwise_provenance.json").read_text())
        subset = document["reviewed_subset"]
        self.assertEqual(subset["count"], 12)
        self.assertEqual(subset["length_range"], [37, 72])
        self.assertEqual(subset["length_sum"], 678)
        self.assertEqual(sorted(subset["names"]), REVIEWED)
        self.assertEqual(document["assets"]["allowlist"]["sha256"],
                         "840f4a1566cfedaae3a450a82113b28185a01a7daab3a2d352ea212993ecf168")
        self.assertEqual(document["assets"]["hits"]["sha256"],
                         "1aa8ed8addd6e0f851919e451ed4c9daefefb35e79663421695029615b2e4d78")
        self.assertEqual(document["verifier_script"]["sha256"],
                         PROVENANCE.sha256_file(PROVENANCE_SCRIPT))
        references = {record["name"]: record for record in document["references"]}
        self.assertEqual(sorted(references), REVIEWED)
        self.assertTrue(all(record["status"] == "supported" for record in references.values()))
        self.assertEqual([record["name"] for record in document["references"]
                          if record["retrieval_ceiling_reached"]], ["2LGW.pdb"])
        self.assertEqual(document["group_relatedness"]["distinct_groups"], 12)
        self.assertEqual(document["group_relatedness"]["remote_purge_within_allowlist"],
                         {"identity20_coverage50": [["nat-015", "nat-052"],
                                                    ["nat-025", "nat-078"]],
                          "identity20_coverage80": []})
        self.assertEqual(document["singles_contract"]["cohort_schema"],
                         PROVENANCE.SINGLES_SCHEMA)
        self.assertIsNone(document["profile_parity"]["rebuilt_by_builder"])
        self.assertEqual(document["profile_parity"]["rebuilt_in_process"], [])


if __name__ == "__main__":
    unittest.main()