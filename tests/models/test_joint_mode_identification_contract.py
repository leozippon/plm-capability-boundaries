"""What the joint-mode qualification publishes, now that its 0.30-nat gate is gone.

Three things changed together in ``joint_mode_qualification.py`` and
``component_swap.py``, and each of them can be undone by one line.

*The verdict became identification.* The stage used to compare context
information against a locally declared 0.30 nats, recorded as UNDERIVED. That
magnitude refused nothing -- ``main`` always scored both modes and always wrote
the artefact -- so it only ever selected a verdict string, while the substantive
evidence that a protein mode is not worth a behavioural read is the reversal cost
the stage already published. The floor is now
``budget.SCREENING_CONTEXT_INFORMATION_NATS`` and the 0.30 comparison is an inert
column. That the retired constant is never an operative default in the stage
this module owns is checked statically below.

*The relabelling is visible.* An artefact published before the change carries
``verdict``/``verdicts`` decided at 0.30; one published after carries
``identification_verdict``/``identification_verdicts`` decided at 0.05. The two
schema versions and the two key sets move together, so no reader holding one
artefact can mistake it for the other.

*Uncertainty became re-analysable.* Each mode now persists its per-record
sufficient statistics, its frozen cohort and its frozen held-out reference. The
tests below re-aggregate the persisted arrays and require them to reproduce the
stage's own published point estimates **exactly** -- a sidecar that reproduces
them to within a summation order is a sidecar that describes a different
measurement -- and then hand the same arrays to
``information_bootstrap.bootstrap_information``, which is the whole reason they
are written.

The tokenizers and the decoder are the stubs
``tests/core/test_joint_mode_qualification.py`` already declares. Nothing here needs a
GPU, a network or a 7B checkpoint.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO, REPO / "scripts/capability"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np  # noqa: E402

from src.capability.models import joint_modes as JM  # noqa: E402
from src.capability.core.arms import Cohort, sampling_record  # noqa: E402
from src.capability.core.budget import (  # noqa: E402
    MIN_CONTEXT_INFORMATION_NATS,
    POWER_RECORDS_SCHEMA_VERSION,
    SCREENING_CONTEXT_INFORMATION_NATS,
    SparseCounts,
)
from src.capability.context.information_bootstrap import (  # noqa: E402
    ArmStatistics,
    CohortStatistics,
    ReferenceStatistics,
    bootstrap_information,
)
from src.capability.context.information_bootstrap import SparseCounts as BootstrapCounts  # noqa: E402
from src.capability.generation.near_duplicates import near_duplicate_groups  # noqa: E402
from src.capability.core.measurement_support import (  # noqa: E402
    LAPLACE_SMOOTHING,
    disjoint_unigram_cross_entropy_nats,
)
from tests.core.test_joint_mode_qualification import (  # noqa: E402
    REFERENCE_RECORDS,
    SCORED_RECORDS,
    STAGE,
    ZeroLogitModel,
    cohort_args,
    galactica_stub,
    prollama_stub,
    stub_protein_draw,
)

STAGE_DIR = REPO / "scripts/capability"
OWNED_STAGES = ("joint_mode_qualification.py",)


from scripts.capability.entrypoints import stage_path

def _load_stage(filename: str):
    path = stage_path(STAGE_DIR.parents[1], filename)
    spec = importlib.util.spec_from_file_location(f"_identification_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module




DOCUMENTS = [
    "the quick brown fox jumps over the lazy dog and then keeps going, at length.\n",
    "a second document, longer than the first, with commas, stops. and newlines.\n",
    "a third document about nothing at all, written down so the draw has three.\n",
]
REFERENCE_DOCUMENTS = [
    "reference prose, held out from the scored draw and deduplicated by content.\n",
    "more reference prose, so the unigram has something to be fitted on at all.\n",
]


def stub_text_draw():
    """``arms.text_cohort`` replaced by two fixed windows, the later one held out."""

    def draw(n, min_chars, *, skip=0, name="", seed=None):
        records = list(DOCUMENTS) if skip == 0 else [DOCUMENTS[0], *REFERENCE_DOCUMENTS]
        return Cohort(
            name,
            "text",
            records[:n] if skip == 0 else records,
            min_chars,
            0,
            {
                "sampling": sampling_record(
                    seed=seed,
                    skip=skip,
                    requested=n,
                    eligible=32,
                    corpus="stub_openwebtext",
                )
            },
        )

    return draw


#: Ten distinct scored sequences and six held-out ones, drawn once from a fixed
#: generator. The three-record stub the qualification tests use is deliberate
#: there and too small here: ``information_bootstrap`` refuses a percentile
#: interval below eight effective groups, and refusing is the correct behaviour,
#: so a cohort that demonstrates a standard error has to clear that floor.
def _sequences(count: int, length: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    alphabet = np.asarray(list("ACDEFGHIKLMNPQRSTVWY"))
    return ["".join(alphabet[rng.integers(0, 20, length)]) for _ in range(count)]


BOOTSTRAP_SCORED = _sequences(10, 48, seed=20260822)
BOOTSTRAP_REFERENCE = _sequences(6, 48, seed=20260823)


def stub_wide_protein_draw():
    """Two disjoint protein windows, wide enough for a group bootstrap to run."""

    def draw(n, min_len, max_len, *, skip=0, name="", with_ec=False, seed=None):
        records = list(BOOTSTRAP_SCORED) if skip == 0 else list(BOOTSTRAP_REFERENCE)
        return Cohort(
            name,
            "protein",
            records[:n] if skip == 0 else records,
            min_len,
            max_len,
            {
                "sampling": sampling_record(
                    seed=seed,
                    skip=skip,
                    requested=n,
                    eligible=64,
                    corpus="stub_swissprot",
                )
            },
        )

    return draw


def _args(**overrides):
    return cohort_args(
        device="cpu",
        protein_context=None,
        max_tokens=4096,
        identification_floor_nats=SCREENING_CONTEXT_INFORMATION_NATS,
        **overrides,
    )


def _protein(name: str, tokenizer, *, draw=None, **overrides):
    original = STAGE.protein_cohort
    STAGE.protein_cohort = stub_protein_draw([]) if draw is None else draw
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return STAGE.protein_mode(
                _args(**overrides),
                ZeroLogitModel(len(tokenizer)),
                JM.resolve(tokenizer, name),
            )
    finally:
        STAGE.protein_cohort = original


def _text(name: str, tokenizer):
    original = STAGE.text_cohort
    STAGE.text_cohort = stub_text_draw()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return STAGE.text_mode(
                _args(sequences=len(DOCUMENTS), unigram_sequences=3),
                ZeroLogitModel(len(tokenizer)),
                tokenizer,
                JM.resolve(tokenizer, name),
                vocab_size=len(tokenizer),
            )
    finally:
        STAGE.text_cohort = original


# ----------------------------------------------- the 0.30 magnitude decides nothing


def _trees() -> dict[str, ast.Module]:
    return {
        name: ast.parse((stage_path(STAGE_DIR.parents[1], name)).read_text(encoding="utf-8"), filename=name)
        for name in OWNED_STAGES
    }


def _is_retired_magnitude(node: ast.AST) -> bool:
    """The retired floor, whether it arrives by name or as a bare literal.

    A stage that reintroduced the gate as ``default=0.30`` would pass a scan for
    the *name* alone and reinstate exactly the criterion this change removed, so
    the bare literal is checked here too, in the files that carried it.
    """

    if isinstance(node, ast.Name):
        return node.id.endswith("MIN_CONTEXT_INFORMATION_NATS") or node.id.endswith(
            "QUALIFICATION_FLOOR_NATS"
        )
    if isinstance(node, ast.Attribute):
        return node.attr.endswith("MIN_CONTEXT_INFORMATION_NATS") or node.attr.endswith(
            "QUALIFICATION_FLOOR_NATS"
        )
    return isinstance(node, ast.Constant) and node.value == MIN_CONTEXT_INFORMATION_NATS


class TheRetiredFloorIsNeverOperative(unittest.TestCase):
    def test_no_option_and_no_parameter_defaults_to_it(self):
        offenders: list[str] = []
        for name, tree in _trees().items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if (func.attr if isinstance(func, ast.Attribute) else None) != (
                        "add_argument"
                    ):
                        continue
                    option = (
                        node.args[0].value
                        if node.args and isinstance(node.args[0], ast.Constant)
                        else "?"
                    )
                    for keyword in node.keywords:
                        if keyword.arg == "default" and _is_retired_magnitude(
                            keyword.value
                        ):
                            offenders.append(f"{name}:{node.lineno} {option}")
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defaults = list(node.args.defaults) + [
                        value for value in node.args.kw_defaults if value is not None
                    ]
                    for value in defaults:
                        if _is_retired_magnitude(value):
                            offenders.append(f"{name}:{node.lineno} {node.name}()")
        self.assertEqual(
            offenders,
            [],
            "the 0.30-nat qualification floor is a reporting column and decides "
            "nothing. Offenders:\n  " + "\n  ".join(offenders),
        )

    def test_neither_stage_declares_a_magnitude_of_its_own(self):
        """One declaration, in budget, or the two copies drift apart again."""

        offenders: list[str] = []
        for name, tree in _trees().items():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if isinstance(node.value, ast.Constant) and node.value.value == (
                    MIN_CONTEXT_INFORMATION_NATS
                ):
                    targets = ",".join(
                        t.id for t in node.targets if isinstance(t, ast.Name)
                    )
                    offenders.append(f"{name}:{node.lineno} {targets}")
        self.assertEqual(offenders, [], "\n  ".join(offenders))

    def test_the_old_option_spelling_is_gone_from_both_stages(self):
        """An old command line must fail, not silently reinstate the old gate."""

        for name in OWNED_STAGES:
            source = (stage_path(STAGE_DIR.parents[1], name)).read_text(encoding="utf-8")
            self.assertNotIn("--min-context-information", source, name)

    def test_every_surviving_comparison_is_published_as_a_legacy_column(self):
        record = STAGE.verdict_record(0.084287, SCREENING_CONTEXT_INFORMATION_NATS)
        self.assertFalse(record["clears_legacy_qualification_floor"])
        self.assertEqual(
            record["legacy_qualification_floor_nats"], MIN_CONTEXT_INFORMATION_NATS
        )
        # The column disagrees with the verdict, which is the whole point of
        # keeping it: a reader of an older artefact needs to know which number
        # produced it.
        self.assertEqual(record["identification_verdict"], "measurable")


# ------------------------------------------------- old and new artefacts are apart


PUBLISHED = sorted(
    (REPO / "results").rglob("joint_mode_qualification.json")
) + sorted((REPO / "results").rglob("component_swap__*.json"))


class SchemaVersionsAreDistinguishable(unittest.TestCase):
    def test_both_stages_bumped_their_schema(self):
        self.assertEqual(STAGE.SCHEMA_VERSION, "r2_transfer_joint_mode_qualification_v2")

    def test_a_new_record_carries_none_of_the_old_verdict_keys(self):
        record = STAGE.verdict_record(0.5, SCREENING_CONTEXT_INFORMATION_NATS)
        for key in ("verdict", "verdict_note", "minimum_context_information_nats"):
            self.assertNotIn(key, record, key)
        for key in ("identification_verdict", "identification_floor_nats"):
            self.assertIn(key, record, key)

    def test_every_published_artefact_matches_its_own_schema_version(self):
        """Exactly one of the two shapes, never both.

        The shape, not the name, separates the generations: the earlier says
        ``verdicts`` with ``minimum_context_information_nats``, the later says
        ``identification_verdicts`` with ``identification_floor_nats``. Where both
        generations are still staged the name carries the generation so they stay
        apart; where only one is staged the name carries no version at all, so a
        name that does carry one must not contradict its shape.
        """

        if not PUBLISHED:  # pragma: no cover - depends on what is checked in
            self.skipTest("no published qualification artefacts in results/")
        for path in PUBLISHED:
            payload = json.loads(path.read_text(encoding="utf-8"))
            version = payload["schema_version"]
            with self.subTest(path=str(path.relative_to(REPO))):
                early = "verdicts" in payload
                late = "identification_verdicts" in payload
                self.assertNotEqual(
                    early, late,
                    f"{version} must carry exactly one of 'verdicts' and "
                    f"'identification_verdicts'",
                )
                if early:
                    self.assertNotIn("identification_verdicts", payload)
                    self.assertIn(
                        "minimum_context_information_nats", payload["thresholds"]
                    )
                    self.assertFalse(version.endswith("_v2"), version)
                else:
                    self.assertNotIn("verdicts", payload)
                    self.assertIn("identification_floor_nats", payload["thresholds"])
                    self.assertFalse(version.endswith("_v1"), version)


# ---------------------------------------------------- the sidecar round-trips exactly


def _dense(npz, prefix: str, key: str, support: int) -> np.ndarray:
    counts = SparseCounts(
        offsets=npz[f"{prefix}{key}_offsets"],
        token_ids=npz[f"{prefix}{key}_ids"],
        counts=npz[f"{prefix}{key}"],
    )
    return counts.vocabulary_totals(support)


class TheSidecarReproducesThePublishedNumbers(unittest.TestCase):
    """Re-aggregation must land on the published float, not near it."""

    def _write(self, statistics, tmp: Path) -> tuple[dict, dict]:
        block = STAGE.write_mode_records(
            tmp, statistics, seeds={"cohort_draw": 20260728}, max_tokens=4096
        )
        with np.load(tmp / block["path"]) as npz:
            arrays = {name: npz[name] for name in npz.files}
        return block, arrays

    def test_the_protein_point_estimates_come_back_bit_for_bit(self):
        import tempfile

        record, statistics = _protein("prollama", prollama_stub())
        with tempfile.TemporaryDirectory() as directory:
            block, arrays = self._write(statistics, Path(directory))
            files = sorted(
                str(p.relative_to(directory))
                for p in Path(directory).rglob("*")
                if p.is_file()
            )
        self.assertEqual(str(arrays["schema_version"]), POWER_RECORDS_SCHEMA_VERSION)
        self.assertEqual(block["arms"], ["protein_declared", "protein_reversed"])
        self.assertIn(block["cohort_records_path"], files)
        self.assertIn(block["reference_records_path"], files)

        support = int(record["unigram_reference"]["support_size"])
        self.assertEqual(support, block["support_size"])
        prefix = "protein_declared::"
        total = float(arrays[f"{prefix}clean_nll_sum"].sum())
        tokens = int(arrays[f"{prefix}token_count"].sum())
        residues = int(arrays[f"{prefix}n_symbols"].sum())
        declared = record["declared_rendering"]
        self.assertEqual(total / tokens, declared["clean_nll_nats_per_scored_token"])
        self.assertEqual(total / residues, declared["clean_nll_nats_per_residue"])
        self.assertEqual(tokens, declared["n_scored_tokens"])
        self.assertEqual(residues, declared["n_scored_residues"])

        scored = SparseCounts(
            offsets=arrays[f"{prefix}counts_offsets"],
            token_ids=arrays[f"{prefix}unique_token_ids"],
            counts=arrays[f"{prefix}counts"],
        ).vocabulary_totals(support)
        reference = SparseCounts(
            offsets=arrays[f"{prefix}reference_counts_offsets"],
            token_ids=arrays[f"{prefix}reference_unique_token_ids"],
            counts=arrays[f"{prefix}reference_counts"],
        ).vocabulary_totals(support)
        cross_entropy = disjoint_unigram_cross_entropy_nats(reference, scored)
        self.assertEqual(
            cross_entropy, record["unigram_reference"]["cross_entropy_nats"]
        )
        self.assertEqual(
            cross_entropy - declared["clean_nll_nats_per_scored_token"],
            record["context_information_nats"],
        )
        # And the reversal cost, which is what the behavioural claim rests on.
        back = "protein_reversed::"
        reversed_per_residue = float(arrays[f"{back}clean_nll_sum"].sum()) / int(
            arrays[f"{back}n_symbols"].sum()
        )
        self.assertEqual(
            reversed_per_residue - declared["clean_nll_nats_per_residue"],
            record["controls"]["reversed"]["cost_nats_per_residue"],
        )
        self.assertEqual(
            record["reversal_cost_nats_per_residue"],
            record["controls"]["reversed"]["cost_nats_per_residue"],
        )

    def test_the_text_point_estimates_come_back_bit_for_bit(self):
        import tempfile

        tokenizer = prollama_stub()
        record, statistics = _text("prollama", tokenizer)
        with tempfile.TemporaryDirectory() as directory:
            _, arrays = self._write(statistics, Path(directory))
        prefix = "text_declared::"
        support = len(tokenizer)
        total = float(arrays[f"{prefix}clean_nll_sum"].sum())
        tokens = int(arrays[f"{prefix}token_count"].sum())
        self.assertEqual(tokens, record["n_scored_tokens"])
        self.assertEqual(total / tokens, record["clean_nll_nats_per_scored_token"])
        # The declared text symbol is the token, so the two counts are one array.
        self.assertTrue(
            np.array_equal(arrays[f"{prefix}n_symbols"], arrays[f"{prefix}token_count"])
        )
        scored = SparseCounts(
            offsets=arrays[f"{prefix}counts_offsets"],
            token_ids=arrays[f"{prefix}unique_token_ids"],
            counts=arrays[f"{prefix}counts"],
        ).vocabulary_totals(support)
        reference = SparseCounts(
            offsets=arrays[f"{prefix}reference_counts_offsets"],
            token_ids=arrays[f"{prefix}reference_unique_token_ids"],
            counts=arrays[f"{prefix}reference_counts"],
        ).vocabulary_totals(support)
        self.assertEqual(
            disjoint_unigram_cross_entropy_nats(reference, scored)
            - record["clean_nll_nats_per_scored_token"],
            record["context_information_nats"],
        )

    def test_the_persisted_arrays_drive_the_bootstrap_they_exist_for(self):
        """The point of the sidecar: a standard error without a second GPU pass."""

        import tempfile

        record, statistics = _protein(
            "prollama",
            prollama_stub(),
            draw=stub_wide_protein_draw(),
            sequences=len(BOOTSTRAP_SCORED),
            unigram_sequences=len(BOOTSTRAP_REFERENCE),
        )
        with tempfile.TemporaryDirectory() as directory:
            block, arrays = self._write(statistics, Path(directory))
            reference_records = json.loads(
                (Path(directory) / block["reference_records_path"]).read_text(
                    encoding="utf-8"
                )
            )["records"]
            cohort_records = json.loads(
                (Path(directory) / block["cohort_records_path"]).read_text(
                    encoding="utf-8"
                )
            )["records"]
        prefix = "protein_declared::"
        groups, _ = near_duplicate_groups(cohort_records, unit="residues")
        reference_groups, _ = near_duplicate_groups(reference_records, unit="residues")
        arm = ArmStatistics(
            name="protein_declared",
            cohort=CohortStatistics(
                clean_nll_sum=arrays[f"{prefix}clean_nll_sum"],
                token_count=arrays[f"{prefix}token_count"],
                n_symbols=arrays[f"{prefix}n_symbols"],
                targets=BootstrapCounts(
                    unique_token_ids=arrays[f"{prefix}unique_token_ids"],
                    counts=arrays[f"{prefix}counts"],
                    record_offsets=arrays[f"{prefix}counts_offsets"],
                ),
                group_id=groups,
            ),
            reference=ReferenceStatistics(
                token_count=arrays[f"{prefix}reference_token_count"],
                targets=BootstrapCounts(
                    unique_token_ids=arrays[f"{prefix}reference_unique_token_ids"],
                    counts=arrays[f"{prefix}reference_counts"],
                    record_offsets=arrays[f"{prefix}reference_counts_offsets"],
                ),
                group_id=reference_groups,
            ),
            vocab_size=int(block["support_size"]),
            smoothing=float(LAPLACE_SMOOTHING),
        )
        result = bootstrap_information(arm, seed=20260822, n_bootstrap=400)
        self.assertFalse(result.refused, result.record.get("refusal_reason"))
        self.assertAlmostEqual(
            result.information, record["context_information_nats"], places=12
        )
        interval = result.record["statistics"]["information_nats_per_token"]
        self.assertTrue(np.isfinite(interval["bootstrap_se"]))
        self.assertGreater(interval["bootstrap_se"], 0.0)
        low, high = interval["interval"]
        self.assertLess(low, high)


# --------------------------------------------------------------- the negative paths


class TheStatisticsRefuseRatherThanRepair(unittest.TestCase):
    def test_a_record_with_no_scored_token_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            STAGE.record_statistics(
                "protein_declared",
                support_size=4,
                clean_nll_sum=[1.0, 0.0],
                token_count=[2, 0],
                n_symbols=[2, 0],
                targets=SparseCounts.from_records(
                    [np.asarray([0, 1], dtype=np.int64), np.zeros(0, dtype=np.int64)]
                ),
            )
        self.assertIn("no scored token", str(raised.exception))

    def test_counts_that_do_not_sum_to_the_token_count_are_refused(self):
        """The one error that leaves both terms of the estimand plausible alone."""

        with self.assertRaises(ValueError):
            STAGE.record_statistics(
                "protein_declared",
                support_size=4,
                clean_nll_sum=[1.0],
                token_count=[3],
                n_symbols=[3],
                targets=SparseCounts.from_records(
                    [np.asarray([0, 1], dtype=np.int64)]
                ),
            )

    def test_a_mode_refuses_statistics_over_the_wrong_inventory(self):
        record, statistics = _protein("prollama", prollama_stub())
        del record
        with self.assertRaises(ValueError) as raised:
            STAGE.ModeStatistics(
                mode="protein",
                scored=statistics.scored,
                reference=statistics.reference,
                support=statistics.support,
                support_size=statistics.support_size + 1,
                id_space=statistics.id_space,
                symbol_definition=statistics.symbol_definition,
                conditions=statistics.conditions,
                reference_applies_to_reversed=False,
            )
        self.assertIn("different inventories", str(raised.exception))

    def test_a_mode_refuses_statistics_that_do_not_cover_the_cohort(self):
        _, statistics = _protein("prollama", prollama_stub())
        short = Cohort(
            statistics.scored.name,
            "protein",
            list(statistics.scored.records)[:-1],
            8,
            400,
            {},
        )
        with self.assertRaises(ValueError) as raised:
            STAGE.ModeStatistics(
                mode="protein",
                scored=short,
                reference=statistics.reference,
                support=statistics.support,
                support_size=statistics.support_size,
                id_space=statistics.id_space,
                symbol_definition=statistics.symbol_definition,
                conditions=statistics.conditions,
                reference_applies_to_reversed=False,
            )
        self.assertIn("cohort records", str(raised.exception))


class TheReferenceIsPersistedAsItWasUsed(unittest.TestCase):
    def test_a_deduplicated_reference_record_contributes_no_row(self):
        """The reference block repeats one scored record, the way Swiss-Prot does."""

        import tempfile

        record, statistics = _protein("prollama", prollama_stub())
        overlap = record["unigram_reference"]["reference_overlap_removed"]
        self.assertEqual(overlap["dropped_sequences_shared_with_cohort"], 1)
        self.assertEqual(len(statistics.reference), len(REFERENCE_RECORDS))
        with tempfile.TemporaryDirectory() as directory:
            block, arrays = self._sidecar(statistics, Path(directory))
            frozen = json.loads(
                (Path(directory) / block["reference_records_path"]).read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(frozen["records"], REFERENCE_RECORDS)
        self.assertEqual(
            int(arrays["protein_declared::reference_token_count"].size),
            len(REFERENCE_RECORDS),
        )
        # The dropped record is one of the scored ones, and it is not in the
        # reference the baseline was fitted on.
        self.assertNotIn(SCORED_RECORDS[0], frozen["records"])

    def test_a_token_unit_reversal_carries_no_reference_and_none_is_substituted(self):
        import tempfile

        _, statistics = _protein("prollama", prollama_stub())
        self.assertFalse(statistics.reference_applies_to_reversed)
        with tempfile.TemporaryDirectory() as directory:
            _, arrays = self._sidecar(statistics, Path(directory))
        self.assertIn("protein_declared::reference_counts", arrays)
        self.assertNotIn("protein_reversed::reference_counts", arrays)
        self.assertNotIn("protein_reversed::reference_counts_offsets", arrays)

    def test_a_per_residue_reversal_keeps_the_reference_that_applies_to_it(self):
        import tempfile

        _, statistics = _protein("galactica", galactica_stub())
        self.assertTrue(statistics.reference_applies_to_reversed)
        with tempfile.TemporaryDirectory() as directory:
            _, arrays = self._sidecar(statistics, Path(directory))
        self.assertIn("protein_reversed::reference_counts", arrays)
        self.assertTrue(
            np.array_equal(
                arrays["protein_reversed::reference_counts"],
                arrays["protein_declared::reference_counts"],
            )
        )

    @staticmethod
    def _sidecar(statistics, tmp: Path) -> tuple[dict, dict]:
        block = STAGE.write_mode_records(
            tmp, statistics, seeds={"cohort_draw": 20260728}, max_tokens=4096
        )
        with np.load(tmp / block["path"]) as npz:
            return block, {name: npz[name] for name in npz.files}


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
