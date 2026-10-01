"""What the corpus k-mer background must get right to be worth counting.

The background exists to answer one objection: that a protein decoder scoring a
homologue-free referent above chance is reproducing corpus fragment statistics.
An objection is only answered by a number that is right, and three ways of
getting it wrong all produce a number that looks right.

**A k-mer that spans two records.** Concatenating 60 million records and sliding
a window over the join produces fragments that exist in no protein, and they are
systematically unusual because they join a C-terminus to an N-terminus. This is
the invariant the whole count rests on and it is checked directly, by requiring
that a fragment which exists *only* across a boundary is never counted.

**A k-mer that is dropped at a line break.** The natural way to buy the first
property is to treat the newline as an invalid symbol -- and that also discards
every window spanning FASTA line wrapping, one in twenty at k = 3 on a
60-column file. Those are real windows of real proteins. The test that catches
it wraps a record and requires the counts to equal the unwrapped record's.

**A k-mer lost or double-counted at a chunk boundary.** The corpus does not fit
in memory. A chunk boundary is not a record boundary, and a carry bug moves the
count by a few parts in a million -- far too little to notice and quite enough to
matter if the background is ever the thing a positive result is defended
against. The test runs one file at chunk sizes from a single byte upward and
requires every count to be identical.

Two smaller properties are checked because they are cheap and silent: header text
is protein-shaped and must contribute nothing, and a non-canonical residue must
drop every window containing it rather than being folded into a neighbour.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.capability.context.kmer_background import (  # noqa: E402
    ALPHABET,
    KmerBackground,
    count_kmers,
    kmer_index,
    load,
    save,
)


def _write(directory: Path, name: str, text: str) -> Path:
    path = Path(directory) / name
    path.write_text(text, encoding="utf-8")
    return path


class KmerIndexing(unittest.TestCase):
    """The encoding a consumer needs in order to look one k-mer up."""

    def test_index_is_base_twenty_in_alphabet_order(self) -> None:
        self.assertEqual(kmer_index("A"), 0)
        self.assertEqual(kmer_index("AAA"), 0)
        self.assertEqual(kmer_index("AAC"), 1)
        self.assertEqual(kmer_index("ACA"), len(ALPHABET))
        self.assertEqual(kmer_index("YYY"), len(ALPHABET) ** 3 - 1)

    def test_a_symbol_outside_the_alphabet_is_refused(self) -> None:
        for kmer in ("AXA", "aaa", "A*A"):
            with self.assertRaises(ValueError):
                kmer_index(kmer)


class RecordBoundaries(unittest.TestCase):
    """No window may reach from one record into the next."""

    def test_a_fragment_that_exists_only_across_the_join_is_never_counted(self) -> None:
        with TemporaryDirectory() as work:
            fasta = _write(work, "two.faa", ">one\nAAAAA\n>two\nCCCCC\n")
            background = count_kmers(fasta, (3,))
            counts = background.counts[3]
            self.assertEqual(int(counts[kmer_index("AAA")]), 3)
            self.assertEqual(int(counts[kmer_index("CCC")]), 3)
            # These exist only if the two records were concatenated.
            for crossing in ("AAC", "ACC"):
                self.assertEqual(int(counts[kmer_index(crossing)]), 0, crossing)
            self.assertEqual(int(counts.sum()), 6)
            self.assertEqual(background.records, 2)
            self.assertEqual(background.residues, 10)

    def test_adjacent_records_do_not_merge_when_the_second_is_short(self) -> None:
        # A record shorter than k contributes nothing and must not lend its
        # residues to its neighbour.
        with TemporaryDirectory() as work:
            fasta = _write(work, "short.faa", ">a\nAAAA\n>b\nC\n>c\nDDDD\n")
            counts = count_kmers(fasta, (3,)).counts[3]
            self.assertEqual(int(counts[kmer_index("AAA")]), 2)
            self.assertEqual(int(counts[kmer_index("DDD")]), 2)
            self.assertEqual(int(counts.sum()), 4)

    def test_header_text_is_not_counted_as_sequence(self) -> None:
        # The header is protein-shaped on purpose: an implementation that only
        # strips the '>' would count it and nothing would look wrong.
        with TemporaryDirectory() as work:
            plain = count_kmers(_write(work, "p.faa", ">x\nACDEF\n"), (3,))
            loaded = count_kmers(_write(work, "l.faa", ">ACDEFACDEF ACDEF\nACDEF\n"), (3,))
            self.assertTrue(np.array_equal(plain.counts[3], loaded.counts[3]))
            self.assertEqual(loaded.residues, 5)
            self.assertEqual(int(loaded.counts[3].sum()), 3)


class LineWrapping(unittest.TestCase):
    """Wrapping is a property of the file, not of the protein."""

    def test_a_wrapped_record_counts_the_same_as_an_unwrapped_one(self) -> None:
        sequence = "ACDEFGHIKLMNPQRSTVWY" * 5
        with TemporaryDirectory() as work:
            flat = count_kmers(_write(work, "flat.faa", f">x\n{sequence}\n"), (3, 4))
            wrapped_text = "\n".join(
                sequence[start : start + 7] for start in range(0, len(sequence), 7)
            )
            wrapped = count_kmers(_write(work, "wrap.faa", f">x\n{wrapped_text}\n"), (3, 4))
            for k in (3, 4):
                self.assertTrue(
                    np.array_equal(flat.counts[k], wrapped.counts[k]),
                    f"line wrapping changed the k = {k} counts",
                )
            self.assertEqual(flat.residues, wrapped.residues)

    def test_the_window_total_is_the_record_local_total(self) -> None:
        # One record of n residues holds exactly n - k + 1 windows. An
        # implementation that drops line breaks scores below this; one that
        # concatenates records scores above it.
        sequence = "ACDEFGHIKLMNPQRSTVWY" * 3
        with TemporaryDirectory() as work:
            wrapped = "\n".join(
                sequence[start : start + 11] for start in range(0, len(sequence), 11)
            )
            background = count_kmers(_write(work, "w.faa", f">x\n{wrapped}\n"), (3, 4))
            for k in (3, 4):
                self.assertEqual(int(background.counts[k].sum()), len(sequence) - k + 1)


class NonCanonicalResidues(unittest.TestCase):
    """An unknown residue drops its windows; it is never read as something else."""

    def test_every_window_containing_an_unknown_residue_is_dropped(self) -> None:
        with TemporaryDirectory() as work:
            background = count_kmers(_write(work, "x.faa", ">x\nAAXAA\n"), (3,))
            self.assertEqual(int(background.counts[3].sum()), 0)
            self.assertEqual(background.residues, 4)

    def test_windows_on_either_side_of_an_unknown_residue_survive(self) -> None:
        with TemporaryDirectory() as work:
            counts = count_kmers(_write(work, "x.faa", ">x\nAAAXCCC\n"), (3,)).counts[3]
            self.assertEqual(int(counts[kmer_index("AAA")]), 1)
            self.assertEqual(int(counts[kmer_index("CCC")]), 1)
            self.assertEqual(int(counts.sum()), 2)

    def test_lowercase_is_not_silently_uppercased(self) -> None:
        # Soft-masked FASTA is common. Reading 'a' as 'A' would make a masked
        # region indistinguishable from an unmasked one.
        with TemporaryDirectory() as work:
            background = count_kmers(_write(work, "s.faa", ">x\naaaaa\n"), (3,))
            self.assertEqual(int(background.counts[3].sum()), 0)
            self.assertEqual(background.residues, 0)


class ChunkBoundaries(unittest.TestCase):
    """The counts must not depend on how the file was read."""

    def _corpus(self, directory: Path) -> Path:
        rng = np.random.default_rng(20260812)
        records = []
        for number in range(23):
            length = int(rng.integers(1, 40))
            residues = "".join(ALPHABET[i] for i in rng.integers(0, len(ALPHABET), length))
            wrapped = "\n".join(residues[s : s + 9] for s in range(0, len(residues), 9))
            records.append(f">record_{number} some ACDEF text\n{wrapped}")
        return _write(directory, "corpus.faa", "\n".join(records) + "\n")

    def test_every_chunk_size_gives_the_same_counts(self) -> None:
        with TemporaryDirectory() as work:
            fasta = self._corpus(Path(work))
            reference = count_kmers(fasta, (3, 4), chunk_bytes=1 << 20)
            for chunk in (1, 2, 3, 5, 7, 13, 64, 512, 4096):
                observed = count_kmers(fasta, (3, 4), chunk_bytes=chunk)
                self.assertEqual(observed.residues, reference.residues, chunk)
                self.assertEqual(observed.records, reference.records, chunk)
                for k in (3, 4):
                    self.assertTrue(
                        np.array_equal(observed.counts[k], reference.counts[k]),
                        f"chunk_bytes={chunk} changed the k = {k} counts",
                    )

    def test_a_file_without_a_final_newline_keeps_its_last_record(self) -> None:
        with TemporaryDirectory() as work:
            with_newline = count_kmers(_write(work, "a.faa", ">x\nAAAA\n>y\nCCCC\n"), (3,))
            without = count_kmers(_write(work, "b.faa", ">x\nAAAA\n>y\nCCCC"), (3,))
            self.assertTrue(np.array_equal(with_newline.counts[3], without.counts[3]))
            self.assertEqual(without.records, 2)
            self.assertEqual(without.residues, 8)


class Reporting(unittest.TestCase):
    """The record and the round-trip a consumer depends on."""

    def _background(self, work: Path) -> KmerBackground:
        sequence = "ACDEFGHIKLMNPQRSTVWY" * 4
        return count_kmers(_write(work, "r.faa", f">x\n{sequence}\n"), (3, 4))

    def test_the_record_states_coverage_and_the_window_rule(self) -> None:
        with TemporaryDirectory() as work:
            record = self._background(Path(work)).record()
            self.assertEqual(record["schema_version"], "kmer_background_v1")
            self.assertEqual(record["alphabet"], ALPHABET)
            self.assertEqual(record["counts"]["3"]["possible"], len(ALPHABET) ** 3)
            self.assertEqual(record["counts"]["4"]["possible"], len(ALPHABET) ** 4)
            self.assertIn("record boundary", record["window_rule"])

    def test_frequencies_normalise_and_refuse_an_empty_background(self) -> None:
        with TemporaryDirectory() as work:
            background = self._background(Path(work))
            self.assertAlmostEqual(float(background.frequencies(3).sum()), 1.0, places=12)
            empty = count_kmers(_write(work, "e.faa", ">x\nAX\n"), (3,))
            with self.assertRaises(ValueError):
                empty.frequencies(3)

    def test_save_and_load_round_trip(self) -> None:
        with TemporaryDirectory() as work:
            background = self._background(Path(work))
            out = Path(work) / "bg"
            save(background, out)
            restored = load(out)
            self.assertEqual(restored.ks, background.ks)
            self.assertEqual(restored.residues, background.residues)
            self.assertEqual(restored.records, background.records)
            for k in background.ks:
                self.assertTrue(np.array_equal(restored.counts[k], background.counts[k]))

    def test_load_refuses_counts_whose_digest_has_moved(self) -> None:
        with TemporaryDirectory() as work:
            background = self._background(Path(work))
            out = Path(work) / "bg"
            save(background, out)
            corrupted = np.load(out / "kmer_counts_k3.npy")
            corrupted[0] += 1
            np.save(out / "kmer_counts_k3.npy", corrupted)
            with self.assertRaises(RuntimeError):
                load(out)


class Refusals(unittest.TestCase):
    """Configuration faults fail where they are made."""

    def test_a_missing_corpus_and_an_invalid_request_are_refused(self) -> None:
        with TemporaryDirectory() as work:
            fasta = _write(work, "r.faa", ">x\nACDEF\n")
            with self.assertRaises(FileNotFoundError):
                count_kmers(Path(work) / "absent.faa", (3,))
            with self.assertRaises(ValueError):
                count_kmers(fasta, ())
            with self.assertRaises(ValueError):
                count_kmers(fasta, (0,))
            with self.assertRaises(ValueError):
                count_kmers(fasta, (3,), chunk_bytes=0)

    def test_a_malformed_background_is_refused_at_construction(self) -> None:
        good = np.zeros(len(ALPHABET) ** 3, dtype=np.int64)
        with self.assertRaises(ValueError):
            KmerBackground({}, 0, 0, Path("x"), 0, 0.0)
        with self.assertRaises(ValueError):
            KmerBackground({3: np.zeros(7, dtype=np.int64)}, 0, 0, Path("x"), 0, 0.0)
        with self.assertRaises(ValueError):
            KmerBackground({3: good.astype(np.int32)}, 0, 0, Path("x"), 0, 0.0)


class HeldOutSubtraction(unittest.TestCase):
    """Counts are additive over records, which is what makes a held-out estimate cheap.

    The higher-order fragment channel is admitted on held-out cross-entropy, and
    the training counts for that estimate are the corpus counts minus the counts
    of the held-out records rather than a second 24 GB pass. That is only exact
    because no window spans a record; if it ever stopped being exact the held-out
    number would be optimistic and nothing else would look wrong.
    """

    def test_the_corpus_minus_a_held_out_subset_is_the_complement(self) -> None:
        records = [
            "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
            "MKVLAAGIVGLNLGGWLAAQ",
            "PPPPQQQQRRRRSSSSTTTT",
            "WYACDEFGHIKLMNPQRSTV",
            "GGGGGGGGGGGGGGGGGGGGGGGG",
        ]
        held = {1, 3}
        with TemporaryDirectory() as work:
            def write(name, chosen):
                lines = []
                for index in chosen:
                    lines.append(f">r{index}")
                    body = records[index]
                    lines.extend(body[at : at + 7] for at in range(0, len(body), 7))
                return _write(work, name, "\n".join(lines) + "\n")

            whole = count_kmers(write("all.faa", range(len(records))), (3, 4, 5))
            subset = count_kmers(write("held.faa", sorted(held)), (3, 4, 5))
            rest = count_kmers(
                write("rest.faa", [i for i in range(len(records)) if i not in held]),
                (3, 4, 5),
            )
            for k in (3, 4, 5):
                np.testing.assert_array_equal(
                    whole.counts[k] - subset.counts[k], rest.counts[k]
                )
            self.assertEqual(whole.residues - subset.residues, rest.residues)


if __name__ == "__main__":
    unittest.main()
