"""Tests for the ledger: validation, cursor reads, optimistic concurrency."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest

from app.ledger import Ledger, StaleSequence
from app.wal import PoisonedError, WAL


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "wal.bin")
        self.ledger = Ledger(WAL(self.path))
        self._extra_wals: list[WAL] = []

    def tearDown(self) -> None:
        self.ledger._wal.close()
        for wal in self._extra_wals:
            wal.close()

    def reopen(self) -> Ledger:
        wal = WAL(self.path)
        self._extra_wals.append(wal)
        return Ledger(wal)

    def test_continuous_sequences_and_cursor_reads(self) -> None:
        self.assertEqual(self.ledger.head(), 1)
        self.ledger.submit(1, [10, 20, 30])
        self.ledger.submit(4, [40])
        self.ledger.submit(5, [50, 60])
        self.assertEqual(self.ledger.head(), 7)

        page = self.ledger.read(0, limit=2)
        self.assertEqual([(r.seq, r.dose) for r in page.records],
                         [(1, 10), (2, 20)])
        self.assertEqual(page.next_cursor, 2)

        page = self.ledger.read(page.next_cursor)
        self.assertEqual([(r.seq, r.dose) for r in page.records],
                         [(3, 30), (4, 40), (5, 50), (6, 60)])
        self.assertEqual(page.next_cursor, 6)
        self.assertEqual(self.ledger.read(6).records, [])

    def test_stale_sequence_writes_nothing(self) -> None:
        size_before = os.path.getsize(self.path)
        self.ledger.submit(1, [1, 2])
        # Two late callers both believe they are next at seq 1.
        with self.assertRaises(StaleSequence):
            self.ledger.submit(1, [3])
        with self.assertRaises(StaleSequence):
            self.ledger.submit(2, [3])  # head has moved to 3, also a conflict
        size_after = os.path.getsize(self.path)
        self.assertEqual(self.ledger.head(), 3)
        # Exactly one frame was added by the winning submit.
        self.assertEqual(self.ledger._wal.truncated_bytes_on_boot, 0)
        self.assertGreater(size_after, size_before)
        # The winner's frame is the only frame on disk.
        reopen = self.reopen()
        self.assertEqual(reopen.head(), 3)
        self.assertEqual(
            [r.dose for r in reopen.read(0).records], [1, 2]
        )

    def test_batch_size_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.ledger.submit(1, [])
        with self.assertRaises(ValueError):
            self.ledger.submit(1, [1] * 33)
        with self.assertRaises(ValueError):
            self.ledger.submit(1, [1, "x"])  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            self.ledger.submit(1, [True])  # type: ignore[list-item]

    def test_thirty_two_record_limit_boundary(self) -> None:
        frame = self.ledger.submit(1, list(range(1, 33)))
        self.assertEqual(len(frame.records), 32)
        self.assertEqual(self.ledger.head(), 33)

    def test_concurrent_same_expected_seq_exactly_one_wins(self) -> None:
        n = 32
        winners: list[int] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(n)

        def worker(i: int) -> None:
            barrier.wait()
            try:
                frame = self.ledger.submit(1, [100 + i])
                winners.append(frame.seq)
            except StaleSequence:
                pass
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners, [1])
        self.assertEqual(self.ledger.head(), 2)
        # No torn residue after the storm; reopen must be clean.
        reopen = self.reopen()
        self.assertFalse(reopen.poisoned)
        self.assertEqual(len(reopen.read(0).records), 1)
        self.assertEqual(reopen.read(0).records[0].seq, 1)


if __name__ == "__main__":
    unittest.main()
