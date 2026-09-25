"""Tests for seal frames, persisted roots and recovery cross-checks."""

from __future__ import annotations

import os
import tempfile
import unittest

from app.ledger import Ledger, StaleSequence
from app.merkle import (
    LEFT,
    RIGHT,
    Proof,
    inner_digest,
    leaf_digest,
    merkle_root,
    verify_proof,
)
from app.wal import (
    MAGIC_SEAL,
    PoisonedError,
    WAL,
    canonical_seal_payload,
    encode_frame,
)


class SealFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "wal.bin")
        self._opened: list[WAL] = []

    def tearDown(self) -> None:
        for wal in self._opened:
            wal.close()

    def open_wal(self) -> WAL:
        wal = WAL(self.path)
        self._opened.append(wal)
        return wal

    def raw(self) -> bytearray:
        with open(self.path, "rb") as fh:
            return bytearray(fh.read())

    def write_raw(self, data: bytes) -> None:
        with open(self.path, "wb") as fh:
            fh.write(data)


def _records(n: int):
    return [(i, i * 11 - 4) for i in range(1, n + 1)]


class LedgerSealTests(SealFixture):
    def _populated(self, batches=((10, 20, 30), (40, 50))) -> Ledger:
        ledger = Ledger(self.open_wal())
        head = 1
        for batch in batches:
            ledger.submit(head, list(batch))
            head += len(batch)
        return ledger

    def test_seal_fixes_root_count_and_id(self) -> None:
        ledger = self._populated()
        seal = ledger.seal(6)
        recs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
        self.assertEqual(seal.count, 5)
        self.assertEqual(seal.root, merkle_root(recs))
        # seal_id is the first half of the root hex, so it identifies the
        # sealed root without a second lookup channel.
        self.assertEqual(seal.seal_id, seal.root[:16].hex())

    def test_seal_requires_current_head(self) -> None:
        ledger = self._populated()  # head == 6
        with self.assertRaises(StaleSequence):
            ledger.seal(5)
        # A failed seal wrote nothing: still one seal possible afterwards.
        seal = ledger.seal(6)
        self.assertEqual(seal.count, 5)

    def test_empty_prefix_cannot_be_sealed(self) -> None:
        ledger = Ledger(self.open_wal())
        with self.assertRaises(ValueError):
            ledger.seal(1)

    def test_repeated_seals_over_growing_prefixes(self) -> None:
        ledger = Ledger(self.open_wal())
        ledger.submit(1, [1, 2])
        first = ledger.seal(3)
        self.assertEqual(first.count, 2)
        ledger.submit(3, [3])
        second = ledger.seal(4)
        self.assertEqual(second.count, 3)
        self.assertNotEqual(first.seal_id, second.seal_id)
        # The older seal stays citable at its original prefix.
        p = ledger.proof(first.seal_id, 1)
        self.assertEqual(p.count, 2)
        self.assertTrue(verify_proof(p))
        self.assertEqual(p.root, first.root)

    def test_proof_is_repeatable_and_verifiable(self) -> None:
        ledger = self._populated()
        seal = ledger.seal(6)
        p1 = ledger.proof(seal.seal_id, 3)
        p2 = ledger.proof(seal.seal_id, 3)
        self.assertEqual(p1, p2)
        self.assertEqual(p1.leaf, leaf_digest(3, 30))
        self.assertTrue(verify_proof(p1))
        rec = ledger.sealed_record(seal.seal_id, 3)
        self.assertEqual((rec.seq, rec.dose), (3, 30))

    def test_proof_outside_prefix_and_unknown_seal(self) -> None:
        ledger = self._populated()
        seal = ledger.seal(6)
        ledger.submit(6, [60])
        # seq 6 exists in the log but was not part of this sealed prefix.
        with self.assertRaises(LookupError):
            ledger.proof(seal.seal_id, 6)
        with self.assertRaises(KeyError):
            ledger.proof("deadbeef" * 4, 1)
        with self.assertRaises(ValueError):
            ledger.proof(seal.seal_id, 0)

    def test_proof_survives_restart(self) -> None:
        ledger = self._populated()
        seal = ledger.seal(6)
        ledger._wal.close()
        self._opened.clear()
        reopened = Ledger(self.open_wal())
        proof = reopened.proof(seal.seal_id, 2)
        self.assertTrue(verify_proof(proof))
        self.assertEqual(proof.root, seal.root)

    def test_concurrent_seals_same_head_are_idempotent_one_frame(self) -> None:
        import threading

        ledger = self._populated()  # head == 6
        size_before = os.path.getsize(self.path)
        n = 12
        barrier = threading.Barrier(n)
        seals: list = []
        errors: list = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            try:
                seal = ledger.seal(6)
                with lock:
                    seals.append(seal)
            except Exception as exc:  # pragma: no cover
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        # Sealing the identical prefix is idempotent: every caller observes
        # the same seal point ...
        self.assertEqual(len(seals), n)
        self.assertEqual({s.seal_id for s in seals}, {seals[0].seal_id})
        self.assertEqual({s.count for s in seals}, {5})
        # ... but exactly one physical seal frame exists.
        ledger._wal.close()
        self._opened.clear()
        reopened = Ledger(self.open_wal())
        self.assertEqual(len(reopened._wal.seals()), 1)
        # One frame beyond the two record frames grew on disk.
        self.assertGreater(os.path.getsize(self.path), size_before)

    def test_stale_head_seal_conflicts_and_writes_nothing(self) -> None:
        ledger = self._populated()  # head == 6
        size_before = os.path.getsize(self.path)
        with self.assertRaises(StaleSequence):
            ledger.seal(5)
        self.assertEqual(os.path.getsize(self.path), size_before)
        ledger._wal.close()
        self._opened.clear()
        self.assertEqual(Ledger(self.open_wal())._wal.seals(), [])


class SealFrameTests(SealFixture):
    def test_seal_and_record_frames_share_one_ordinal_space(self) -> None:
        wal = self.open_wal()
        wal.append_batch([1, 2])           # physical frame 1
        root = merkle_root([(1, 1), (2, 2)])
        seal = wal.append_seal(root[:16].hex(), 2, root)  # frame 2
        wal.append_batch([3])              # physical frame 3
        self.assertEqual(
            [f.frame_no for f in wal.snapshot()], [1, 3])
        self.assertEqual([s.frame_no for s in wal.seals()], [2])
        self.assertEqual(seal.seal_id, root[:16].hex())
        wal.close()
        self._opened.remove(wal)

        wal2 = self.open_wal()
        self.assertFalse(wal2.poisoned)
        self.assertEqual(wal2.next_seq, 4)
        self.assertEqual([s.root for s in wal2.seals()], [root])

    def test_torn_tail_truncates_a_half_written_seal_frame(self) -> None:
        wal = self.open_wal()
        wal.append_batch([1])
        wal.close()
        self._opened.remove(wal)
        good = bytes(self.raw())
        root = merkle_root([(1, 1)])
        half = encode_frame(
            2, canonical_seal_payload(root[:16].hex(), 1, root), MAGIC_SEAL
        )[:25]
        self.write_raw(good + half)
        wal2 = self.open_wal()
        self.assertFalse(wal2.poisoned)
        self.assertEqual(wal2.seals(), [])
        self.assertEqual(self.raw(), good)

    def test_seal_root_disagreement_poisons(self) -> None:
        # A structurally complete, digest-valid seal frame whose root does
        # NOT recompute from the preceding records must poison on restart:
        # a persisted root is never trusted.
        wal = self.open_wal()
        wal.append_batch([1, 2])
        wal.close()
        self._opened.remove(wal)
        data = bytes(self.raw())
        fake_root = b"\xab" * 32
        bogus = encode_frame(
            2,
            canonical_seal_payload(fake_root[:16].hex(), 2, fake_root),
            MAGIC_SEAL,
        )
        self.write_raw(data + bogus)
        wal2 = self.open_wal()
        self.assertTrue(wal2.poisoned)
        with self.assertRaises(PoisonedError):
            wal2.seals()
        with self.assertRaises(PoisonedError):
            wal2.append_batch([9])

    def test_seal_count_disagreement_poisons(self) -> None:
        wal = self.open_wal()
        wal.append_batch([1, 2])
        root = merkle_root([(1, 1), (2, 2)])
        wal.close()
        self._opened.remove(wal)
        data = bytes(self.raw())
        bogus = encode_frame(
            2,
            canonical_seal_payload(root[:16].hex(), 3, root),  # count 3
            MAGIC_SEAL,
        )
        self.write_raw(data + bogus)
        wal2 = self.open_wal()
        self.assertTrue(wal2.poisoned)

    def test_bad_seal_magic_with_matching_ordinal_poisons(self) -> None:
        wal = self.open_wal()
        wal.append_batch([1])
        root = merkle_root([(1, 1)])
        payload = canonical_seal_payload(root[:16].hex(), 1, root)
        frame = encode_frame(2, payload, MAGIC_SEAL)
        wal.close()
        self._opened.remove(wal)
        data = bytearray(bytes(self.raw()) + frame)
        data[0:8] = b"XXXXXXXX"  # corrupt the *first* frame's magic
        self.write_raw(bytes(data))
        self.assertTrue(self.open_wal().poisoned)

    def test_ordinal_gap_when_seal_frame_deleted_poisons(self) -> None:
        wal = self.open_wal()
        wal.append_batch([1])
        root = merkle_root([(1, 1)])
        seal_bytes = encode_frame(
            2, canonical_seal_payload(root[:16].hex(), 1, root), MAGIC_SEAL
        )
        wal.close()
        self._opened.remove(wal)
        good = bytes(self.raw())
        # Append a record frame numbered 3 right after frame 1 (seal frame 2
        # missing): the shared ordinal space must detect the gap.
        frame3 = encode_frame(3, b"[2]")
        self.write_raw(good + frame3)
        self.assertTrue(self.open_wal().poisoned)
        # Sanity: with the seal frame present, ordinal 3 is accepted.
        self.write_raw(good + seal_bytes + frame3)
        wal2 = self.open_wal()
        self.assertFalse(wal2.poisoned)
        self.assertEqual(wal2.next_seq, 3)


class SealProofDirectionTests(SealFixture):
    def test_sided_path_folds_back_to_sealed_root(self) -> None:
        ledger = Ledger(self.open_wal())
        ledger.submit(1, [10, 20, 30, 40, 50])
        seal = ledger.seal(6)
        for seq in range(1, 6):
            proof: Proof = ledger.proof(seal.seal_id, seq)
            node = proof.leaf
            for side, sibling in proof.steps:
                if side == LEFT:
                    node = inner_digest(sibling, node)
                elif side == RIGHT:
                    node = inner_digest(node, sibling)
                else:
                    self.fail(f"bad direction {side!r}")
            self.assertEqual(node, seal.root)


if __name__ == "__main__":
    unittest.main()
