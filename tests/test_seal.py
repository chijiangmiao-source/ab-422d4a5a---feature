"""Tests for seal frames: persistence, prefix-root re-verification, proofs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest

from app.ledger import (
    EmptyPrefix,
    Ledger,
    RecordOutsideSeal,
    StaleSequence,
    UnknownSeal,
)
from app.merkle import LEAF_PREFIX, NODE_PREFIX, canonical_record
from app.wal import (
    DIGEST_LEN,
    HEADER_LEN,
    MAGIC,
    PoisonedError,
    WAL,
    encode_frame,
    seal_payload,
)


def independent_root(doses: list[int]) -> bytes:
    """Pinned-rule root computed without importing app.merkle."""
    leaves = [
        hashlib.sha256(LEAF_PREFIX + canonical_record(i, d)).digest()
        for i, d in enumerate(doses, start=1)
    ]

    def sub(nodes: list[bytes]) -> bytes:
        if len(nodes) == 1:
            return nodes[0]
        k = 1 << ((len(nodes) - 1).bit_length() - 1)
        return hashlib.sha256(
            NODE_PREFIX + sub(nodes[:k]) + sub(nodes[k:])
        ).digest()

    return sub(leaves)


class _SealCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "wal.bin")
        self._wals: list[WAL] = []
        self.ledger = self._open_ledger()

    def tearDown(self) -> None:
        for wal in self._wals:
            wal.close()

    def _open_ledger(self) -> Ledger:
        wal = WAL(self.path)
        self._wals.append(wal)
        return Ledger(wal)

    @property
    def wal(self) -> WAL:
        return self.ledger._wal

    def reopen(self) -> Ledger:
        return self._open_ledger()

    def raw(self) -> bytearray:
        with open(self.path, "rb") as fh:
            return bytearray(fh.read())

    def write_raw(self, data: bytes) -> None:
        with open(self.path, "wb") as fh:
            fh.write(data)


class SealCreationTests(_SealCase):
    def _populate(self) -> None:
        self.ledger.submit(1, [10, 20, 30])
        self.ledger.submit(4, [40, 50])

    def test_seal_fixes_full_prefix_and_returns_durable_summary(self) -> None:
        self._populate()
        seal = self.ledger.seal(6)
        self.assertEqual((seal.seal_id, seal.count, seal.next_seq),
                         ("s1", 5, 6))
        self.assertEqual(seal.root, independent_root([10, 20, 30, 40, 50]).hex())

    def test_seal_requires_observed_next_seq(self) -> None:
        self._populate()
        with self.assertRaises(StaleSequence):
            self.ledger.seal(5)  # head is already 6
        with self.assertRaises(StaleSequence):
            self.ledger.seal(7)
        # A failed seal writes no bytes and consumes no id/ordinal.
        size = os.path.getsize(self.path)
        with self.assertRaises(StaleSequence):
            self.ledger.seal(1)
        self.assertEqual(os.path.getsize(self.path), size)
        seal = self.ledger.seal(6)
        self.assertEqual(seal.seal_id, "s1")

    def test_empty_prefix_cannot_be_sealed(self) -> None:
        with self.assertRaises(EmptyPrefix):
            self.ledger.seal(1)
        # Malformed input is a plain ValueError (HTTP 400).
        with self.assertRaises(ValueError):
            self.ledger.seal(0)
        with self.assertRaises(ValueError):
            self.ledger.seal("6")  # type: ignore[arg-type]

    def test_seal_does_not_consume_record_sequence(self) -> None:
        self._populate()
        self.assertEqual(self.ledger.seal(6).count, 5)
        self.assertEqual(self.ledger.head(), 6)
        frame = self.ledger.submit(6, [60])
        self.assertEqual((frame.seq, self.ledger.head()), (6, 7))

    def test_multiple_seals_share_one_append_stream(self) -> None:
        self._populate()
        s1 = self.ledger.seal(6)
        self.ledger.submit(6, [60, 70])
        s2 = self.ledger.seal(8)
        self.assertEqual((s1.seal_id, s1.count), ("s1", 5))
        self.assertEqual((s2.seal_id, s2.count), ("s2", 7))
        # Physical frames share one numbering: batches 1,2; seals 3,5; the
        # batch after the first seal is physical frame 4.
        self.assertEqual(
            [f.frame_no for f in self.wal.snapshot()], [1, 2, 4])
        self.assertEqual(
            [x.frame_no for x in self.wal.seals_snapshot()], [3, 5])
        self.assertEqual(s1.root, independent_root([10, 20, 30, 40, 50]).hex())
        self.assertEqual(
            s2.root,
            independent_root([10, 20, 30, 40, 50, 60, 70]).hex(),
        )

    def test_seal_payload_on_disk_has_pinned_shape(self) -> None:
        self._populate()
        self.ledger.seal(6)
        data = self.raw()
        # First two frames are batches; the third is the seal object.
        first = HEADER_LEN + len(b"[10,20,30]") + DIGEST_LEN
        second = HEADER_LEN + len(b"[40,50]") + DIGEST_LEN
        off = first + second
        self.assertEqual(bytes(data[off:off + 8]), MAGIC)
        self.assertEqual(int.from_bytes(data[off + 8:off + 16], "big"), 3)
        plen = int.from_bytes(data[off + 16:off + 24], "big")
        payload = json.loads(bytes(data[off + 24:off + 24 + plen]))
        self.assertEqual(
            sorted(payload.keys()), ["count", "id", "root"])
        self.assertEqual(payload["id"], "s1")
        self.assertEqual(payload["count"], 5)


class ProofTests(_SealCase):
    def setUp(self) -> None:
        super().setUp()
        self.ledger.submit(1, [10, 20, 30])
        self.ledger.submit(4, [40, 50, 60])
        self.seal = self.ledger.seal(7)
        self.doses = [10, 20, 30, 40, 50, 60]

    def _fold(self, proof) -> bytes:
        current = hashlib.sha256(
            LEAF_PREFIX + canonical_record(proof.seq, proof.dose)
        ).digest()
        for sibling, side in zip(proof.path, proof.direction):
            if side == "left":
                current = hashlib.sha256(
                    NODE_PREFIX + current + sibling).digest()
            else:
                current = hashlib.sha256(
                    NODE_PREFIX + sibling + current).digest()
        return current

    def test_every_record_proof_folds_back_to_sealed_root(self) -> None:
        for seq in range(1, 7):
            proof = self.ledger.proof(self.seal.seal_id, seq)
            self.assertEqual(proof.dose, self.doses[seq - 1])
            self.assertEqual(self._fold(proof).hex(), self.seal.root)
            self.assertEqual(proof.root.hex(), self.seal.root)

    def test_repeated_queries_return_identical_proofs(self) -> None:
        first = self.ledger.proof("s1", 4)
        second = self.ledger.proof("s1", 4)
        self.assertEqual(first, second)
        # Also identical after records are appended later: the sealed prefix
        # and its proof stay fixed.
        self.ledger.submit(7, [70, 80])
        third = self.ledger.proof("s1", 4)
        self.assertEqual(first, third)

    def test_unknown_seal_and_out_of_prefix_seq(self) -> None:
        with self.assertRaises(UnknownSeal):
            self.ledger.proof("s999", 1)
        with self.assertRaises(RecordOutsideSeal):
            self.ledger.proof("s1", 7)
        with self.assertRaises(ValueError):
            self.ledger.proof("s1", 0)

    def test_older_seal_proof_stays_valid_after_later_seal(self) -> None:
        self.ledger.submit(7, [70])
        s2 = self.ledger.seal(8)
        proof = self.ledger.proof(self.seal.seal_id, 6)
        self.assertEqual(self._fold(proof).hex(), self.seal.root)
        self.assertNotEqual(s2.root, self.seal.root)


class SealRecoveryTests(_SealCase):
    def _build_log(self) -> None:
        self.ledger.submit(1, [10, 20, 30])
        self.ledger.seal(4)
        self.ledger.submit(4, [40, 50])
        self.ledger.seal(6)

    def test_rebuilds_seals_and_rechecks_prefix_roots(self) -> None:
        self._build_log()
        self.wal.close()
        self._wals.clear()
        reopened = self.reopen()
        seals = reopened._wal.seals_snapshot()
        self.assertEqual(
            [(s.seal_id, s.count) for s in seals], [("s1", 3), ("s2", 5)])
        self.assertFalse(reopened.poisoned)
        for seal_id, count in (("s1", 3), ("s2", 5)):
            proof = reopened.proof(seal_id, count)
            folded = Ledger.root_from_proof(
                proof.seq, proof.dose, proof.path, proof.direction)
            self.assertEqual(folded.hex(),
                             reopened._wal.get_seal(seal_id).root)

    def _rewrite_last_seal(self, payload: bytes) -> None:
        self._build_log()
        self.wal.close()
        self._wals.clear()
        data = self.raw()
        # Physical layout: batch(1), seal(2), batch(3), seal(4).
        f1 = HEADER_LEN + len(b"[10,20,30]") + DIGEST_LEN
        s1 = HEADER_LEN + len(seal_payload(1, 3, b"\x00" * 32)) + DIGEST_LEN
        f3 = HEADER_LEN + len(b"[40,50]") + DIGEST_LEN
        cut = f1 + s1 + f3
        self.write_raw(bytes(data[:cut]) + bytes(encode_frame(4, payload)))

    def test_forged_root_in_complete_seal_poisons(self) -> None:
        # A seal frame whose SHA digest is valid but whose root is forged:
        # only the prefix-root cross-check can detect it.
        self._rewrite_last_seal(seal_payload(2, 5, b"\xab" * 32))
        wal = WAL(self.path)
        self._wals.append(wal)
        self.assertTrue(wal.poisoned)
        self.assertIn("prefix root mismatch", wal.poison_reason or "")

    def test_forged_id_poisons(self) -> None:
        self._rewrite_last_seal(seal_payload(9, 5, b"\xab" * 32))
        wal = WAL(self.path)
        self._wals.append(wal)
        self.assertTrue(wal.poisoned)

    def test_seal_claiming_missing_records_poisons(self) -> None:
        self._rewrite_last_seal(seal_payload(2, 500, b"\xab" * 32))
        wal = WAL(self.path)
        self._wals.append(wal)
        self.assertTrue(wal.poisoned)
        self.assertIn("only 5 precede", wal.poison_reason or "")

    def test_poisoned_seal_serves_no_proof_and_no_record(self) -> None:
        self._rewrite_last_seal(seal_payload(2, 5, b"\xab" * 32))
        ledger = self.reopen()
        self.assertTrue(ledger.poisoned)
        with self.assertRaises(PoisonedError):
            ledger.proof("s1", 1)
        with self.assertRaises(PoisonedError):
            ledger.read(0)
        with self.assertRaises(PoisonedError):
            ledger.seal(6)

    def test_torn_seal_frame_at_tail_is_truncated(self) -> None:
        self._build_log()
        self.wal.close()
        self._wals.clear()
        good = self.raw()
        good_size = len(good)
        self.write_raw(bytes(good) + MAGIC + b"\x00\x00")  # partial header
        reopened = self.reopen()
        self.assertFalse(reopened.poisoned)
        self.assertEqual(os.path.getsize(self.path), good_size)
        self.assertEqual(
            [s.seal_id for s in reopened._wal.seals_snapshot()],
            ["s1", "s2"])
        # The log keeps accepting appends; ordinal continuity survives.
        frame = reopened.submit(6, [60])
        self.assertEqual((frame.frame_no, frame.seq), (5, 6))
        s3 = reopened.seal(7)
        self.assertEqual(s3.seal_id, "s3")

    def test_frame_ordinal_gap_after_dropped_seal_poisons(self) -> None:
        self._build_log()
        self.wal.close()
        self._wals.clear()
        data = self.raw()
        f1 = HEADER_LEN + len(b"[10,20,30]") + DIGEST_LEN
        s1 = HEADER_LEN + len(seal_payload(1, 3, b"\x00" * 32)) + DIGEST_LEN
        f3 = HEADER_LEN + len(b"[40,50]") + DIGEST_LEN
        # Drop seal frame #2 but keep frame 3's ordinal; recovery must poison.
        self.write_raw(bytes(data[:f1]) + bytes(data[f1 + s1:]))
        self.assertTrue(self.reopen().poisoned)


if __name__ == "__main__":
    unittest.main()
