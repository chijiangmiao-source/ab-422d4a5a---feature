"""Unit tests for WAL framing, torn-tail recovery and corruption isolation."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest

from app.wal import (
    DIGEST_LEN,
    HEADER_LEN,
    MAGIC,
    PoisonedError,
    WAL,
    canonical_payload,
    decode_payload,
    encode_frame,
    recover,
)


class FramingTests(unittest.TestCase):
    def test_canonical_payload_round_trip(self) -> None:
        for recs in ([0], [1, 2, 3], [-5, 10**18, -(2**63)]):
            payload = canonical_payload(recs)
            self.assertEqual(decode_payload(payload), recs)

    def test_frame_layout_and_digest(self) -> None:
        payload = canonical_payload([7, 8])
        frame = encode_frame(1, payload)
        self.assertEqual(frame[:8], MAGIC)
        self.assertEqual(int.from_bytes(frame[8:16], "big"), 1)
        self.assertEqual(
            int.from_bytes(frame[16:24], "big"), len(payload)
        )
        body = frame[: HEADER_LEN + len(payload)]
        self.assertEqual(
            frame[-DIGEST_LEN:], hashlib.sha256(body).digest()
        )

    def test_rejects_non_integer_payload(self) -> None:
        with self.assertRaises(PoisonedError):
            decode_payload(b"[1, true]")
        with self.assertRaises(PoisonedError):
            decode_payload(b"[1.5]")
        with self.assertRaises(PoisonedError):
            decode_payload(b"[]")


class _TempWal(unittest.TestCase):
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

    def assert_reopen_poisoned(self) -> WAL:
        wal = self.open_wal()
        self.assertTrue(wal.poisoned)
        return wal


class AppendRecoveryTests(_TempWal):
    def test_batches_get_continuous_record_sequences(self) -> None:
        wal = self.open_wal()
        f1 = wal.append_batch([10, 20, 30])
        f2 = wal.append_batch([40])
        f3 = wal.append_batch([50, 60])
        self.assertEqual((f1.frame_no, f1.seq), (1, 1))
        self.assertEqual((f2.frame_no, f2.seq), (2, 4))
        self.assertEqual((f3.frame_no, f3.seq), (3, 5))
        self.assertEqual(wal.next_seq, 7)
        wal.close()
        self._opened.remove(wal)

        wal2 = self.open_wal()
        self.assertEqual(wal2.next_seq, 7)
        self.assertEqual(
            [(f.frame_no, f.seq, f.records) for f in wal2.snapshot()],
            [(1, 1, [10, 20, 30]), (2, 4, [40]), (3, 5, [50, 60])],
        )

    def test_torn_header_at_tail_is_truncated(self) -> None:
        wal = self.open_wal()
        wal.append_batch([1, 2])
        wal.close()
        self._opened.remove(wal)
        good_size = os.path.getsize(self.path)
        with open(self.path, "ab") as fh:
            fh.write(MAGIC + b"\x00\x03")  # partial next frame header
        wal2 = self.open_wal()
        self.assertFalse(wal2.poisoned)
        self.assertEqual(wal2.truncated_bytes_on_boot, 10)
        self.assertEqual(os.path.getsize(self.path), good_size)
        self.assertEqual(wal2.next_seq, 3)

    def test_torn_payload_at_tail_is_truncated(self) -> None:
        wal = self.open_wal()
        wal.append_batch([1])
        wal.close()
        self._opened.remove(wal)
        good = self.raw()
        half = encode_frame(2, canonical_payload([9, 8, 7]))[:30]
        self.write_raw(bytes(good) + half)
        result = recover(self.path)
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.truncated_bytes, 30)
        self.assertEqual(self.raw(), good)

    def test_last_frame_missing_one_digest_byte_is_torn_tail(self) -> None:
        wal = self.open_wal()
        wal.append_batch([1])
        wal.append_batch([2, 3])
        wal.close()
        self._opened.remove(wal)
        size = os.path.getsize(self.path)
        with open(self.path, "r+b") as fh:
            fh.truncate(size - 1)
        wal2 = self.open_wal()
        self.assertFalse(wal2.poisoned)
        self.assertEqual(wal2.next_seq, 2)
        # The log is usable again and numbering stays continuous.
        f = wal2.append_batch([40])
        self.assertEqual((f.seq, wal2.next_seq), (2, 3))

    def test_empty_and_absent_file(self) -> None:
        self.assertEqual(recover(self.path).frames, [])
        self.write_raw(b"")
        self.assertEqual(self.open_wal().next_seq, 1)


class CorruptionIsolationTests(_TempWal):
    def _three_batches(self) -> None:
        wal = self.open_wal()
        wal.append_batch([1])
        wal.append_batch([2])
        wal.append_batch([3, 4, 5])
        wal.close()
        self._opened.remove(wal)

    def test_payload_tamper_in_first_frame_poisons(self) -> None:
        self._three_batches()
        data = self.raw()
        data[HEADER_LEN] ^= 0xFF
        self.write_raw(bytes(data))
        wal = self.assert_reopen_poisoned()
        self.assertIn("SHA-256", wal.poison_reason or "")
        with self.assertRaises(PoisonedError):
            wal.snapshot()
        with self.assertRaises(PoisonedError):
            wal.append_batch([9])

    def test_digest_tamper_in_middle_frame_poisons(self) -> None:
        self._three_batches()
        data = self.raw()
        first_frame_len = HEADER_LEN + len(b"[1]") + DIGEST_LEN
        data[first_frame_len + HEADER_LEN + len(b"[2]")] ^= 0x01
        self.write_raw(bytes(data))
        self.assert_reopen_poisoned()

    def test_corrupted_length_with_later_frame_poisons(self) -> None:
        self._three_batches()
        data = self.raw()
        first_frame_len = HEADER_LEN + len(b"[1]") + DIGEST_LEN
        # Inflate frame 2's length field so it logically overruns EOF;
        # frame 3's marker still physically follows.
        off = first_frame_len + 16
        data[off : off + 8] = (9_000_000).to_bytes(8, "big")
        self.write_raw(bytes(data))
        wal = self.assert_reopen_poisoned()
        self.assertIn("mid-log", wal.poison_reason or "")

    def test_dropped_middle_frame_poisons_on_ordinal_gap(self) -> None:
        self._three_batches()
        data = self.raw()
        first = HEADER_LEN + len(b"[1]") + DIGEST_LEN
        second = HEADER_LEN + len(b"[2]") + DIGEST_LEN
        # Physically remove frame 2: frame 3 follows frame 1 but keeps
        # frame_no=3, which must be detected even though its digest is valid.
        self.write_raw(bytes(data[:first]) + bytes(data[first + second :]))
        self.assert_reopen_poisoned()

    def test_bad_magic_poisons(self) -> None:
        self._three_batches()
        data = self.raw()
        data[0:8] = b"XXXXXXXX"
        self.write_raw(bytes(data))
        self.assert_reopen_poisoned()

    def test_zero_length_poisons(self) -> None:
        self._three_batches()
        data = self.raw()
        data[16:24] = (0).to_bytes(8, "big")
        self.write_raw(bytes(data))
        self.assert_reopen_poisoned()

    def test_torn_bytes_before_first_header_removed_not_poisoned(self) -> None:
        # Fewer bytes than a header cannot contain any committed frame.
        self.write_raw(b"\x00" * (HEADER_LEN - 1))
        wal = self.open_wal()
        self.assertFalse(wal.poisoned)
        self.assertEqual(wal.next_seq, 1)

    def test_poisoned_file_is_not_truncated(self) -> None:
        self._three_batches()
        data = self.raw()
        data[HEADER_LEN] ^= 0xFF
        self.write_raw(bytes(data))
        before = os.path.getsize(self.path)
        self.assert_reopen_poisoned()
        self.assertEqual(os.path.getsize(self.path), before)


if __name__ == "__main__":
    unittest.main()
