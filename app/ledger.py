"""Ledger: serial-number allocation, optimistic concurrency, cursor reads."""

from __future__ import annotations

import threading
from typing import List, NamedTuple, Optional

from .wal import MAX_RECORDS, MIN_RECORDS, Frame, PoisonedError, WAL

# Dose records are plain integers. The protocol statement says "整数剂量
# 记录" with no stated bound; accept any signed 64-bit-safe JSON integer.
MIN_DOSE = -(2**63)
MAX_DOSE = 2**63 - 1
MAX_LIMIT = 1000


class StaleSequence(Exception):
    """Optimistic-concurrency conflict: expected_seq != next sequence."""


class Record(NamedTuple):
    seq: int
    dose: int


class Page(NamedTuple):
    records: List[Record]
    next_cursor: int  # cursor to resume with; equal to last seq when exhausted


class Ledger:
    def __init__(self, wal: WAL) -> None:
        self._wal = wal
        self._lock = threading.Lock()

    def head(self) -> int:
        """Next serial number to be assigned (1-based)."""
        return self._wal.next_seq

    @property
    def poisoned(self) -> bool:
        return self._wal.poisoned

    @property
    def poison_reason(self) -> Optional[str]:
        return self._wal.poison_reason

    def submit(self, expected_seq: int, records: List[int]) -> Frame:
        """Commit one batch if and only if ``expected_seq`` equals head.

        On conflict nothing is written -- no bytes, no consumed sequence
        numbers -- and StaleSequence is raised before touching the WAL.
        """
        self._validate_records(records)
        with self._lock:
            if self._wal.poisoned:
                raise PoisonedError(self._wal.poison_reason or "poisoned")
            current = self._wal.next_seq
            if expected_seq != current:
                raise StaleSequence(f"expected {expected_seq}, actual {current}")
            return self._wal.append_batch(records)

    @staticmethod
    def _validate_records(records: List[int]) -> None:
        if not isinstance(records, list):
            raise ValueError("records must be a list")
        if not (MIN_RECORDS <= len(records) <= MAX_RECORDS):
            raise ValueError(
                f"batch size must be between {MIN_RECORDS} and {MAX_RECORDS}"
            )
        for r in records:
            if isinstance(r, bool) or not isinstance(r, int):
                raise ValueError("every record must be an integer")
            if not (MIN_DOSE <= r <= MAX_DOSE):
                raise ValueError("dose integer out of range")

    def read(self, cursor: int, limit: int = MAX_LIMIT) -> Page:
        """Read records with serial numbers strictly greater than ``cursor``.

        Raises PoisonedError if the ledger is poisoned; callers must not be
        shown potentially misleading data from a damaged log.
        """
        if not isinstance(cursor, int) or cursor < 0:
            raise ValueError("cursor must be a non-negative integer")
        if not isinstance(limit, int) or not (1 <= limit <= MAX_LIMIT):
            raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
        frames = self._wal.snapshot()
        out: List[Record] = []
        for frame in frames:
            base = frame.seq
            for i, dose in enumerate(frame.records):
                seq = base + i
                if seq <= cursor:
                    continue
                out.append(Record(seq=seq, dose=dose))
                if len(out) >= limit:
                    return Page(records=out, next_cursor=out[-1].seq)
        next_cursor = out[-1].seq if out else cursor
        return Page(records=out, next_cursor=next_cursor)
