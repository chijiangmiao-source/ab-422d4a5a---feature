"""Ledger: serial-number allocation, optimistic concurrency, cursor reads."""

from __future__ import annotations

import threading
from typing import List, NamedTuple, Optional

from .merkle import Proof, merkle_proof, merkle_root, root_from_proof
from .wal import (
    MAX_RECORDS,
    MIN_RECORDS,
    Frame,
    PoisonedError,
    SealFrame,
    WAL,
)

# Dose records are plain integers. The protocol statement says "整数剂量
# 记录" with no stated bound; accept any signed 64-bit-safe JSON integer.
MIN_DOSE = -(2**63)
MAX_DOSE = 2**63 - 1
MAX_LIMIT = 1000


class StaleSequence(Exception):
    """Optimistic-concurrency conflict: expected_seq != next sequence."""


class EmptyPrefix(Exception):
    """A seal was requested before any record existed."""


class UnknownSeal(Exception):
    """The referenced seal id has never been committed."""


class RecordOutsideSeal(Exception):
    """The record seq is not part of the sealed prefix."""


class Record(NamedTuple):
    seq: int
    dose: int


class Page(NamedTuple):
    records: List[Record]
    next_cursor: int  # cursor to resume with; equal to last seq when exhausted


class Seal(NamedTuple):
    seal_id: str
    count: int
    next_seq: int  # caller-observed next seq at seal creation == count + 1
    root: str  # lowercase hex Merkle root of the fixed prefix


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

    # -- sealing ----------------------------------------------------------
    def seal(self, expected_seq: int) -> Seal:
        """Freeze the prefix ending just before ``expected_seq``.

        ``expected_seq`` is the next sequence number the caller observed and
        must equal the live head under the same optimistic-concurrency rule
        as :meth:`submit`.  The seal fixes *all* records currently in append
        order (``count == expected_seq - 1``) together with their canonical
        Merkle root and a stable id; it becomes referenceable only after its
        whole WAL frame is durable.  No record sequence numbers are consumed.
        """
        if isinstance(expected_seq, bool) or not isinstance(expected_seq, int):
            raise ValueError("expected_seq must be a positive integer")
        if expected_seq < 1:
            raise ValueError("expected_seq must be a positive integer")
        with self._lock:
            if self._wal.poisoned:
                raise PoisonedError(self._wal.poison_reason or "poisoned")
            current = self._wal.next_seq
            if expected_seq != current:
                raise StaleSequence(
                    f"expected {expected_seq}, actual {current}"
                )
            count = current - 1
            if count == 0:
                raise EmptyPrefix("cannot seal an empty prefix")
            records = self._wal.prefix_records(count)
            root = merkle_root(records)
            sealed = self._wal.append_seal(count, root)
            return Seal(
                seal_id=sealed.seal_id,
                count=count,
                next_seq=current,
                root=sealed.root,
            )

    def proof(self, seal_id: str, seq: int) -> Proof:
        """Return the deterministic leaf-to-root proof for ``seq`` in a seal.

        Raises UnknownSeal if no committed seal carries ``seal_id`` and
        RecordOutsideSeal if ``seq`` is not within the fixed prefix.  The
        proof is recomputed from the canonical prefix on every call, so
        repeated queries for the same seal yield byte-identical results
        without depending on cached tree objects.
        """
        if not isinstance(seal_id, str) or not seal_id:
            raise ValueError("seal_id must be a non-empty string")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise ValueError("seq must be a positive integer")
        with self._lock:
            sealed = self._wal.get_seal(seal_id)
            if sealed is None:
                raise UnknownSeal(f"unknown seal {seal_id!r}")
            if seq > sealed.count:
                raise RecordOutsideSeal(
                    f"record {seq} is outside seal {seal_id} "
                    f"(prefix count {sealed.count})"
                )
            records = self._wal.prefix_records(sealed.count)
            result = merkle_proof(records, seq)
            if result.root.hex() != sealed.root:
                # The on-disk seal passed recovery but memory disagrees; do
                # not hand out a proof that cannot fold back to the root.
                raise PoisonedError(
                    f"in-memory prefix root for seal {seal_id} does not "
                    "match the committed seal"
                )
            return result

    @staticmethod
    def root_from_proof(
        seq: int,
        dose: int,
        path: List[bytes],
        direction: List[str],
    ) -> bytes:
        """Independent leaf-to-root folding; exposed for tests/verifiers."""
        return root_from_proof(seq, dose, path, direction)
