"""Ledger: serial-number allocation, optimistic concurrency, seals, reads."""

from __future__ import annotations

import threading
from typing import Dict, List, NamedTuple, Optional, Tuple

from .merkle import (
    Proof,
    build_proof,
    merkle_root,
)
from .wal import MAX_RECORDS, MIN_RECORDS, Frame, PoisonedError, SealFrame, WAL

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


class Seal(NamedTuple):
    """Citable summary of a confirmed record prefix."""

    seal_id: str
    count: int
    root: bytes
    seq: int  # next_seq the caller observed when creating the seal point


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

    # ------------------------------------------------------------------
    # Sealing
    # ------------------------------------------------------------------
    def _prefix_records(self, frames: List[Frame]
                        ) -> List[Tuple[int, int]]:
        """Canonical ``(seq, dose)`` list for every frame, in append order."""
        out: List[Tuple[int, int]] = []
        for frame in frames:
            for i, dose in enumerate(frame.records):
                out.append((frame.seq + i, dose))
        return out

    def seal(self, expected_seq: int) -> Seal:
        """Freeze the prefix currently ending just before ``expected_seq``.

        ``expected_seq`` is the next sequence number the caller observed
        (1-based head). It must equal the current head and must be at least
        2, i.e. at least one confirmed record must exist to seal. The Merkle
        root, record count and seal id are fixed in one seal frame and the
        call returns only after that frame is durably persisted.
        """
        if isinstance(expected_seq, bool) or not isinstance(expected_seq, int):
            raise ValueError("expected_seq must be an integer")
        if expected_seq < 2:
            raise ValueError(
                "expected_seq must be the head of a non-empty prefix (>= 2)"
            )
        with self._lock:
            if self._wal.poisoned:
                raise PoisonedError(self._wal.poison_reason or "poisoned")
            current = self._wal.next_seq
            if expected_seq != current:
                raise StaleSequence(f"expected {expected_seq}, actual {current}")
            records = self._prefix_records(self._wal.snapshot())
            count = len(records)
            root = merkle_root(records)
            # A prefix determines (count, root) and therefore seal_id; sealing
            # the identical prefix again is idempotent (safe retries / several
            # auditors observing the same head): return the persisted seal
            # point without appending a duplicate frame.
            for existing in self._wal.seals():
                if existing.count == count:
                    if existing.root != root:
                        # Append-only history makes this impossible; treat it
                        # as corruption rather than serving a mismatched seal.
                        reason = ("seal idempotency: stored root disagrees "
                                  "with the recomputed prefix root")
                        self._wal.poison(reason)
                        raise PoisonedError(reason)
                    return Seal(seal_id=existing.seal_id,
                                count=existing.count,
                                root=existing.root,
                                seq=current)
            self._wal.append_seal(root[:16].hex(), count, root)
            return Seal(seal_id=root[:16].hex(), count=count, root=root,
                        seq=current)

    def _seal_index(self) -> Dict[str, SealFrame]:
        return {s.seal_id: s for s in self._wal.seals()}

    def get_seal(self, seal_id: str) -> SealFrame:
        """Look up a persisted seal frame by id; KeyError when unknown."""
        if not isinstance(seal_id, str):
            raise ValueError("seal_id must be a hex string")
        return self._seal_index()[seal_id]

    def proof(self, seal_id: str, seq: int) -> Proof:
        """Return the canonical membership proof of ``seq`` under a seal.

        The proof is rebuilt deterministically from records reconstructed in
        WAL append order -- never from JSON object iteration order or other
        transient state. Raises PoisonedError on a poisoned log, KeyError on
        an unknown seal and LookupError when the record is outside the
        sealed prefix.
        """
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise ValueError("seq must be a positive integer")
        with self._lock:
            if self._wal.poisoned:
                raise PoisonedError(self._wal.poison_reason or "poisoned")
            seal = self._seal_index()[seal_id]
            records = self._prefix_records(self._wal.snapshot())
            prefix = records[: seal.count]
            # A proof is only valid against exactly the frozen prefix.
            if len(prefix) != seal.count:
                raise PoisonedError(
                    "sealed prefix is not fully present in the log"
                )
            proof = build_proof(prefix, seq)
            # Independent recomputation must reproduce the persisted root;
            # refuse to return anything otherwise.
            if proof.root != seal.root:
                raise PoisonedError(
                    "recomputed proof root disagrees with the sealed root"
                )
            return proof

    def sealed_record(self, seal_id: str, seq: int) -> Record:
        """Return the leaf record covered by a seal (value fetch)."""
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise ValueError("seq must be a positive integer")
        with self._lock:
            if self._wal.poisoned:
                raise PoisonedError(self._wal.poison_reason or "poisoned")
            seal = self._seal_index()[seal_id]
            if seq > seal.count:
                raise LookupError(
                    f"seq {seq} is beyond the sealed prefix of {seal.count}"
                )
            # Records are numbered 1..count within a prefix.
            for frame in self._wal.snapshot():
                last = frame.seq + len(frame.records) - 1
                if frame.seq <= seq <= last:
                    return Record(seq=seq,
                                  dose=frame.records[seq - frame.seq])
            raise PoisonedError("sealed record missing from the log")
