"""Append-only write-ahead log of length-prefixed, SHA-256 sealed frames.

Two kinds of frames share one physical, strictly increasing frame ordinal;
both use the same framing and durability/corruption rules::

    magic    8 bytes
               record frame: b"DSL1WAL\\n"
               seal frame:   b"DSL1SEAL"
    frame_no 8 bytes  uint64, 1-based ordinal across *all* physical frames
    length   8 bytes  uint64, length of the canonical payload that follows
    payload  N bytes  canonical compact JSON (see canonical_payload)
    sha256  32 bytes  digest of magic + frame_no + length + payload

Record-frame payload: a JSON list of 1-32 integer doses in batch order.
Seal-frame payload::

    {"count":<records in the sealed prefix>,
     "root":"<64 hex sha256 merkle root>",
     "seal_id":"<32 hex seal identifier>"}

Both magic byte sequences can never occur inside the JSON payloads (integer
lists, or lowercase-hex strings plus fixed JSON keys), so they also serve as
unambiguous frame-boundary markers during recovery: if a declared frame runs
past EOF yet another frame marker follows it, the file has mid-log structural
damage rather than a torn tail.

Durability contract
-------------------
A frame is answered as committed/sealed only after a full ``write`` of every
frame byte followed by ``fsync`` of both the file and (best effort) its
parent directory. If the process is killed mid-write the tail frame is
physically incomplete; recovery truncates exactly that tail and reopens.

Recovery cross-checks every seal frame: its ``count`` must equal the number
of records rebuilt before it and its Merkle ``root`` is recomputed from those
records. A complete seal frame that disagrees with the rebuilt prefix is
corruption and poisons the log -- a seal is never served on trust.

Corruption isolation
--------------------
Only a *physically incomplete frame at end of file* is recoverable (it can
only be the frame a crash interrupted). Any other structural problem -- a
complete frame whose digest does not match, a bad magic, an implausible
length, a gap/duplicate in frame ordinals, an invalid payload, or a seal root
that does not recompute -- poisons the ledger. A poisoned log refuses every
append and every read, so callers can never observe misleading data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from typing import List, NamedTuple, Optional, Tuple, Union

from .merkle import merkle_root

MAGIC = b"DSL1WAL\n"  # record frame marker (back/forward compatible name)
MAGIC_RECORD = b"DSL1WAL\n"
MAGIC_SEAL = b"DSL1SEAL"
_ALL_MAGICS = (MAGIC_RECORD, MAGIC_SEAL)
HEADER_LEN = 8 + 8 + 8
DIGEST_LEN = 32
# uint64 length field, but keep a generous sanity ceiling so a corrupted
# length field is detected as corruption instead of a giant allocation.
MAX_PAYLOAD_LEN = 64 * 1024 * 1024

MIN_RECORDS = 1
MAX_RECORDS = 32

_ROOT_HEX_RE = re.compile(r"[0-9a-f]{64}")
_SEAL_ID_RE = re.compile(r"[0-9a-f]{32}")


class Frame(NamedTuple):
    """One committed record batch."""

    frame_no: int  # 1-based physical frame ordinal
    seq: int  # serial number of the first record in the batch
    records: List[int]


class SealFrame(NamedTuple):
    """One committed seal over the record prefix of size ``count``."""

    frame_no: int  # 1-based physical frame ordinal
    seal_id: str
    count: int
    root: bytes


AnyFrame = Union[Frame, SealFrame]


class PoisonedError(RuntimeError):
    """The WAL is structurally corrupted and may not be appended/read."""


class TruncatedTail(Exception):
    """Raised internally while scanning: the file ends mid-frame."""


def canonical_payload(records: List[int]) -> bytes:
    """Canonical payload for a record batch.

    Compact, whitespace-free UTF-8 JSON. The record list order supplied by
    the caller is preserved (it defines dose order within the batch).
    """
    return json.dumps(
        records, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_seal_payload(seal_id: str, count: int, root: bytes) -> bytes:
    """Canonical payload for a seal frame; key order is fixed alphabetically."""
    return json.dumps(
        {"count": count, "root": root.hex(), "seal_id": seal_id},
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")


def encode_frame(frame_no: int, payload: bytes,
                 magic: bytes = MAGIC_RECORD) -> bytes:
    header = (
        magic
        + frame_no.to_bytes(8, "big")
        + len(payload).to_bytes(8, "big")
    )
    body = header + payload
    return body + hashlib.sha256(body).digest()


def decode_payload(payload: bytes) -> List[int]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoisonedError("frame payload is not valid UTF-8 JSON") from exc
    if not isinstance(value, list) or not value:
        raise PoisonedError("frame payload is not a non-empty list")
    for item in value:
        # bool is a subclass of int; reject it explicitly.
        if isinstance(item, bool) or not isinstance(item, int):
            raise PoisonedError("frame payload contains a non-integer record")
    return value


def decode_seal_payload(payload: bytes) -> Tuple[str, int, bytes]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoisonedError("seal payload is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PoisonedError("seal payload is not a JSON object")
    seal_id = value.get("seal_id")
    count = value.get("count")
    root_hex = value.get("root")
    if not isinstance(seal_id, str) or not _SEAL_ID_RE.fullmatch(seal_id):
        raise PoisonedError("seal payload carries an invalid seal_id")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise PoisonedError("seal payload carries an invalid count")
    if not isinstance(root_hex, str) or not _ROOT_HEX_RE.fullmatch(root_hex):
        raise PoisonedError("seal payload carries an invalid root")
    return seal_id, count, bytes.fromhex(root_hex)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

class RecoveryResult(NamedTuple):
    frames: List[Frame]  # record batches, in append order
    seals: List[SealFrame]  # seal frames, in append order
    truncated_bytes: int  # bytes removed from a torn tail


def _scan(data: bytes) -> Tuple[List[Frame], List[SealFrame]]:
    """Parse all physical frames from ``data``; raise TruncatedTail on a tear.

    A frame whose declared bounds run past EOF is ambiguous: it may be a
    genuinely torn tail (crashed append), or an earlier frame's length field
    may be corrupted. The two cases are distinguished by searching the
    remainder of the file for any later frame marker: a torn append is always
    the last thing physically in the file, so if a marker follows the
    over-long frame the damage is mid-log and must poison the ledger.
    """
    frames: List[Frame] = []
    seals: List[SealFrame] = []
    records_so_far: List[Tuple[int, int]] = []  # (seq, dose)
    pos = 0
    size = len(data)
    while pos < size:
        start = pos
        remaining = size - pos
        if remaining < HEADER_LEN:
            # A few trailing header bytes with nothing after them: torn tail.
            raise TruncatedTail(start)
        magic = data[pos : pos + 8]
        if magic not in _ALL_MAGICS:
            raise PoisonedError(
                f"bad frame magic at offset {start}: {magic!r}"
            )
        physical_no = len(frames) + len(seals) + 1
        frame_no = int.from_bytes(data[pos + 8 : pos + 16], "big")
        plen = int.from_bytes(data[pos + 16 : pos + 24], "big")
        if frame_no != physical_no:
            raise PoisonedError(
                f"frame ordinal gap/duplicate at offset {start}: "
                f"got {frame_no}, expected {physical_no}"
            )
        if plen == 0 or plen > MAX_PAYLOAD_LEN:
            raise PoisonedError(
                f"implausible payload length {plen} at offset {start}"
            )
        frame_end = pos + HEADER_LEN + plen + DIGEST_LEN
        if size < frame_end:
            tail = data[pos:size]
            # If any frame marker occurs *inside* the supposedly-missing
            # region it means real frame data follows the damaged frame ->
            # mid-log damage, not a torn tail.
            if any(marker in tail[1:] for marker in _ALL_MAGICS):
                raise PoisonedError(
                    f"frame at offset {start} overruns EOF but a later frame "
                    "marker exists; mid-log structural corruption"
                )
            raise TruncatedTail(start)
        body = data[pos : pos + HEADER_LEN + plen]
        digest = data[frame_end - DIGEST_LEN : frame_end]
        if hashlib.sha256(body).digest() != digest:
            raise PoisonedError(
                f"SHA-256 mismatch on complete frame at offset {start}"
            )
        payload = body[HEADER_LEN:]
        if magic == MAGIC_SEAL:
            seal_id, count, root = decode_seal_payload(payload)
            # Recheck the seal against the prefix rebuilt from record frames.
            # Never trust a persisted root: recompute it.
            if count != len(records_so_far):
                raise PoisonedError(
                    f"seal frame at offset {start} fixes {count} records but "
                    f"{len(records_so_far)} precede it in the log"
                )
            rebuilt = merkle_root(records_so_far)
            if rebuilt != root:
                raise PoisonedError(
                    f"seal frame at offset {start} root does not match the "
                    "rebuilt record prefix"
                )
            expected_id = rebuilt[:16].hex()
            if seal_id != expected_id:
                raise PoisonedError(
                    f"seal frame at offset {start} seal_id does not derive "
                    "from its root"
                )
            seals.append(
                SealFrame(frame_no=frame_no, seal_id=seal_id,
                          count=count, root=root)
            )
        else:
            records = decode_payload(payload)
            if not (MIN_RECORDS <= len(records) <= MAX_RECORDS):
                raise PoisonedError(
                    f"frame at offset {start} carries {len(records)} records"
                )
            seq = records_so_far[-1][0] + 1 if records_so_far else 1
            frames.append(
                Frame(frame_no=frame_no, seq=seq, records=records)
            )
            for i, dose in enumerate(records):
                records_so_far.append((seq + i, dose))
        pos = frame_end
    return frames, seals


def recover(path: str) -> RecoveryResult:
    """Open/recover a WAL file.

    Truncates a torn tail in place (and fsyncs). Raises PoisonedError for
    any corruption that is not a single incomplete frame at EOF, including a
    seal whose root does not recompute from the rebuilt prefix.
    """
    if not os.path.exists(path):
        return RecoveryResult(frames=[], seals=[], truncated_bytes=0)
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        frames, seals = _scan(data)
    except TruncatedTail as torn:
        cut = torn.args[0]
        removed = len(data) - cut
        # Rewrite the file to exactly its valid prefix, durably.
        with open(path, "r+b") as fh:
            fh.truncate(cut)
            fh.flush()
            os.fsync(fh.fileno())
        _fsync_dir(os.path.dirname(path) or ".")
        frames, seals = _scan(data[:cut])
        return RecoveryResult(frames=frames, seals=seals,
                              truncated_bytes=removed)
    return RecoveryResult(frames=frames, seals=seals, truncated_bytes=0)


def _fsync_dir(directory: str) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Some filesystems do not support fsync on directories; the file
        # fsync already covers the frame bytes on Linux ext4/overlayfs.
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Live append log
# ---------------------------------------------------------------------------

class WAL:
    """Append-only handle with a mutex guarding in-memory state and writes."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        try:
            result = recover(path)
        except PoisonedError as exc:
            # Stay open in poisoned mode: the HTTP layer can report 503 on
            # health checks and refuse every append/read instead of crash
            # looping and obscuring the corruption.
            self._frames: List[Frame] = []
            self._seals: List[SealFrame] = []
            self._poisoned = True
            self._poison_reason: Optional[str] = str(exc)
            self.truncated_bytes_on_boot = 0
        else:
            self._frames = list(result.frames)
            self._seals = list(result.seals)
            self._poisoned = False
            self._poison_reason = None
            self.truncated_bytes_on_boot = result.truncated_bytes
        self._fh = open(path, "ab")

    # -- introspection ----------------------------------------------------
    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def poison_reason(self) -> Optional[str]:
        return self._poison_reason

    def _physical_count(self) -> int:
        """Number of physical frames (record + seal) already persisted."""
        return len(self._frames) + len(self._seals)

    @property
    def next_seq(self) -> int:
        """Sequence number assigned to the next record (1-based)."""
        with self._lock:
            if self._frames:
                last = self._frames[-1]
                return last.seq + len(last.records)
            return 1

    def snapshot(self) -> List[Frame]:
        with self._lock:
            if self._poisoned:
                raise PoisonedError(self._poison_reason or "poisoned")
            return list(self._frames)

    def seals(self) -> List[SealFrame]:
        with self._lock:
            if self._poisoned:
                raise PoisonedError(self._poison_reason or "poisoned")
            return list(self._seals)

    def poison(self, reason: str) -> None:
        """Mark the log poisoned (used when an invariant is violated)."""
        with self._lock:
            self._poisoned = True
            self._poison_reason = reason

    # -- mutation ---------------------------------------------------------
    def _append_physical(self, frame_bytes: bytes) -> None:
        """Durably append one fully framed blob or poison the log."""
        try:
            # One write call per frame; either all of it lands or the tail
            # is visibly incomplete after a crash.
            self._fh.write(frame_bytes)
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except OSError as exc:
            # A partial write may exist. Mark poisoned rather than risk
            # appending a second frame over an unknown on-disk state.
            self._poisoned = True
            self._poison_reason = f"write failure: {exc}"
            raise PoisonedError(self._poison_reason) from exc
        _fsync_dir(os.path.dirname(self.path) or ".")

    def append_batch(self, records: List[int]) -> Frame:
        """Append one fully sealed record frame. Caller validates count/range."""
        with self._lock:
            if self._poisoned:
                raise PoisonedError(self._poison_reason or "poisoned")
            frame_no = self._physical_count() + 1
            seq = (
                self._frames[-1].seq + len(self._frames[-1].records)
                if self._frames
                else 1
            )
            frame_bytes = encode_frame(
                frame_no, canonical_payload(records), MAGIC_RECORD
            )
            self._append_physical(frame_bytes)
            frame = Frame(frame_no=frame_no, seq=seq,
                          records=list(records))
            self._frames.append(frame)
            return frame

    def append_seal(self, seal_id: str, count: int,
                    root: bytes) -> SealFrame:
        """Append one fully sealed seal frame. Caller computes the root."""
        with self._lock:
            if self._poisoned:
                raise PoisonedError(self._poison_reason or "poisoned")
            frame_no = self._physical_count() + 1
            frame_bytes = encode_frame(
                frame_no,
                canonical_seal_payload(seal_id, count, root),
                MAGIC_SEAL,
            )
            self._append_physical(frame_bytes)
            seal = SealFrame(frame_no=frame_no, seal_id=seal_id,
                             count=count, root=root)
            self._seals.append(seal)
            return seal

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except OSError:
                pass
