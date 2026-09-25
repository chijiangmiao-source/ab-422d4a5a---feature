"""Append-only write-ahead log of length-prefixed, SHA-256 sealed frames.

Frame layout (all integers big-endian)::

    magic    8 bytes  = b"DSL1WAL\\n"
    frame_no 8 bytes  uint64, 1-based batch ordinal
    length   8 bytes  uint64, length of the canonical payload that follows
    payload  N bytes  canonical JSON (see :func:`canonical_payload`)
    sha256  32 bytes  digest of magic + frame_no + length + payload

The magic bytes (``D`` ``S`` ``L`` ``1`` ``W`` ``A`` ``L`` ``\\n``) can never
occur inside the JSON payload of an integer list, so they also serve as
unambiguous frame-boundary markers during recovery: if a declared frame runs
past EOF yet another magic marker follows it, the file has mid-log
structural damage rather than a torn tail.

Durability contract
-------------------
A frame is answered as committed only after a full ``write`` of every frame
byte followed by ``fsync`` of both the file and (best effort) its parent
directory.  If the process is killed mid-write the tail frame is physically
incomplete; recovery truncates exactly that tail and reopens the log.

Corruption isolation
--------------------
Only a *physically incomplete frame at end of file* is recoverable (it can
only be the frame a crash interrupted).  Any other structural problem --
a complete frame whose digest does not match, a bad magic, an implausible
length, a gap/duplicate in batch sequence numbers -- poisons the ledger.
A poisoned log refuses every append and every record read, so callers can
never observe a half batch or silently skip damaged data.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import List, NamedTuple, Optional

MAGIC = b"DSL1WAL\n"
HEADER_LEN = 8 + 8 + 8
DIGEST_LEN = 32
# uint64 length field, but keep a generous sanity ceiling so a corrupted
# length field is detected as corruption instead of a giant allocation.
MAX_PAYLOAD_LEN = 64 * 1024 * 1024

MIN_RECORDS = 1
MAX_RECORDS = 32


class Frame(NamedTuple):
    frame_no: int  # 1-based batch ordinal
    seq: int  # serial number of the first record in the batch
    records: List[int]


class PoisonedError(RuntimeError):
    """The WAL is structurally corrupted and may not be appended/read."""


class TruncatedTail(Exception):
    """Raised internally while scanning: the file ends mid-frame."""


def canonical_payload(records: List[int]) -> bytes:
    """Canonical payload for a batch.

    Compact, sorted-key, whitespace-free UTF-8 JSON.  The record list order
    supplied by the caller is preserved (it defines dose order within the
    batch); only object key serialization is canonicalized.
    """
    return json.dumps(
        records, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def encode_frame(frame_no: int, payload: bytes) -> bytes:
    header = (
        MAGIC
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


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

class RecoveryResult(NamedTuple):
    frames: List[Frame]
    truncated_bytes: int  # bytes removed from a torn tail


def _scan(data: bytes) -> List[Frame]:
    """Parse all frames from ``data``; raise TruncatedTail on a torn tail.

    A frame whose declared bounds run past EOF is ambiguous: it may be a
    genuinely torn tail (crashed append), or an earlier frame's length field
    may be corrupted.  The two cases are distinguished by searching the
    remainder of the file for the next frame marker: a torn append is always
    the last thing physically in the file, so if a marker follows the
    over-long frame the damage is mid-log and must poison the ledger.
    """
    frames: List[Frame] = []
    pos = 0
    size = len(data)
    while pos < size:
        start = pos
        remaining = size - pos
        if remaining < HEADER_LEN:
            # A few trailing header bytes with nothing after them: torn tail.
            raise TruncatedTail(start)
        magic = data[pos : pos + 8]
        if magic != MAGIC:
            raise PoisonedError(
                f"bad frame magic at offset {start}: {magic!r}"
            )
        frame_no = int.from_bytes(data[pos + 8 : pos + 16], "big")
        plen = int.from_bytes(data[pos + 16 : pos + 24], "big")
        expected_no = len(frames) + 1
        if frame_no != expected_no:
            raise PoisonedError(
                f"frame ordinal gap/duplicate at offset {start}: "
                f"got {frame_no}, expected {expected_no}"
            )
        if plen == 0 or plen > MAX_PAYLOAD_LEN:
            raise PoisonedError(
                f"implausible payload length {plen} at offset {start}"
            )
        frame_end = pos + HEADER_LEN + plen + DIGEST_LEN
        if size < frame_end:
            tail = data[pos:size]
            # If the marker occurs *inside* the supposedly-missing region it
            # means real frame data follows the damaged frame -> mid-log
            # damage, not a torn tail.
            if MAGIC in tail[1:]:
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
        records = decode_payload(body[HEADER_LEN:])
        if not (MIN_RECORDS <= len(records) <= MAX_RECORDS):
            raise PoisonedError(
                f"frame at offset {start} carries {len(records)} records"
            )
        seq = (
            frames[-1].seq + len(frames[-1].records) if frames else 1
        )
        frames.append(
            Frame(frame_no=frame_no, seq=seq, records=records)
        )
        pos = frame_end
    return frames


def recover(path: str) -> RecoveryResult:
    """Open/recover a WAL file.

    Truncates a torn tail in place (and fsyncs).  Raises PoisonedError for
    any corruption that is not a single incomplete frame at EOF.
    """
    if not os.path.exists(path):
        return RecoveryResult(frames=[], truncated_bytes=0)
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        frames = _scan(data)
    except TruncatedTail as torn:
        cut = torn.args[0]
        removed = len(data) - cut
        # Rewrite the file to exactly its valid prefix, durably.
        with open(path, "r+b") as fh:
            fh.truncate(cut)
            fh.flush()
            os.fsync(fh.fileno())
        _fsync_dir(os.path.dirname(path) or ".")
        frames = _scan(data[:cut])
        return RecoveryResult(frames=frames, truncated_bytes=removed)
    return RecoveryResult(frames=frames, truncated_bytes=0)


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
            self._poisoned = True
            self._poison_reason: Optional[str] = str(exc)
            self.truncated_bytes_on_boot = 0
        else:
            self._frames = list(result.frames)
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

    # -- mutation ---------------------------------------------------------
    def append_batch(self, records: List[int]) -> Frame:
        """Append one fully sealed frame. Caller validates count/range."""
        with self._lock:
            if self._poisoned:
                raise PoisonedError(self._poison_reason or "poisoned")
            frame_no = len(self._frames) + 1
            seq = (
                self._frames[-1].seq + len(self._frames[-1].records)
                if self._frames
                else 1
            )
            frame_bytes = encode_frame(frame_no, canonical_payload(records))
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
            frame = Frame(frame_no=frame_no, seq=seq, records=list(records))
            self._frames.append(frame)
            return frame

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except OSError:
                pass
