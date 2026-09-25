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
from typing import List, NamedTuple, Optional, Tuple

from .merkle import merkle_root

MAGIC = b"DSL1WAL\n"
HEADER_LEN = 8 + 8 + 8
DIGEST_LEN = 32
# uint64 length field, but keep a generous sanity ceiling so a corrupted
# length field is detected as corruption instead of a giant allocation.
MAX_PAYLOAD_LEN = 64 * 1024 * 1024

MIN_RECORDS = 1
MAX_RECORDS = 32

# Payload kinds. Batch payloads are JSON arrays of integers (the original
# frame kind); seal payloads are JSON objects with a fixed key set.  The
# first payload byte ([ vs {) already distinguishes them; parsing validates
# the shape explicitly so object key order never matters.
SEAL_KEYS = ("id", "count", "root")
ROOT_HEX_LEN = 64


class Frame(NamedTuple):
    frame_no: int  # 1-based batch ordinal
    seq: int  # serial number of the first record in the batch
    records: List[int]


class SealFrame(NamedTuple):
    frame_no: int  # 1-based ordinal of this physical frame
    seal_id: str  # stable identifier, "s" + 1-based seal ordinal
    count: int  # number of records fixed by this seal (prefix length)
    root: str  # lowercase hex SHA-256 canonical Merkle root of the prefix


class PoisonedError(RuntimeError):
    """The WAL is structurally corrupted and may not be appended/read."""


class TruncatedTail(Exception):
    """Raised internally while scanning: the file ends mid-frame."""


def canonical_payload(records: List[int]) -> bytes:
    """Canonical payload for a batch.

    Compact, whitespace-free UTF-8 JSON.  The record list order
    supplied by the caller is preserved (it defines dose order within the
    batch); seal payloads are separate objects (see :func:`seal_payload`).
    """
    return json.dumps(
        records, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def seal_payload(seal_no: int, count: int, root: bytes) -> bytes:
    """Canonical payload for a seal frame: a fixed-shape JSON object.

    Keys are emitted in the pinned order in :data:`SEAL_KEYS`; recovery
    parses JSON into a dict (so on-disk key order is never trusted) and
    re-validates every field.  ``seal_no`` is derived from append position,
    and ``id`` is ``"s" + str(seal_no)``.
    """
    obj = {
        "id": f"s{seal_no}",
        "count": count,
        "root": root.hex(),
    }
    return json.dumps(
        obj, separators=(",", ":"), ensure_ascii=False, sort_keys=False
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
    """Decode a batch payload (JSON array of 1-32 integers)."""
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


def decode_seal_payload(payload: bytes) -> Tuple[str, int, str]:
    """Decode/validate a seal payload, returning ``(id, count, root_hex)``.

    Shape, key set, types and hex encoding are all re-checked; the seal id
    is still cross-checked by the caller against frame append position.
    """
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoisonedError("seal payload is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PoisonedError("seal payload is not a JSON object")
    if tuple(sorted(value.keys())) != tuple(sorted(SEAL_KEYS)):
        raise PoisonedError(
            f"seal payload keys must be exactly {SEAL_KEYS!r}"
        )
    seal_id = value["id"]
    count = value["count"]
    root_hex = value["root"]
    if (
        not isinstance(seal_id, str)
        or not seal_id.startswith("s")
        or not seal_id[1:].isdigit()
        or int(seal_id[1:]) < 1
    ):
        raise PoisonedError("seal payload has invalid id")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise PoisonedError("seal payload count must be a positive integer")
    if not isinstance(root_hex, str) or len(root_hex) != ROOT_HEX_LEN:
        raise PoisonedError("seal payload root must be 64 hex chars")
    try:
        bytes.fromhex(root_hex)
    except ValueError:
        raise PoisonedError("seal payload root is not valid hex")
    if root_hex != root_hex.lower():
        raise PoisonedError("seal payload root must be lowercase hex")
    return seal_id, count, root_hex


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

class RecoveryResult(NamedTuple):
    frames: List[Frame]
    seals: List["SealFrame"]
    truncated_bytes: int = 0  # bytes removed from a torn tail


def _records_prefix(frames: List[Frame], count: int) -> List[int]:
    """The first ``count`` records (by seq) across ``frames``."""
    out: List[int] = []
    for frame in frames:
        take = min(len(frame.records), count - len(out))
        if take > 0:
            out.extend(frame.records[:take])
        if len(out) == count:
            break
    return out


def _scan(data: bytes) -> Tuple[List[Frame], List[SealFrame]]:
    """Parse all frames from ``data``; raise TruncatedTail on a torn tail.

    Both batch frames and seal frames share one physical, monotonically
    numbered append stream (the "same append order").  Every fully
    persistent seal is re-verified here: the prefix it claims is
    reconstructed solely from the batch frames physically before it and
    its canonical Merkle root is recomputed; any mismatch poisons the
    ledger so a tampered-but-"complete" seal can never serve a proof.

    A frame whose declared bounds run past EOF is ambiguous: it may be a
    genuinely torn tail (crashed append), or an earlier frame's length field
    may be corrupted.  The two cases are distinguished by searching the
    remainder of the file for the next frame marker: a torn append is always
    the last thing physically in the file, so if a marker follows the
    over-long frame the damage is mid-log and must poison the ledger.
    """
    frames: List[Frame] = []
    seals: List[SealFrame] = []
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
        physical_no = len(frames) + len(seals) + 1
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
        payload = body[HEADER_LEN:]
        kind_byte = payload[:1]
        if kind_byte == b"[":
            records = decode_payload(payload)
            if not (MIN_RECORDS <= len(records) <= MAX_RECORDS):
                raise PoisonedError(
                    f"frame at offset {start} carries {len(records)} records"
                )
            seq = (
                frames[-1].seq + len(frames[-1].records) if frames else 1
            )
            frames.append(Frame(frame_no=frame_no, seq=seq, records=records))
        elif kind_byte == b"{":
            payload_id, count, root_hex = decode_seal_payload(payload)
            seal_no = len(seals) + 1
            expected_id = f"s{seal_no}"
            if payload_id != expected_id:
                raise PoisonedError(
                    f"seal at offset {start} carries id {payload_id!r}, "
                    f"expected {expected_id!r} from append position"
                )
            total_records = (
                frames[-1].seq + len(frames[-1].records) - 1 if frames else 0
            )
            if count > total_records:
                raise PoisonedError(
                    f"seal {expected_id} at offset {start} fixes {count} "
                    f"records but only {total_records} precede it in the log"
                )
            recomputed = merkle_root(_records_prefix(frames, count)).hex()
            if recomputed != root_hex:
                raise PoisonedError(
                    f"sealed prefix root mismatch for {expected_id} at "
                    f"offset {start}: payload claims {root_hex}, rebuilt "
                    f"prefix root is {recomputed}"
                )
            seals.append(
                SealFrame(
                    frame_no=frame_no,
                    seal_id=expected_id,
                    count=count,
                    root=root_hex,
                )
            )
        else:
            raise PoisonedError(
                f"frame at offset {start} is neither a batch nor a seal "
                f"payload"
            )
        pos = frame_end
    return frames, seals


def recover(path: str) -> RecoveryResult:
    """Open/recover a WAL file.

    Truncates a torn tail in place (and fsyncs).  Raises PoisonedError for
    any corruption that is not a single incomplete frame at EOF, including
    a complete seal frame whose rebuilt prefix root does not match.
    """
    if not os.path.exists(path):
        return RecoveryResult(frames=[], seals=[])
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
        return RecoveryResult(
            frames=frames, seals=seals, truncated_bytes=removed
        )
    return RecoveryResult(frames=frames, seals=seals)


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
    """Append-only handle with a mutex guarding in-memory state and writes.

    Batch frames and seal frames share one physical append stream; physical
    frame ordinals count both kinds.  Seal frames never contribute record
    sequence numbers.
    """

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

    def seals_snapshot(self) -> List[SealFrame]:
        with self._lock:
            if self._poisoned:
                raise PoisonedError(self._poison_reason or "poisoned")
            return list(self._seals)

    def get_seal(self, seal_id: str) -> Optional[SealFrame]:
        """Look a committed seal up by its stable id; None if unknown."""
        with self._lock:
            if self._poisoned:
                raise PoisonedError(self._poison_reason or "poisoned")
            for seal in self._seals:
                if seal.seal_id == seal_id:
                    return seal
            return None

    def prefix_records(self, count: int) -> List[int]:
        """Return the first ``count`` record doses in sequence order."""
        with self._lock:
            if self._poisoned:
                raise PoisonedError(self._poison_reason or "poisoned")
            total = self.next_seq - 1
            if count < 1 or count > total:
                raise ValueError(
                    f"sealed count {count} does not match {total} records"
                )
            return _records_prefix(self._frames, count)

    # -- mutation ---------------------------------------------------------
    def _physical_frame_no(self) -> int:
        return len(self._frames) + len(self._seals) + 1

    def _append_physical(self, frame_bytes: bytes) -> None:
        """One durable frame write; marks the log poisoned on I/O failure."""
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
        """Append one fully sealed batch frame. Caller validates count/range."""
        with self._lock:
            if self._poisoned:
                raise PoisonedError(self._poison_reason or "poisoned")
            frame_no = self._physical_frame_no()
            seq = (
                self._frames[-1].seq + len(self._frames[-1].records)
                if self._frames
                else 1
            )
            frame_bytes = encode_frame(frame_no, canonical_payload(records))
            self._append_physical(frame_bytes)
            frame = Frame(frame_no=frame_no, seq=seq, records=list(records))
            self._frames.append(frame)
            return frame

    def append_seal(self, count: int, root: bytes) -> SealFrame:
        """Persist one seal frame covering the first ``count`` records.

        Caller computes the canonical root of exactly that prefix; recovery
        will recompute and cross-check it on every subsequent startup.
        Returns only after the full frame is written and fsynced.
        """
        with self._lock:
            if self._poisoned:
                raise PoisonedError(self._poison_reason or "poisoned")
            total = self.next_seq - 1
            if not isinstance(count, int) or isinstance(count, bool):
                raise ValueError("count must be an integer")
            if count < 1 or count > total:
                raise ValueError(
                    f"cannot seal {count} records; log contains {total}"
                )
            seal_no = len(self._seals) + 1
            frame_no = self._physical_frame_no()
            frame_bytes = encode_frame(
                frame_no, seal_payload(seal_no, count, root)
            )
            self._append_physical(frame_bytes)
            seal = SealFrame(
                frame_no=frame_no,
                seal_id=f"s{seal_no}",
                count=count,
                root=root.hex(),
            )
            self._seals.append(seal)
            return seal

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except OSError:
                pass
