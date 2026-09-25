"""Canonical Merkle tree over a sealed prefix of dose records.

Every byte of the construction is pinned so that two independent
implementations must agree; nothing depends on dict iteration order or on
in-memory tree objects:

* **Leaf**  = ``SHA256(0x00 || canonical_record)`` where
  ``canonical_record`` is the compact UTF-8 JSON array ``[seq, dose]``
  (1-based ``seq``, ``separators=(",", ":")``, ``ensure_ascii=False``).
* **Internal node** = ``SHA256(0x01 || left_digest || right_digest)``.
  The two domain-separator bytes make a leaf encodable as an internal
  node only by accident of hash collision.
* A prefix of one record hashes straight to its leaf digest.
* For ``n > 1`` records the tree is split at
  ``k = 2**floor(log2(n - 1))``: the left subtree covers records
  ``1..k`` and the right subtree records ``k+1..n``.  The left subtree
  is therefore always a complete binary tree and the remainder (possibly
  incomplete) sits on the right -- e.g. 6 leaves split 4 + 2.
* A proof is ordered **leaf-to-root**.  Every entry gives the sibling
  digest (``path``) and the side (``direction``) the *current* hash
  occupies in that concatenation (``"left"`` means the current hash is
  the left child; the sibling is on the right).
"""

from __future__ import annotations

import hashlib
import json
from typing import List, NamedTuple, Sequence

# Domain-separator bytes. Pinned constants, never reuse one for the other.
LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"

DIRECTION_LEFT = "left"
DIRECTION_RIGHT = "right"


class Proof(NamedTuple):
    seq: int
    dose: int
    leaf: bytes
    path: List[bytes]  # sibling digests, leaf-to-root order
    direction: List[str]  # side of the current hash at each step
    root: bytes
    count: int


def canonical_record(seq: int, dose: int) -> bytes:
    """Canonical encoding of one record: the JSON array ``[seq, dose]``."""
    return json.dumps(
        [seq, dose], separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def leaf_digest(seq: int, dose: int) -> bytes:
    return hashlib.sha256(LEAF_PREFIX + canonical_record(seq, dose)).digest()


def _node_digest(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


def _split(n: int) -> int:
    """Largest power of two strictly smaller than ``n`` (n >= 2)."""
    return 1 << ((n - 1).bit_length() - 1)


def _leaf_layer(records: Sequence[int]) -> List[bytes]:
    return [leaf_digest(seq, dose) for seq, dose in enumerate(records, start=1)]


def _subroot(leaves: Sequence[bytes], lo: int, hi: int) -> bytes:
    n = hi - lo
    if n == 1:
        return leaves[lo]
    k = _split(n)
    return _node_digest(
        _subroot(leaves, lo, lo + k), _subroot(leaves, lo + k, hi)
    )


def merkle_root(records: Sequence[int]) -> bytes:
    """Root digest of the canonical tree over ``records`` (non-empty)."""
    if not isinstance(records, (list, tuple)):
        records = list(records)
    if len(records) == 0:
        raise ValueError("cannot hash an empty prefix")
    leaves = _leaf_layer(records)
    return _subroot(leaves, 0, len(leaves))


def _build_proof(
    leaves: Sequence[bytes],
    lo: int,
    hi: int,
    idx: int,
    path: List[bytes],
    direction: List[str],
) -> None:
    n = hi - lo
    if n == 1:
        return
    k = _split(n)
    mid = lo + k
    if idx < mid:
        # Current hash is the left child; sibling is the whole right subtree.
        path.append(_subroot(leaves, mid, hi))
        direction.append(DIRECTION_LEFT)
        _build_proof(leaves, lo, mid, idx, path, direction)
    else:
        # Current hash is the right child; sibling is the whole left subtree.
        path.append(_subroot(leaves, lo, mid))
        direction.append(DIRECTION_RIGHT)
        _build_proof(leaves, mid, hi, idx, path, direction)


def merkle_proof(records: Sequence[int], seq: int) -> Proof:
    """Membership proof for 1-based ``seq`` within ``records``.

    The returned root is recomputed from the prefix, so callers can pin it
    against the value stored in the seal frame.
    """
    if not isinstance(records, (list, tuple)):
        records = list(records)
    count = len(records)
    if count == 0:
        raise ValueError("cannot prove against an empty prefix")
    if not isinstance(seq, int) or isinstance(seq, bool) or not (1 <= seq <= count):
        raise ValueError(f"seq must be an integer in 1..{count}")
    leaves = _leaf_layer(records)
    idx = seq - 1
    path: List[bytes] = []
    direction: List[str] = []
    _build_proof(leaves, 0, count, idx, path, direction)
    # Descent collects siblings root-side first; the fixed proof order is
    # leaf-to-root, so reverse both sequences together.
    path.reverse()
    direction.reverse()
    root = _subroot(leaves, 0, count)
    return Proof(
        seq=seq,
        dose=records[idx],
        leaf=leaves[idx],
        path=path,
        direction=direction,
        root=root,
        count=count,
    )


def root_from_proof(
    seq: int,
    dose: int,
    path: Sequence[bytes],
    direction: Sequence[str],
) -> bytes:
    """Independently fold a proof back up to a candidate root."""
    current = leaf_digest(seq, dose)
    if len(path) != len(direction):
        raise ValueError("path and direction must have equal length")
    for sibling, side in zip(path, direction):
        if side == DIRECTION_LEFT:
            current = _node_digest(current, sibling)
        elif side == DIRECTION_RIGHT:
            current = _node_digest(sibling, current)
        else:
            raise ValueError(f"invalid direction {side!r}")
    return current


def verify_proof(
    seq: int,
    dose: int,
    path: Sequence[bytes],
    direction: Sequence[str],
    root: bytes,
) -> bool:
    try:
        return root_from_proof(seq, dose, path, direction) == root
    except ValueError:
        return False
