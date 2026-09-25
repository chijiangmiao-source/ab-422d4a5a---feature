"""Canonical Merkle tree over sealed dose-record prefixes.

Everything an external verifier needs is pinned here, independent of JSON
object traversal order or any live in-memory state:

* **Leaf domain separation** -- a leaf digest is
  ``SHA256(0x00 || canonical_record(seq, dose))``.
* **Inner domain separation** -- an inner digest is
  ``SHA256(0x01 || left_digest || right_digest)``.
* **Canonical record encoding** -- ``{"dose":<int>,"seq":<int>}`` serialized
  with ``separators=(",", ":")``, sorted/ASCII keys, UTF-8. Key order is a
  property of this function, never of a dict's iteration order.
* **Odd/non-full level split** -- when a level contains an odd number of
  nodes, the *last* node is carried up unchanged (no duplication, no
  re-hashing) and only the preceding even prefix is paired; pairing always
  takes nodes in fixed index order (left then right).

With these rules the root of a record prefix is a pure function of the
``(seq, dose)`` sequence, and a proof is verified by folding the returned
sibling hashes together with the returned directions starting from the leaf.
"""

from __future__ import annotations

import hashlib
import json
from typing import List, NamedTuple, Sequence, Tuple

LEAF_PREFIX = b"\x00"
INNER_PREFIX = b"\x01"
DIGEST_LEN = 32

# Direction markers carried in proofs. They name the position of the
# *sibling* at each folding step: LEFT means the sibling is the left child
# (so the proven node is the right child), RIGHT the mirror case. A carried
# (unpaired) node produces no step.
LEFT = "left"
RIGHT = "right"


def canonical_record(seq: int, dose: int) -> bytes:
    """Canonical per-record encoding covered by a leaf digest.

    Compact JSON with keys serialized in one fixed alphabetical order
    (``dose`` before ``seq``); bools are rejected even though Python treats
    them as ints.
    """
    if isinstance(seq, bool) or not isinstance(seq, int):
        raise TypeError("seq must be an integer")
    if isinstance(dose, bool) or not isinstance(dose, int):
        raise TypeError("dose must be an integer")
    return json.dumps(
        {"dose": dose, "seq": seq},
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")


def leaf_digest(seq: int, dose: int) -> bytes:
    return hashlib.sha256(LEAF_PREFIX + canonical_record(seq, dose)).digest()


def inner_digest(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(INNER_PREFIX + left + right).digest()


def _promote(level: Sequence[bytes]) -> List[bytes]:
    """Fold one level: pairs in index order, odd last node carried as-is."""
    nxt: List[bytes] = []
    for i in range(0, len(level) - 1, 2):
        nxt.append(inner_digest(level[i], level[i + 1]))
    if len(level) % 2 == 1:
        nxt.append(level[-1])
    return nxt


def merkle_root(records: Sequence[Tuple[int, int]]) -> bytes:
    """Root of records given as an ordered ``(seq, dose)`` sequence.

    The empty prefix has the fixed root
    ``SHA256(0x01)`` (an inner node over nothing), never ``b"\\x00" * 32``.
    """
    level = [leaf_digest(seq, dose) for seq, dose in records]
    if not level:
        return hashlib.sha256(INNER_PREFIX).digest()
    while len(level) > 1:
        level = _promote(level)
    return level[0]


class ProofStep(NamedTuple):
    """One folding step.

    ``side`` names the position of the **sibling**: ``LEFT`` means the
    sibling is the left child (the proven node is the right child) and
    ``RIGHT`` means the sibling is the right child.
    """

    side: str
    sibling: bytes  # raw 32-byte digest


class Proof(NamedTuple):
    seq: int
    leaf: bytes
    steps: List[ProofStep]  # bottom-up; empty only for a single-record tree
    root: bytes
    count: int  # number of records in the sealed prefix


def build_proof(records: Sequence[Tuple[int, int]], seq: int) -> Proof:
    """Build a membership proof for ``seq`` against the ordered records.

    Raises LookupError if the sequence number is not in the prefix.
    """
    index = -1
    for i, (s, _dose) in enumerate(records):
        if s == seq:
            index = i
            break
    if index < 0:
        raise LookupError(f"seq {seq} is not part of the sealed prefix")

    level = [leaf_digest(s, dose) for s, dose in records]
    if not level:
        raise LookupError("cannot prove against an empty prefix")
    leaf = level[index]
    node_index = index
    steps: List[ProofStep] = []
    while len(level) > 1:
        if node_index % 2 == 0:
            # Proven node is a left child; sibling is on its right. If the
            # node is itself the unpaired carry-up, there is no sibling at
            # this level -- it simply moves up unchanged.
            if node_index + 1 < len(level):
                steps.append(ProofStep(RIGHT, level[node_index + 1]))
        else:
            steps.append(ProofStep(LEFT, level[node_index - 1]))
        level = _promote(level)
        node_index //= 2
    return Proof(
        seq=seq,
        leaf=leaf,
        steps=steps,
        root=level[0],
        count=len(records),
    )


def verify_proof(proof: Proof) -> bool:
    """Independently recompute the root from leaf + sibling path."""
    node = proof.leaf
    for side, sibling in proof.steps:
        if side == LEFT:
            # Sibling sits on the left; the proven node is the right child.
            node = inner_digest(sibling, node)
        elif side == RIGHT:
            # Sibling sits on the right; the proven node is the left child.
            node = inner_digest(node, sibling)
        else:
            return False
        if len(sibling) != DIGEST_LEN:
            return False
    return node == proof.root


def encode_sibling_path(steps: Sequence[ProofStep]) -> Tuple[str, bytes]:
    """Serialize a path into (fixed-order direction string, sibling bytes).

    Directions are one character per step (``L``/``R`` for the proven node's
    side); sibling digests are concatenated in the same bottom-up order.
    Both orders -- and therefore the proof -- are fixed, never derived from
    a dict/memory layout.
    """
    chars = []
    blob = bytearray()
    for side, sibling in steps:
        if side == LEFT:
            chars.append("L")
        elif side == RIGHT:
            chars.append("R")
        else:  # pragma: no cover - all steps come from build_proof
            raise ValueError(f"unknown proof side {side!r}")
        blob.extend(sibling)
    return "".join(chars), bytes(blob)
