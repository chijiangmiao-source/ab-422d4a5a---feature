"""Tests for the canonical Merkle construction and proof ordering.

The folding verifier used here is intentionally reimplemented locally
instead of importing :mod:`app.merkle` helpers: an external reviewer only
has the pinned byte rules, and the proof must recompute from those alone.
"""

from __future__ import annotations

import hashlib
import json
import unittest

from app.merkle import (
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    LEAF_PREFIX,
    NODE_PREFIX,
    Proof,
    canonical_record,
    leaf_digest,
    merkle_proof,
    merkle_root,
)


def independent_fold(seq: int, dose: int, path: list[bytes],
                     direction: list[str]) -> bytes:
    """Verifier-side folding using only the documented wire rules."""
    encoded = json.dumps(
        [seq, dose], separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    current = hashlib.sha256(b"\x00" + encoded).digest()
    for sibling, side in zip(path, direction):
        if side == "left":
            current = hashlib.sha256(b"\x01" + current + sibling).digest()
        else:
            current = hashlib.sha256(b"\x01" + sibling + current).digest()
    return current


class CanonicalEncodingTests(unittest.TestCase):
    def test_record_encoding_is_compact_json_array(self) -> None:
        self.assertEqual(canonical_record(3, -17), b"[3,-17]")
        self.assertEqual(canonical_record(1, 2**40), f"[1,{2**40}]".encode())

    def test_leaf_uses_leaf_domain_prefix(self) -> None:
        self.assertEqual(
            leaf_digest(1, 7),
            hashlib.sha256(LEAF_PREFIX + b"[1,7]").digest(),
        )

    def test_leaf_and_node_domains_cannot_collide_encoding(self) -> None:
        # A 32-byte leaf preimage (prefix + [seq,dose]) is far shorter than a
        # 65-byte node preimage; the distinct first byte additionally binds
        # the kind.
        self.assertNotEqual(LEAF_PREFIX, NODE_PREFIX)
        self.assertEqual(LEAF_PREFIX, b"\x00")
        self.assertEqual(NODE_PREFIX, b"\x01")


class RootSplitTests(unittest.TestCase):
    def test_single_record_root_is_its_leaf(self) -> None:
        self.assertEqual(merkle_root([42]), leaf_digest(1, 42))

    def test_two_records_pair(self) -> None:
        expected = hashlib.sha256(
            NODE_PREFIX + leaf_digest(1, 10) + leaf_digest(2, 20)
        ).digest()
        self.assertEqual(merkle_root([10, 20]), expected)

    def test_non_full_tree_splits_power_of_two_to_the_left(self) -> None:
        # 6 leaves must split 4 + 2, not 3 + 3: build the expectation by the
        # same fixed rule but from explicitly grouped subtrees.
        doses = [10, 20, 30, 40, 50, 60]

        def sub(lo: int, hi: int) -> bytes:
            n = hi - lo
            if n == 1:
                return leaf_digest(lo + 1, doses[lo])
            k = 1 << ((n - 1).bit_length() - 1)
            return hashlib.sha256(
                NODE_PREFIX + sub(lo, lo + k) + sub(lo + k, hi)
            ).digest()

        self.assertEqual(merkle_root(doses), sub(0, 6))

    def test_roots_for_split_sizes_1_through_64(self) -> None:
        # Exhaustively pins the split rule across every non-full shape up to
        # a depth of 6 (prefix counts 1..64).
        def reference_root(doses: list[int]) -> bytes:
            leaves = [
                hashlib.sha256(
                    LEAF_PREFIX + canonical_record(i, d)
                ).digest()
                for i, d in enumerate(doses, start=1)
            ]

            def sub(leaves: list[bytes]) -> bytes:
                if len(leaves) == 1:
                    return leaves[0]
                k = 1 << ((len(leaves) - 1).bit_length() - 1)
                return hashlib.sha256(
                    NODE_PREFIX + sub(leaves[:k]) + sub(leaves[k:])
                ).digest()

            return sub(leaves)

        for n in range(1, 65):
            doses = [(i * 37 - 11) % 9973 for i in range(1, n + 1)]
            self.assertEqual(merkle_root(doses), reference_root(doses), n)

    def test_empty_prefix_rejected(self) -> None:
        with self.assertRaises(ValueError):
            merkle_root([])


class ProofTests(unittest.TestCase):
    def _check(self, doses: list[int], seq: int) -> Proof:
        root = merkle_root(doses)
        proof = merkle_proof(doses, seq)
        self.assertEqual(proof.seq, seq)
        self.assertEqual(proof.dose, doses[seq - 1])
        self.assertEqual(proof.root, root)
        self.assertEqual(proof.count, len(doses))
        self.assertEqual(len(proof.path), len(proof.direction))
        self.assertTrue(
            all(d in (DIRECTION_LEFT, DIRECTION_RIGHT)
                for d in proof.direction)
        )
        # Independent recomputation must recover the exact returned root.
        self.assertEqual(
            independent_fold(proof.seq, proof.dose, proof.path,
                             proof.direction),
            root,
        )
        return proof

    def test_every_leaf_of_every_prefix_size_proves(self) -> None:
        for n in range(1, 33):
            doses = [((i * 101) % 7919) - 3000 for i in range(1, n + 1)]
            for seq in range(1, n + 1):
                self._check(doses, seq)

    def test_proof_path_order_is_leaf_to_root(self) -> None:
        # With 6 leaves (split 4+2) a proof for leaf 1 first meets its leaf
        # sibling (leaf 2); the root-side sibling (the right 2-leaf subtree)
        # must come last.
        doses = [1, 2, 3, 4, 5, 6]
        proof = self._check(doses, 1)
        self.assertEqual(
            proof.path[0], leaf_digest(2, 2),
            "first sibling on the path must be the leaf neighbour",
        )
        right_subroot = hashlib.sha256(
            NODE_PREFIX + leaf_digest(5, 5) + leaf_digest(6, 6)
        ).digest()
        self.assertEqual(proof.path[-1], right_subroot)
        self.assertEqual(proof.direction[0], DIRECTION_LEFT)
        self.assertEqual(proof.direction[-1], DIRECTION_LEFT)

    def test_single_record_proof_has_empty_path(self) -> None:
        proof = self._check([42], 1)
        self.assertEqual(proof.path, [])
        self.assertEqual(proof.direction, [])

    def test_proof_directions_are_explicit_both_sides(self) -> None:
        # 5 leaves split 4+1: leaf 5 is a right child at the root and a left
        # child nowhere; leaf 1 is left at every level.
        doses = [1, 2, 3, 4, 5]
        p5 = self._check(doses, 5)
        self.assertIn(DIRECTION_RIGHT, p5.direction)
        self.assertEqual(p5.direction[-1], DIRECTION_RIGHT)
        p1 = self._check(doses, 1)
        self.assertTrue(
            all(d == DIRECTION_LEFT for d in p1.direction)
        )

    def test_out_of_range_seq_rejected(self) -> None:
        with self.assertRaises(ValueError):
            merkle_proof([1, 2], 0)
        with self.assertRaises(ValueError):
            merkle_proof([1, 2], 3)
        with self.assertRaises(ValueError):
            merkle_proof([1, 2], True)  # type: ignore[arg-type]

    def test_wrong_dose_does_not_fold_to_root(self) -> None:
        doses = [1, 2, 3, 4]
        proof = merkle_proof(doses, 2)
        forged = independent_fold(2, 999, proof.path, proof.direction)
        self.assertNotEqual(forged, proof.root)

    def test_repeated_proofs_are_byte_identical(self) -> None:
        doses = list(range(1, 11))
        first = merkle_proof(doses, 7)
        second = merkle_proof(doses, 7)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
