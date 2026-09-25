"""Unit tests for the canonical Merkle tree: domain separation, vectors.

These pin the wire-level hash scheme an external verifier reimplements:
leaf prefix 0x00, inner prefix 0x01, canonical record JSON, odd-level
carry-up (no node duplication) and a fixed bottom-up proof path order.
Proof directions name the *sibling's* side ("L" = sibling is left child).
"""

from __future__ import annotations

import hashlib
import unittest

from app.merkle import (
    DIGEST_LEN,
    INNER_PREFIX,
    LEAF_PREFIX,
    LEFT,
    RIGHT,
    Proof,
    ProofStep,
    build_proof,
    canonical_record,
    encode_sibling_path,
    inner_digest,
    leaf_digest,
    merkle_root,
    verify_proof,
)


class CanonicalEncodingTests(unittest.TestCase):
    def test_record_encoding_is_fixed(self) -> None:
        # Keys always serialize dose,seq; never dictated by a dict order.
        self.assertEqual(canonical_record(1, 10), b'{"dose":10,"seq":1}')
        self.assertEqual(canonical_record(7, -3), b'{"dose":-3,"seq":7}')
        raw = b'{"dose":10,"seq":1}'
        self.assertEqual(
            leaf_digest(1, 10),
            hashlib.sha256(LEAF_PREFIX + raw).digest(),
        )

    def test_leaf_inner_domain_separation(self) -> None:
        raw = b'{"dose":10,"seq":1}'
        self.assertNotEqual(
            hashlib.sha256(LEAF_PREFIX + raw).digest(),
            hashlib.sha256(INNER_PREFIX + raw).digest(),
        )
        a, b = leaf_digest(1, 1), leaf_digest(2, 2)
        self.assertEqual(
            inner_digest(a, b),
            hashlib.sha256(INNER_PREFIX + a + b).digest(),
        )

    def test_single_record_tree_is_just_its_leaf(self) -> None:
        self.assertEqual(merkle_root([(1, 10)]), leaf_digest(1, 10))

    def test_empty_prefix_has_a_fixed_nonzero_root(self) -> None:
        self.assertEqual(
            merkle_root([]), hashlib.sha256(INNER_PREFIX).digest()
        )

    def test_record_order_changes_the_root(self) -> None:
        r1 = merkle_root([(1, 10), (2, 20)])
        r2 = merkle_root([(1, 20), (2, 10)])
        self.assertNotEqual(r1, r2)


class MerkleRootVectorTests(unittest.TestCase):
    def test_seven_record_root_vector(self) -> None:
        # Full 4-leaf subtree joined with a carried 2-leaf pair and a leaf
        # carried at one level; pins the odd-level split rule.
        recs = [(i, i * 10) for i in range(1, 8)]
        self.assertEqual(
            merkle_root(recs).hex(),
            "9f9d09d068aa4ad77fbf45e41874fb5de6318d85e75796c848badc1284a18f87",
        )

    def test_full_power_of_two_tree(self) -> None:
        l = [leaf_digest(i, i * 10) for i in range(1, 5)]
        expect = inner_digest(inner_digest(l[0], l[1]),
                              inner_digest(l[2], l[3]))
        self.assertEqual(
            merkle_root([(i, i * 10) for i in range(1, 5)]), expect
        )

    def test_odd_levels_carry_without_duplication(self) -> None:
        expect = inner_digest(
            inner_digest(leaf_digest(1, 10), leaf_digest(2, 20)),
            leaf_digest(3, 30),
        )
        self.assertEqual(
            merkle_root([(1, 10), (2, 20), (3, 30)]), expect
        )


class ProofTests(unittest.TestCase):
    def test_path_order_and_directions_are_fixed(self) -> None:
        recs = [(i, i * 10) for i in range(1, 8)]
        # seq 1 is the leftmost node: every sibling sits on its right.
        p = build_proof(recs, 1)
        chars, blob = encode_sibling_path(p.steps)
        self.assertEqual(chars, "RRR")
        self.assertEqual([s for s, _ in p.steps],
                         [RIGHT, RIGHT, RIGHT])
        self.assertEqual(len(blob), 3 * DIGEST_LEN)

    def test_directions_for_carried_leaf(self) -> None:
        # Five records: seq5's leaf is carried unchanged at the first level
        # (no step there), then joins the 4-leaf subtree as a right child,
        # so its single sibling sits on the left.
        recs = [(i, i * 10) for i in range(1, 6)]
        p5 = build_proof(recs, 5)
        self.assertEqual([s for s, _ in p5.steps], [LEFT])
        self.assertTrue(verify_proof(p5))

    def test_all_indices_verify_at_many_sizes(self) -> None:
        for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 100):
            recs = [(i, i * 10 + 3) for i in range(1, n + 1)]
            root = merkle_root(recs)
            for seq in range(1, n + 1):
                proof = build_proof(recs, seq)
                self.assertTrue(verify_proof(proof), (n, seq))
                self.assertEqual(proof.root, root)
                self.assertEqual(proof.count, n)

    def test_single_record_proof_has_empty_path(self) -> None:
        proof = build_proof([(1, 42)], 1)
        self.assertEqual(proof.steps, [])
        self.assertTrue(verify_proof(proof))
        self.assertEqual(proof.leaf, leaf_digest(1, 42))

    def test_missing_seq_lookup_error(self) -> None:
        with self.assertRaises(LookupError):
            build_proof([(1, 10), (2, 20)], 9)

    def test_tampered_sibling_fails_verification(self) -> None:
        proof = build_proof([(i, i) for i in range(1, 5)], 2)
        bad = Proof(
            seq=proof.seq,
            leaf=proof.leaf,
            steps=[ProofStep(s, b"\x00" * 32) for s, _ in proof.steps],
            root=proof.root,
            count=proof.count,
        )
        self.assertFalse(verify_proof(bad))

    def test_tampered_leaf_fails_verification(self) -> None:
        proof = build_proof([(1, 1), (2, 2)], 1)
        bad = Proof(proof.seq, b"\x11" * 32, proof.steps,
                    proof.root, proof.count)
        self.assertFalse(verify_proof(bad))

    def test_unknown_side_fails(self) -> None:
        proof = build_proof([(1, 1), (2, 2)], 1)
        bad = Proof(proof.seq, proof.leaf,
                    [ProofStep("weird", proof.steps[0].sibling)],
                    proof.root, proof.count)
        self.assertFalse(verify_proof(bad))

    def test_path_serialization_round_trip(self) -> None:
        recs = [(i, i * 10) for i in range(1, 8)]
        for seq in range(1, 8):
            proof = build_proof(recs, seq)
            chars, blob = encode_sibling_path(proof.steps)
            self.assertEqual(len(chars), len(proof.steps))
            steps = [
                (LEFT if c == "L" else RIGHT,
                 blob[i * 32:(i + 1) * 32])
                for i, c in enumerate(chars)
            ]
            self.assertTrue(
                verify_proof(Proof(proof.seq, proof.leaf, steps,
                                   proof.root, proof.count))
            )


if __name__ == "__main__":
    unittest.main()
