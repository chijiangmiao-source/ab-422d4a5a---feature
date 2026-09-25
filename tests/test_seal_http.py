"""End-to-end HTTP tests for seal creation and proof retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from app.server import build_server


def _request(method: str, url: str, body: object = None) -> tuple[int, dict]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def independent_verify(body: dict) -> bool:
    """Fold the proof JSON purely from its wire fields, return root match."""
    current = hashlib.sha256(
        b"\x00"
        + json.dumps(
            [body["seq"], body["dose"]], separators=(",", ":")
        ).encode("utf-8")
    ).digest()
    for sibling_hex, side in zip(body["path"], body["direction"]):
        sibling = bytes.fromhex(sibling_hex)
        if side == "left":
            current = hashlib.sha256(b"\x01" + current + sibling).digest()
        else:
            current = hashlib.sha256(b"\x01" + sibling + current).digest()
    return current.hex() == body["root"]


class SealHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.wal_path = os.path.join(self.dir, "wal.bin")
        self.httpd: ThreadingHTTPServer = build_server(
            "127.0.0.1", 0, self.wal_path
        )
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def restart(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        self.httpd = build_server("127.0.0.1", 0, self.wal_path)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    def test_seal_then_proof_round_trip(self) -> None:
        # Two batches across the 1..7 range to exercise cross-frame prefixes.
        status, body = _request(
            "POST", f"{self.base}/api/batches",
            {"expected_seq": 1, "records": [10, 20, 30, 40]})
        self.assertEqual(status, 201, body)
        status, body = _request(
            "POST", f"{self.base}/api/batches",
            {"expected_seq": 5, "records": [50, 60, 70]})
        self.assertEqual(status, 201, body)

        status, seal = _request(
            "POST", f"{self.base}/api/seals", {"expected_seq": 8})
        self.assertEqual(status, 201, seal)
        self.assertEqual(
            seal,
            {"status": "sealed", "seal_id": "s1", "count": 7,
             "next_seq": 8, "root": seal["root"]},
        )
        self.assertEqual(len(seal["root"]), 64)

        for seq in range(1, 8):
            status, proof = _request(
                "GET", f"{self.base}/api/seals/s1/proof?seq={seq}")
            self.assertEqual(status, 200, proof)
            self.assertEqual(proof["seal_id"], "s1")
            self.assertEqual(proof["count"], 7)
            self.assertEqual(proof["seq"], seq)
            self.assertEqual(proof["dose"], seq * 10)
            self.assertEqual(len(proof["path"]), len(proof["direction"]))
            self.assertTrue(independent_verify(proof), proof)
            self.assertEqual(proof["root"], seal["root"])
            # Leaf value must equal the leaf-domain hash of the canonical
            # record encoding.
            expect_leaf = hashlib.sha256(
                b"\x00"
                + json.dumps([seq, seq * 10], separators=(",", ":")).encode()
            ).hexdigest()
            self.assertEqual(proof["leaf"], expect_leaf)

    def test_seal_conflict_is_409_and_writes_nothing(self) -> None:
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 1, "records": [1]})
        size = os.path.getsize(self.wal_path)
        status, body = _request(
            "POST", f"{self.base}/api/seals", {"expected_seq": 1})
        self.assertEqual(status, 409, body)
        self.assertEqual(body["error"], "stale_sequence")
        self.assertEqual(body["current_seq"], 2)
        self.assertEqual(os.path.getsize(self.wal_path), size)
        # Retrying at the true head seals as s1.
        status, body = _request(
            "POST", f"{self.base}/api/seals", {"expected_seq": 2})
        self.assertEqual(status, 201, body)
        self.assertEqual(body["seal_id"], "s1")

    def test_empty_prefix_rejected(self) -> None:
        status, body = _request(
            "POST", f"{self.base}/api/seals", {"expected_seq": 1})
        self.assertEqual(status, 400, body)
        self.assertEqual(body["error"], "empty_prefix")

    def test_seal_bad_requests(self) -> None:
        for payload in ({}, {"expected_seq": 0}, {"expected_seq": -2},
                        {"expected_seq": "3"}):
            status, body = _request(
                "POST", f"{self.base}/api/seals", payload)
            self.assertEqual(status, 400, payload)

    def test_proof_404s(self) -> None:
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 1, "records": [1, 2]})
        _request("POST", f"{self.base}/api/seals", {"expected_seq": 3})

        status, body = _request(
            "GET", f"{self.base}/api/seals/nope/proof?seq=1")
        self.assertEqual(status, 404, body)
        self.assertEqual(body["error"], "unknown_seal")

        status, body = _request(
            "GET", f"{self.base}/api/seals/s1/proof?seq=3")
        self.assertEqual(status, 404, body)
        self.assertEqual(body["error"], "record_outside_seal")

    def test_proof_bad_requests(self) -> None:
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 1, "records": [1]})
        _request("POST", f"{self.base}/api/seals", {"expected_seq": 2})
        for query in ("", "?seq=x", "?seq=0", "?seq=-1"):
            status, body = _request(
                "GET", f"{self.base}/api/seals/s1/proof{query}")
            self.assertEqual(status, 400, query)

    def test_proof_is_stable_across_restart_and_later_writes(self) -> None:
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 1, "records": [3, 1, 4, 1, 5, 9]})
        _, seal = _request(
            "POST", f"{self.base}/api/seals", {"expected_seq": 7})
        _, before = _request(
            "GET", f"{self.base}/api/seals/s1/proof?seq=4")

        # Append more records and a second seal; first proof must not move.
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 7, "records": [2, 6]})
        _, seal2 = _request(
            "POST", f"{self.base}/api/seals", {"expected_seq": 9})
        self.restart()

        status, health = _request("GET", f"{self.base}/healthz")
        self.assertEqual(status, 200, health)
        status, after = _request(
            "GET", f"{self.base}/api/seals/s1/proof?seq=4")
        self.assertEqual(status, 200, after)
        self.assertEqual(after, before)
        self.assertTrue(independent_verify(after))
        # The second seal survives the restart too and proves independently.
        status, p2 = _request(
            "GET", f"{self.base}/api/seals/s2/proof?seq=8")
        self.assertEqual(status, 200, p2)
        self.assertEqual(p2["root"], seal2["root"])
        self.assertTrue(independent_verify(p2))

    def test_poisoned_recovery_returns_503_for_seal_and_proof(self) -> None:
        _request("POST", f"{self.base}/api/batches",
                 {"expected_seq": 1, "records": [1, 2, 3]})
        _request("POST", f"{self.base}/api/seals", {"expected_seq": 4})
        self.restart()
        # Tamper with a complete batch byte: the sealed prefix root no longer
        # rebuilds (or the digest fails); seal/proof endpoints must 503.
        data = bytearray(open(self.wal_path, "rb").read())
        data[24] ^= 0xFF
        with open(self.wal_path, "wb") as fh:
            fh.write(data)
        self.restart()

        self.assertEqual(
            _request("POST", f"{self.base}/api/seals",
                     {"expected_seq": 4})[0],
            503,
        )
        status, body = _request(
            "GET", f"{self.base}/api/seals/s1/proof?seq=1")
        self.assertEqual(status, 503, body)


if __name__ == "__main__":
    unittest.main()
